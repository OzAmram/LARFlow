import os
import time
import warnings

import h5py
import torch
from torch import Tensor

from lardiff.data_loader import DataLoader, DictDataSet, ModelInputDict
from lardiff.preprocessing import Identity, Transformation, compose

__all__ = ["load_cache", "load_and_prepare", "get_data_loaders"]


class CacheDict(dict):
    """Raw per-event tensors read from the preprocessed cache file."""


def get_file_length(path: str) -> int:
    with h5py.File(path, "r") as f:
        return int(f["energy_MeV"].shape[0])


def load_cache(
    path: str,
    *,
    start: int = 0,
    stop: int | None = None,
    max_num_points: int | None = None,
) -> CacheDict:
    with h5py.File(path, "r") as f:
        points = f["points"][start:stop]
        energy = f["energy_MeV"][start:stop]
        n_points = f["n_points"][start:stop]
    # hits are stored sorted by descending edep, so truncation keeps the
    # highest-energy hits
    if max_num_points is not None and max_num_points < points.shape[1]:
        points = points[:, :max_num_points]
        n_points = n_points.clip(max=max_num_points)
    return CacheDict(
        points=torch.from_numpy(points),
        energy=torch.from_numpy(energy),
        n_points=torch.from_numpy(n_points.astype("int64")),
    )


@torch.no_grad()
def initialise_trafos(
    cond: Tensor,
    points: Tensor,
    mask: Tensor,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    cond_trafo: Transformation,
    *,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    if trafos_file is None and world_size > 1:
        raise ValueError(
            "If using distributed training, a trafos_file must be provided to save and load the transformations."
        )
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
    if rank != 0:
        torch.distributed.barrier(device_ids=[local_rank])
    if os.path.isfile(trafos_file):
        if world_size > 1 and rank == 0:
            torch.distributed.barrier(device_ids=[local_rank])
        parameters = torch.load(trafos_file, weights_only=True)
        samples_energy_trafo.load_state_dict(parameters["samples_energy_trafo"])
        samples_coordinate_trafo.load_state_dict(parameters["samples_coordinate_trafo"])
        cond_trafo.load_state_dict(parameters["cond_trafo"])
        print(f"[rank {rank}] Loaded transformations from {trafos_file}")
    else:
        if rank != 0:
            raise RuntimeError(
                "Initialization of transformations is only allowed for rank 0"
            )
        cond_l = cond[:100_000]
        points_l = points[:100_000]
        mask_l = mask[:100_000]
        cond_trafo.fit(cond_l)
        samples_coordinate_trafo.fit(points_l[:, :, :3], mask_l)
        samples_energy_trafo.fit(points_l[:, :, 3], mask_l.squeeze(-1))
        if trafos_file:
            parameters = {
                "samples_energy_trafo": samples_energy_trafo.state_dict(),
                "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
                "cond_trafo": cond_trafo.state_dict(),
            }
            torch.save(parameters, trafos_file)
            print(f"[rank {rank}] Saved transformations to {trafos_file}")
        if world_size > 1:
            time.sleep(5)  # make sure file is on network drive
            torch.distributed.barrier(device_ids=[local_rank])


@torch.no_grad()
def load_and_prepare(
    path: str,
    *,
    samples_energy_trafo: Transformation = Identity(),
    samples_coordinate_trafo: Transformation = Identity(),
    cond_trafo: Transformation = Identity(),
    start: int = 0,
    stop: int | None = None,
    max_num_points: int | None = None,
    do_initialise_trafos: bool = True,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> ModelInputDict:
    data = load_cache(path, start=start, stop=stop, max_num_points=max_num_points)
    points = data["points"]
    # padding rows are exact zeros; every real hit has edep > 0
    mask = points[:, :, [3]] > 0
    # conditioning: [incident energy, number of points, energy ratio].
    # The ratio is a global property of the cloud, which a model acting locally
    # on points reproduces with too wide a spread; supplying it as truth during
    # training turns it from something to discover into something given.
    # It is computed from the (possibly truncated) cache so it is exactly the
    # quantity the model can reproduce.
    e_total = points[:, :, 3].sum(dim=1)
    cond_raw = torch.stack(
        [
            data["energy"],
            data["n_points"].to(points.dtype),
            e_total / data["energy"].clamp_min(1e-6),
        ],
        dim=-1,
    )

    if do_initialise_trafos:
        initialise_trafos(
            cond_raw,
            points,
            mask,
            samples_energy_trafo,
            samples_coordinate_trafo,
            cond_trafo,
            trafos_file=trafos_file,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    x = torch.concat(
        [
            samples_coordinate_trafo(points[:, :, :3]),
            samples_energy_trafo(points[:, :, [3]]),
        ],
        dim=-1,
    )
    # the trafos map raw 0 to nonzero values; restore exact-zero padding
    x[~mask.expand(-1, -1, 4)] = 0.0
    cond = cond_trafo(cond_raw)

    return ModelInputDict(
        x=x,
        cond=cond,
        mask=mask,
        noise=None,
    )


def get_data_loaders(
    config_dataset: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    trafos_file: str = "",
) -> tuple[DataLoader, DataLoader, dict[str, Transformation]]:
    config_dataset = config_dataset.copy()
    data_len = get_file_length(config_dataset["path"])
    if "stop" in config_dataset:
        data_len = min(data_len, config_dataset["stop"])
        del config_dataset["stop"]
    if "val_len" in config_dataset:
        val_len = config_dataset.pop("val_len")
        if val_len > data_len // 2:
            warnings.warn(
                f"val_len {val_len} is larger than 50% of data length {data_len // 2},"
                f" reducing to {data_len // 2}.",
                UserWarning,
            )
            val_len = min(val_len, data_len // 2)
    else:
        val_len = data_len // 10
    split = data_len - val_len
    if "samples_energy_trafo" in config_dataset:
        config_dataset["samples_energy_trafo"] = compose(
            config_dataset["samples_energy_trafo"]
        )
    if "samples_coordinate_trafo" in config_dataset:
        config_dataset["samples_coordinate_trafo"] = compose(
            config_dataset["samples_coordinate_trafo"]
        )
    if "cond_trafo" in config_dataset:
        config_dataset["cond_trafo"] = compose(config_dataset["cond_trafo"])

    start = rank * (split // world_size)
    stop = (rank + 1) * (split // world_size)
    data_train = DictDataSet(
        load_and_prepare(
            **config_dataset,
            start=start,
            stop=stop,
            trafos_file=trafos_file,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
        )
    )
    loader_train = DataLoader(
        data_set=data_train,
        batch_size=batch_size,
        drop_last=(stop - start) > batch_size,
        shuffle=True,
    )
    if rank == 0:
        data_test = DictDataSet(
            load_and_prepare(
                **config_dataset,
                start=split,
                stop=data_len,
                trafos_file=trafos_file,
                do_initialise_trafos=False,
            )
        )
        loader_test = DataLoader(
            data_set=data_test, batch_size=batch_size, drop_last=False, shuffle=False
        )
    else:
        loader_test = DataLoader(
            data_set=DictDataSet(
                ModelInputDict(
                    x=torch.empty(0, 0, 0),
                    cond=torch.empty(0, 0),
                    mask=torch.empty(0, 0, dtype=torch.bool),
                    noise=None,
                )
            ),
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
    trafos = {
        "samples_energy_trafo": config_dataset.get("samples_energy_trafo", Identity()),
        "samples_coordinate_trafo": config_dataset.get(
            "samples_coordinate_trafo", Identity()
        ),
        "cond_trafo": config_dataset.get("cond_trafo", Identity()),
    }
    return loader_train, loader_test, trafos
