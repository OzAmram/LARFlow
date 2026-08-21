import copy
import os
import time
import warnings
from collections.abc import Iterable, Iterator

import h5py
import numpy as np
import torch
from torch import Tensor

from lardiff.data_loader import DataLoader, DataSet, DictDataSet, ModelInputDict
from lardiff.preprocessing import Identity, Transformation, compose

__all__ = [
    "load_cache",
    "load_and_prepare",
    "get_data_loaders",
    "H5DataSet",
    "PackedLoader",
]


class CacheDict(dict):
    """Raw per-event tensors read from the preprocessed cache file."""


def get_file_length(path: str) -> int:
    with h5py.File(path, "r") as f:
        return int(f["energy_MeV"].shape[0])


def holdout_split(data_len: int, holdout_frac: float, val_len: int) -> tuple[int, int]:
    """Boundaries of the train / validation / test regions.

    Returns (split, val_stop).  Training uses logical [0, split); validation
    monitors [split, val_stop); everything from val_stop on is the untouched
    test holdout.  Validation is only a slice of the holdout because scoring
    all of it every epoch would cost as much as a large fraction of the epoch
    itself.  The cache's logical order is a fixed-seed shuffle, so each region
    is a species-balanced random sample.
    """
    n_holdout = int(round(holdout_frac * data_len))
    if n_holdout < val_len:
        # happens when the run is truncated, as --fast-dev-run does; validate
        # on what holdout there is rather than reaching back into training data
        warnings.warn(
            f"holdout of {n_holdout} events is smaller than val_len {val_len};"
            f" validating on the whole holdout instead.",
            UserWarning,
        )
        val_len = n_holdout
    split = data_len - n_holdout
    return split, split + val_len


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


def make_cond_raw(energy: Tensor, n_points: Tensor, e_total: Tensor) -> Tensor:
    """Conditioning: [incident energy, number of points, deposited energy].

    The deposited energy is a global property of the cloud, which a model
    acting locally on points reproduces with too wide a spread; supplying it
    as truth during training turns it from something to discover into
    something given.  It comes from the (possibly truncated) hits so it is
    exactly the quantity the model can reproduce.

    The channel carries the energy rather than the ratio R = E_dep / E_inc
    because every channel goes through the same elementwise Log before a
    per-channel StandardScaler.  log E_dep has the same ~1.6 spread as the
    other two, whereas log R is dominated by the containment atom at R = 1:
    its std is 0.015, so standardizing sent the escape tail to z ~ -16 and
    handed cond_embedding one channel an order of magnitude larger than the
    rest.  No information is lost -- cond_embedding is Linear, so it can form
    log R = log E_dep - log E_inc itself from channels 0 and 2.
    """
    return torch.stack(
        [energy, n_points.to(energy.dtype), e_total.clamp_min(1e-6)], dim=-1
    )


def build_model_input(
    points: Tensor,
    cond_raw: Tensor,
    mask: Tensor,
    *,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    cond_trafo: Transformation,
    label: Tensor | None = None,
) -> ModelInputDict:
    x = torch.concat(
        [
            samples_coordinate_trafo(points[:, :, :3]),
            samples_energy_trafo(points[:, :, [3]]),
        ],
        dim=-1,
    )
    # the trafos map raw 0 to nonzero values; restore exact-zero padding
    x[~mask.expand(-1, -1, 4)] = 0.0
    return ModelInputDict(
        x=x,
        cond=cond_trafo(cond_raw),
        mask=mask,
        noise=None,
        label=label,
    )


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
    cond_raw = make_cond_raw(
        data["energy"], data["n_points"], points[:, :, 3].sum(dim=1)
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

    return build_model_input(
        points,
        cond_raw,
        mask,
        samples_energy_trafo=samples_energy_trafo,
        samples_coordinate_trafo=samples_coordinate_trafo,
        cond_trafo=cond_trafo,
    )


class H5DataSet(DataSet):
    """Serves batches straight from a packed multi-species cache.

    DictDataSet holds the whole padded array in memory, which the
    single-species caches can afford.  All nine species at 8192 points is
    131 GB, so this reads hits as they are needed and pads them then --
    batch_size * 8192 * 4 floats instead of the whole set.  The per-event
    scalars are small enough to keep resident.

    Events are addressed as a contiguous physical range.  That is what makes
    `read_block` possible, and the cache needs no shuffling to justify it: the
    raw file interleaves the species, so physical index correlates with
    species, energy and multiplicity at the 1e-3 level.
    """

    def __init__(
        self,
        path: str,
        start: int,
        stop: int,
        *,
        max_num_points: int,
        samples_energy_trafo: Transformation = Identity(),
        samples_coordinate_trafo: Transformation = Identity(),
        cond_trafo: Transformation = Identity(),
    ) -> None:
        self.path = path
        self.start = int(start)
        self.stop = int(stop)
        self.max_num_points = max_num_points
        self.samples_energy_trafo = samples_energy_trafo
        self.samples_coordinate_trafo = samples_coordinate_trafo
        self.cond_trafo = cond_trafo
        # the loader in this package is a plain iterator with no worker
        # processes, so one handle for the lifetime of the run is safe
        self._file = h5py.File(path, "r")
        self.hits = self._file["hits"]
        self.offsets = self._file["offsets"][:]
        self.n_points_all = self._file["n_points"][:]
        self.energy_all = self._file["energy_MeV"][:]
        self.label_all = self._file["label"][:]

    def __len__(self) -> int:
        return self.stop - self.start

    def pad(self, flat, offsets_local, n_points):
        """Scatter packed hits into a zero-padded (events, max_points, 4)."""
        points = np.zeros((len(n_points), self.max_num_points, 4), dtype=np.float32)
        col = np.arange(int(n_points.sum())) - np.repeat(
            np.cumsum(n_points) - n_points, n_points
        )
        row = np.repeat(np.arange(len(n_points)), n_points)
        # hits are stored by descending edep, so the first n of an event are
        # the ones truncation keeps
        points[row, col] = flat[np.repeat(offsets_local, n_points) + col]
        return points

    def read_block(self, first: int, last: int):
        """Events [first, last) of this range, read with a single call.

        A scattered read costs ~50 ms per event on the parallel file system
        against ~0.02 ms when the same bytes come back in one request, so
        everything that iterates this dataset should come through here.
        """
        lo, hi = self.start + first, self.start + last
        a, b = int(self.offsets[lo]), int(self.offsets[hi])
        flat = self.hits[a:b]
        offsets_local = (self.offsets[lo:hi] - a).astype(np.int64)
        n_points = np.minimum(
            np.diff(self.offsets[lo : hi + 1]), self.max_num_points
        ).astype(np.int64)
        return self.pad(flat, offsets_local, n_points), np.arange(lo, hi)

    def assemble(self, points, ev) -> ModelInputDict:
        points = torch.from_numpy(points)
        cond_raw = make_cond_raw(
            torch.from_numpy(self.energy_all[ev].astype("float32")),
            torch.from_numpy(
                np.minimum(self.n_points_all[ev], self.max_num_points).astype("int64")
            ),
            points[:, :, 3].sum(dim=1),
        )
        return build_model_input(
            points,
            cond_raw,
            points[:, :, [3]] > 0,
            samples_energy_trafo=self.samples_energy_trafo,
            samples_coordinate_trafo=self.samples_coordinate_trafo,
            cond_trafo=self.cond_trafo,
            label=torch.from_numpy(self.label_all[ev].astype("int64")),
        )

    def __getitem__(self, index) -> ModelInputDict:
        """Arbitrary indices into the range; scattered, so keep the count small."""
        if isinstance(index, torch.Tensor):
            index = index.numpy()
        ev = self.start + np.atleast_1d(np.asarray(index))
        p = self.max_num_points
        points = np.zeros((len(ev), p, 4), dtype=np.float32)
        for k in np.argsort(ev):
            e = int(ev[k])
            a = int(self.offsets[e])
            n = min(int(self.offsets[e + 1]) - a, p)
            points[k, :n] = self.hits[a : a + n]
        return self.assemble(points, ev)


class PackedLoader(Iterable[ModelInputDict]):
    """Batches a packed cache by reading contiguous blocks of events.

    DataLoader draws a fresh permutation over the whole split, which for an
    out-of-core dataset means every batch is a scatter of single-event reads.
    Measured on this cache that is ~1.5 s per batch of 64, comparable to the
    training step it is supposed to be feeding.  Reading `block_events`
    consecutive events in one request instead costs ~20 ms for 2048 events,
    and shuffling within that buffer plus shuffling the block order supplies
    the randomness.  It is sound here because the cache's physical order is
    already unordered in species, energy and multiplicity.

    `block_events` must be a multiple of `batch_size` so that batches never
    straddle two blocks.
    """

    def __init__(
        self,
        data_set: H5DataSet,
        batch_size: int,
        block_events: int = 2048,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> None:
        if block_events % batch_size:
            block_events = max(1, block_events // batch_size) * batch_size
        self.data_set = data_set
        self.batch_size = batch_size
        self.block_events = block_events
        self.drop_last = drop_last
        self.shuffle = shuffle

    def block_bounds(self) -> list[tuple[int, int]]:
        n = len(self.data_set)
        return [
            (i, min(i + self.block_events, n))
            for i in range(0, n, self.block_events)
        ]

    def __len__(self) -> int:
        total = 0
        for first, last in self.block_bounds():
            size = last - first
            if self.drop_last:
                total += size // self.batch_size
            else:
                total += (size + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator[ModelInputDict]:
        blocks = self.block_bounds()
        if self.shuffle:
            blocks = [blocks[i] for i in torch.randperm(len(blocks)).tolist()]
        for first, last in blocks:
            points, ev = self.data_set.read_block(first, last)
            size = last - first
            order = (
                torch.randperm(size).numpy()
                if self.shuffle
                else np.arange(size)
            )
            stop = (
                size // self.batch_size * self.batch_size
                if self.drop_last
                else size
            )
            for s in range(0, stop, self.batch_size):
                sel = order[s : s + self.batch_size]
                yield self.data_set.assemble(points[sel], ev[sel])


def is_packed(path: str) -> bool:
    with h5py.File(path, "r") as f:
        return "hits" in f


@torch.no_grad()
def fit_trafos_packed(
    path: str,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    cond_trafo: Transformation,
    *,
    n_events: int = 100_000,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> None:
    """Fit the transformations on a physical prefix of a packed cache.

    The scalers reduce over every axis their `shape` marks with a 1, so the
    hits can be handed over packed as a single (1, H, ...) pseudo-batch -- no
    padded array is ever built, and the padding mask is unnecessary because
    every stored hit is real.  The raw file interleaves the species, so a
    physical prefix is already species-balanced.
    """
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
        return
    if rank != 0:
        raise RuntimeError("Initialization of transformations is only allowed for rank 0")

    with h5py.File(path, "r") as f:
        n_events = min(n_events, len(f["n_points"]))
        offsets = f["offsets"][: n_events + 1]
        hits = torch.from_numpy(f["hits"][: int(offsets[-1])])
        n_points = torch.from_numpy(f["n_points"][:n_events].astype("int64"))
        energy = torch.from_numpy(f["energy_MeV"][:n_events].astype("float32"))
    # per-event deposited energy from the packed hits
    e_total = torch.zeros(n_events, dtype=torch.float32)
    e_total.index_add_(
        0,
        torch.repeat_interleave(torch.arange(n_events), torch.from_numpy(np.diff(offsets))),
        hits[:, 3],
    )
    cond_trafo.fit(make_cond_raw(energy, n_points, e_total))
    samples_coordinate_trafo.fit(hits[None, :, :3], torch.ones(1, len(hits), 1, dtype=torch.bool))
    samples_energy_trafo.fit(hits[None, :, 3], torch.ones(1, len(hits), dtype=torch.bool))
    if trafos_file:
        torch.save(
            {
                "samples_energy_trafo": samples_energy_trafo.state_dict(),
                "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
                "cond_trafo": cond_trafo.state_dict(),
            },
            trafos_file,
        )
        print(f"[rank {rank}] Saved transformations to {trafos_file}")
    if world_size > 1:
        time.sleep(5)  # make sure file is on network drive
        torch.distributed.barrier(device_ids=[local_rank])


def get_data_loaders(
    config_dataset: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    trafos_file: str = "",
) -> tuple[DataLoader, DataLoader, dict[str, Transformation]]:
    config_dataset = config_dataset.copy()
    config_dataset.setdefault("holdout_frac", 0.0)
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
    if not is_packed(config_dataset["path"]):
        config_dataset.pop("holdout_frac")
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

    if is_packed(config_dataset["path"]):
        return get_data_loaders_packed(
            config_dataset,
            batch_size,
            data_len=data_len,
            val_len=val_len,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            trafos_file=trafos_file,
        )

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


def get_data_loaders_packed(
    config_dataset: dict,
    batch_size: int,
    *,
    data_len: int,
    val_len: int,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    trafos_file: str = "",
) -> tuple[PackedLoader, PackedLoader, dict[str, Transformation]]:
    """Loaders over a packed multi-species cache, with a real test holdout.

    The dense path keeps only a validation tail; here a `holdout_frac` of the
    events is withheld from training entirely, validation monitors the first
    `val_len` of it, and the remainder is never touched during the run so it
    is available for generation and evaluation afterwards.  The split is by
    physical position, which is already a random ordering of the events.
    """
    path = config_dataset["path"]
    max_num_points = config_dataset["max_num_points"]
    split, val_stop = holdout_split(data_len, config_dataset["holdout_frac"], val_len)
    trafos = {
        "samples_energy_trafo": config_dataset.get("samples_energy_trafo", Identity()),
        "samples_coordinate_trafo": config_dataset.get(
            "samples_coordinate_trafo", Identity()
        ),
        "cond_trafo": config_dataset.get("cond_trafo", Identity()),
    }
    fit_trafos_packed(
        path,
        **trafos,
        trafos_file=trafos_file,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if rank == 0:
        print(
            f"[rank {rank}] {data_len} events: {split} train, "
            f"{val_stop - split} validation, {data_len - val_stop} test holdout "
            f"({100 * config_dataset['holdout_frac']:.0f}% withheld)"
        )

    # Trainer moves self.trafos onto the GPU, and nn.Module.to is in place, so
    # handing these same objects to the dataset would have it transforming CPU
    # batches with GPU buffers.  They are already fitted; give the dataset its
    # own CPU copies.
    ds_trafos = copy.deepcopy(trafos)

    per_rank = split // world_size
    loader_train = PackedLoader(
        H5DataSet(
            path,
            rank * per_rank,
            (rank + 1) * per_rank,
            max_num_points=max_num_points,
            **ds_trafos,
        ),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )
    # only rank 0 validates, matching the dense path
    val_start, val_end = (split, val_stop) if rank == 0 else (split, split)
    loader_test = PackedLoader(
        H5DataSet(
            path, val_start, val_end, max_num_points=max_num_points, **ds_trafos
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    return loader_train, loader_test, trafos
