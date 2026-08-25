"""Generate LAr events from a trained run directory.

Conditioning (incident energy and, optionally, truth point counts) is read
from the validation tail of a preprocessed cache file, so generated events can
be compared one-to-one with held-out Geant4 events.
"""

import argparse
import os
import sys
import time
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from torch import Tensor, nn

from lardiff import flow_matching as fm
from lardiff import transformer
from lardiff.preprocessing import compose

start = time.perf_counter()


class LArGenerator(nn.Module):
    def __init__(
        self,
        run_dir: str,
        num_timesteps: int = 200,
        compile: bool = False,
        solver: str = "heun",
    ) -> None:
        super().__init__()

        run_params_file = os.path.join(run_dir, "conf.yaml")
        state_dict_file = os.path.join(run_dir, "weights/best.pt")
        trafo_file = os.path.join(run_dir, "preprocessing/trafos.pt")
        self.result_dir = run_dir
        self.num_timesteps = num_timesteps
        self.do_compile = compile

        with open(run_params_file) as f:
            run_params = yaml.load(f, Loader=yaml.FullLoader)

        # v1/v2 runs condition on [E, N]; v3 adds the energy ratio
        self.cond_dim = run_params["model"]["dim_inputs"][2]
        # >1 means the run was trained on several species and its particle
        # embedding has to be fed, or every species decodes as an average
        self.num_particles = run_params["model"].get("num_particles", 1)
        self.__init_model(run_params["model"], state_dict_file, solver=solver)
        self.__init_trafo(run_params["data"], trafo_file)
        self.to(torch.get_default_dtype())
        self.max_points = run_params["data"]["max_num_points"]

    def __init_model(
        self, params: dict[str, Any], state_file: str, solver: str = "heun"
    ) -> None:
        flow_config = params.pop("flow_config") if "flow_config" in params else {}
        flow_config["solver"] = solver
        network = transformer.Transformer(**params)
        state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
        trained_compiled = any("_orig_mod." in key for key in state_dict)
        if trained_compiled and not self.do_compile:
            for k in list(state_dict.keys()):
                if "_orig_mod." in k:
                    new_k = k.replace("_orig_mod.", "")
                    state_dict[new_k] = state_dict.pop(k)
        elif not trained_compiled and self.do_compile:
            for k in list(state_dict.keys()):
                if "network." in k:
                    new_k = k.replace("network.", "network._orig_mod.")
                    state_dict[new_k] = state_dict.pop(k)
        if self.do_compile:
            network = torch.compile(network)
        self.flow = fm.CNF(network, **flow_config)  # type: ignore
        self.flow.load_state_dict(state_dict)

    def __init_trafo(self, params: dict[str, Any], trafo_file: str) -> None:
        self.samples_energy_trafo = compose(params.get("samples_energy_trafo"))
        self.samples_coordinate_trafo = compose(params.get("samples_coordinate_trafo"))
        self.cond_trafo = compose(params.get("cond_trafo"))

        state = torch.load(trafo_file, map_location="cpu", weights_only=True)
        self.samples_energy_trafo.load_state_dict(state["samples_energy_trafo"])
        self.samples_coordinate_trafo.load_state_dict(state["samples_coordinate_trafo"])
        self.cond_trafo.load_state_dict(state["cond_trafo"])

        # The third conditioning channel used to be the ratio R and is now the
        # deposited energy (see lar_data.load_and_prepare).  Feeding one to a
        # model fitted on the other is silently wrong rather than an error, so
        # check: the scaler's fitted mean is log E_dep (~2.3 to 9.2 over the
        # 10 MeV - 10 GeV range) for current runs and log R (~0) for old ones.
        if self.cond_dim > 2:
            mean = state["cond_trafo"]["sub_modules.1.mean"].flatten()[2].item()
            if mean < 1.0:
                raise RuntimeError(
                    "this run was trained with the energy *ratio* as the third "
                    f"conditioning channel (fitted log-mean {mean:.4f}); it "
                    "predates the switch to deposited energy and cannot be "
                    "generated from with this version. Retrain, or check out "
                    "the commit the run was trained on."
                )

    def forward(
        self,
        energies: Tensor,
        num_points: Tensor,
        ratio: Tensor | None = None,
        renormalize: bool = False,
        label: Tensor | None = None,
    ) -> Tensor:
        if self.num_particles > 1 and label is None:
            raise ValueError(
                f"this run was trained on {self.num_particles} species and "
                "conditions on particle type; supply `label`"
            )
        num_points = num_points.clamp(min=1, max=self.max_points)
        device = energies.device
        mask = (
            torch.arange(self.max_points, device=device)[None]
            < num_points[:, None]
        ).unsqueeze(-1)
        cond_parts = [energies, num_points.to(energies.dtype)]
        if self.cond_dim > 2:
            if ratio is None:
                raise ValueError(
                    "this run conditions on the energy ratio; supply `ratio`"
                )
            # the third channel is the deposited energy, not the ratio itself;
            # see the cond_raw comment in lar_data.load_and_prepare
            cond_parts.append(ratio.to(energies.dtype) * energies)
        elif ratio is not None and renormalize is False:
            raise ValueError(
                "this run does not condition on the energy ratio; `ratio` is "
                "only meaningful here together with renormalize=True"
            )
        condition = self.cond_trafo(torch.stack(cond_parts, dim=-1))
        raw_samples = self.flow.sample(
            shape=(condition.shape[0], self.max_points, 4),
            num_timesteps=self.num_timesteps,
            cond=condition,
            mask=mask,
            label=None if label is None else label.to(device),
        )
        samples = torch.zeros_like(raw_samples)
        samples[:, :, :3] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :3])
        samples[:, :, [3]] = self.samples_energy_trafo.inverse(
            raw_samples[:, :, [3]]
        ).clamp_min(0.0)
        # zero out padding and any point whose generated energy collapsed to 0,
        # so that edep > 0 identifies real hits downstream
        mask = mask & (samples[:, :, [3]] > 0)
        samples[~mask.expand(-1, -1, 4)] = 0
        if renormalize:
            if ratio is None:
                raise ValueError("renormalize=True requires `ratio`")
            # enforce the predicted total exactly by scaling every hit in the
            # event; preserves the shape of the cloud and the relative energy
            # ordering, and only rescales the per-hit spectrum by one factor
            current = samples[:, :, 3].sum(dim=1)
            target = ratio.to(samples.dtype) * energies
            scale = torch.where(
                current > 0, target / current.clamp_min(1e-12), torch.ones_like(current)
            )
            samples[:, :, 3] *= scale[:, None]
        return samples


class EmpiricalNSampler:
    """Bootstrap (N, R) from the training set within log-spaced energy bins.

    Pairs are drawn together rather than independently, so the N-R correlation
    survives. This is the training-free baseline the learned global model
    (lardiff.global_model) has to beat: it reproduces the in-sample joint
    exactly but cannot interpolate between bins or extrapolate beyond them.
    """

    def __init__(self, num_bins: int = 40) -> None:
        self.num_bins = num_bins
        self.bin_edges: np.ndarray | None = None
        self.bin_values: list[np.ndarray] = []

    def fit(
        self,
        energies: np.ndarray,
        n_points: np.ndarray,
        ratios: np.ndarray | None = None,
    ) -> None:
        log_e = np.log(energies)
        self.bin_edges = np.linspace(log_e.min(), log_e.max(), self.num_bins + 1)
        index = np.clip(
            np.digitize(log_e, self.bin_edges) - 1, 0, self.num_bins - 1
        )
        if ratios is None:
            ratios = np.full(len(n_points), np.nan)
        pairs = np.stack([n_points.astype(np.float64), ratios], axis=-1)
        self.bin_values = [pairs[index == i] for i in range(self.num_bins)]
        for i in range(self.num_bins):
            if len(self.bin_values[i]) == 0:
                raise ValueError(f"empty energy bin {i}, reduce num_bins")

    def sample(
        self, energies: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.bin_edges is None:
            raise RuntimeError("call fit() first")
        index = np.clip(
            np.digitize(np.log(energies), self.bin_edges) - 1, 0, self.num_bins - 1
        )
        drawn = np.stack(
            [
                self.bin_values[i][rng.integers(len(self.bin_values[i]))]
                for i in index
            ]
        )
        return drawn[:, 0].astype(np.int64), drawn[:, 1]


def print_time(text):
    now = time.perf_counter()
    print(f"[{int(now - start):6d}s]: {text}")
    sys.stdout.flush()


def packed_edep(f, first: int, last: int, max_points: int) -> np.ndarray:
    """Deposited energy per event for a packed cache, matching truncation.

    The packed caches store hits end to end with CSR offsets and no padded
    `points` array, so the per-event sum has to be taken over the offsets.
    Hits are ordered by descending energy, so summing the first
    `max_points` of each event reproduces what the point model is trained on.
    """
    offsets = f["offsets"][first : last + 1].astype(np.int64)
    edep = f["hits"][int(offsets[0]) : int(offsets[-1]), 3]
    base = offsets - offsets[0]
    n = np.minimum(np.diff(offsets), max_points)
    keep = np.repeat(base[:-1], n) + (
        np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    )
    out = np.zeros(last - first, dtype=np.float64)
    np.add.at(out, np.repeat(np.arange(last - first), n), edep[keep])
    return out


def generate(
    generator: LArGenerator,
    energies: Tensor,
    num_points: Tensor,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
    ratios: Tensor | None = None,
    renormalize: bool = False,
    labels: Tensor | None = None,
) -> Tensor:
    if batch_size is None:
        batch_size = energies.shape[0]
    split_energies = torch.split(energies, batch_size, dim=0)
    split_num_points = torch.split(num_points, batch_size, dim=0)
    if ratios is None:
        split_ratios: tuple = (None,) * len(split_energies)
    else:
        split_ratios = torch.split(ratios, batch_size, dim=0)
    if labels is None:
        split_labels: tuple = (None,) * len(split_energies)
    else:
        split_labels = torch.split(labels, batch_size, dim=0)

    generator = generator.to(device)
    generator.eval()
    samples = []
    for i, (energies_l, num_points_l, ratios_l, labels_l) in enumerate(
        zip(split_energies, split_num_points, split_ratios, split_labels)
    ):
        print_time(f"start batch {i:3d}")
        samples_l = generator(
            energies_l.to(device),
            num_points_l.to(device),
            ratio=None if ratios_l is None else ratios_l.to(device),
            renormalize=renormalize,
            label=None if labels_l is None else labels_l.to(device),
        ).cpu()
        samples.append(samples_l)
    samples = torch.cat(samples)
    print_time("generation done")
    return samples


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates new samples")
    parser.add_argument(
        "run_dir",
        help="directory that contains the model's weights and where the generated samples should be saved",
    )
    parser.add_argument(
        "cache_file",
        help="preprocessed cache file with the conditioning information",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=1,
        type=int,
        help="number of samples to generate, taken from the end of the cache file. default: 1",
    )
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="default: 128"
    )
    parser.add_argument("-t", "--num-threads", default=None, type=int)
    parser.add_argument("-d", "--device", default=None, help="device for computations")
    parser.add_argument(
        "--num-timesteps",
        default=200,
        type=int,
        help="number of timesteps for the ODE solver. default: 200",
    )
    parser.add_argument(
        "--solver",
        default="heun",
        type=str,
        help="ODE solver to use during generation. default: heun",
    )
    parser.add_argument(
        "--n-source",
        default="truth",
        choices=["truth", "empirical", "global"],
        help="where the point count (and, for v3 runs, the energy ratio) comes "
        "from: the held-out truth, a bootstrap of the training set, or a "
        "trained global model. default: truth",
    )
    parser.add_argument(
        "--global-model",
        default=None,
        help="run directory of a trained lardiff.global_model, required for "
        "--n-source global",
    )
    parser.add_argument(
        "--pdg",
        default=None,
        type=int,
        help="particle code for the global model; defaults to the cache's "
        "`pdg` attribute",
    )
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="rescale each event's hit energies so the total matches the "
        "conditioned ratio exactly",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="absolute index of the first cache event to condition on. "
             "Default is the tail, data_len - num_samples.  Set it to split a "
             "large sample across parallel jobs, which also means a job that "
             "dies loses only its own chunk",
    )
    parser.add_argument(
        "--out", default=None,
        help="output file. Default is the next free <run_dir>/samplesNN.h5, "
             "which races when several jobs write at once, so parallel chunks "
             "must each name their own",
    )
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = get_args(args)
    print_time("start main")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(parsed_args.seed)
    if parsed_args.num_threads:
        torch.set_num_threads(parsed_args.num_threads)
    print(yaml.dump(vars(parsed_args)), end="")
    if parsed_args.device:
        device = parsed_args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("device:", device)
    sys.stdout.flush()

    generator = LArGenerator(
        run_dir=parsed_args.run_dir,
        num_timesteps=parsed_args.num_timesteps,
        compile=("cuda" in str(device).lower()),
        solver=parsed_args.solver,
    )

    needs_ratio = generator.cond_dim > 2 or parsed_args.renormalize
    with h5py.File(parsed_args.cache_file, "r") as f:
        data_len = f["energy_MeV"].shape[0]
        if parsed_args.start is None:
            first = data_len - parsed_args.num_samples
        else:
            first = parsed_args.start
        last = min(first + parsed_args.num_samples, data_len)
        if first < 0 or first >= data_len:
            raise ValueError(
                f"start {first} outside the cache, which has {data_len} events"
            )
        energies = f["energy_MeV"][first:last]
        n_truth = f["n_points"][first:last].astype(np.int64)
        cache_index = np.arange(first, last, dtype=np.int64)
        packed = "hits" in f
        # truth ratio of the held-out events, from the same (possibly
        # truncated) points the model is trained to reproduce
        r_truth = (
            packed_edep(f, first, last, generator.max_points)
            if packed
            else f["points"][first:last, :, 3].sum(axis=1)
        ).astype(np.float64) / energies
        # a multi-species cache carries one code per event; a single-species
        # one carries it as a file attribute
        if packed:
            pdg_per_event = f["pdg"][first:last].astype(np.int64)
            species = np.asarray(f.attrs["species"], dtype=np.int64)
        else:
            pdg = parsed_args.pdg or int(f.attrs.get("pdg", 0))
            pdg_per_event = np.full(len(energies), pdg, dtype=np.int64)
            species = np.array([pdg], dtype=np.int64)
        if parsed_args.n_source == "empirical":
            with open(os.path.join(parsed_args.run_dir, "conf.yaml")) as cf:
                run_conf = yaml.safe_load(cf)
            val_len = run_conf["data"]["val_len"]
            train_stop = data_len - val_len
            n_sampler = EmpiricalNSampler()
            train_ratio = None
            if needs_ratio:
                train_ratio = (
                    packed_edep(f, 0, train_stop, generator.max_points)
                    if packed
                    else f["points"][:train_stop, :, 3].sum(axis=1)
                ) / f["energy_MeV"][:train_stop]
            n_sampler.fit(
                f["energy_MeV"][:train_stop],
                f["n_points"][:train_stop],
                train_ratio,
            )
            rng = np.random.default_rng(parsed_args.seed)
            n_used, r_used = n_sampler.sample(energies, rng)
        elif parsed_args.n_source == "global":
            if not parsed_args.global_model:
                raise ValueError("--n-source global requires --global-model")
            from lardiff.global_model import GlobalSampler

            sampler = GlobalSampler(parsed_args.global_model, device=device)
            n_used = np.empty(len(energies), dtype=np.int64)
            r_used = np.empty(len(energies), dtype=np.float64)
            for code in np.unique(pdg_per_event):
                sel = pdg_per_event == code
                n_t, r_t = sampler.sample(
                    torch.from_numpy(energies[sel].astype(np.float32)),
                    pdg=int(code),
                    max_num_points=generator.max_points,
                )
                n_used[sel] = n_t.numpy()
                r_used[sel] = r_t.numpy().astype(np.float64)
        else:
            n_used, r_used = n_truth.copy(), r_truth.copy()
    n_used = np.minimum(n_used, generator.max_points)
    if needs_ratio and not np.isfinite(r_used).all():
        raise ValueError("non-finite energy ratios; is the cache missing points?")

    labels = None
    if generator.num_particles > 1:
        if len(species) < generator.num_particles:
            raise ValueError(
                f"the model conditions on {generator.num_particles} species "
                f"but the cache describes {len(species)}; generate from the "
                "cache the model was trained on"
            )
        labels = torch.from_numpy(np.searchsorted(species, pdg_per_event))

    samples = generate(
        generator,
        torch.from_numpy(energies.astype(np.float32)),
        torch.from_numpy(n_used),
        parsed_args.batch_size,
        device,
        ratios=(
            torch.from_numpy(r_used.astype(np.float32)) if needs_ratio else None
        ),
        renormalize=parsed_args.renormalize,
        labels=labels,
    )

    if parsed_args.out:
        file_path = parsed_args.out
        name = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    else:
        for i in range(100):
            name = f"samples{i:02d}"
            file_path = os.path.join(parsed_args.run_dir, name + ".h5")
            if not os.path.exists(file_path):
                break
        else:
            raise RuntimeError("no free sample file name found")

    with h5py.File(file_path, "w") as f:
        f.create_dataset("points", data=samples.numpy())
        f.create_dataset("energy_MeV", data=energies)
        f.create_dataset("n_points_used", data=n_used)
        f.create_dataset("n_points_truth", data=n_truth)
        f.create_dataset("ratio_used", data=r_used)
        f.create_dataset("ratio_truth", data=r_truth)
        f.create_dataset("cache_index", data=cache_index)
        f.create_dataset("pdg", data=pdg_per_event)
        f.attrs["cache_file"] = parsed_args.cache_file
        f.attrs["renormalized"] = parsed_args.renormalize
    with open(
        os.path.join(os.path.dirname(os.path.abspath(file_path)), name + ".yaml"),
        "w",
    ) as f:
        yaml.dump(vars(parsed_args), f)

    print(f"saved to {file_path}")
    print_time("all done")


if __name__ == "__main__":
    main()
