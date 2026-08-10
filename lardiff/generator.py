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

    def forward(self, energies: Tensor, num_points: Tensor) -> Tensor:
        num_points = num_points.clamp(min=1, max=self.max_points)
        device = energies.device
        mask = (
            torch.arange(self.max_points, device=device)[None]
            < num_points[:, None]
        ).unsqueeze(-1)
        condition = self.cond_trafo(
            torch.stack([energies, num_points.to(energies.dtype)], dim=-1)
        )
        raw_samples = self.flow.sample(
            shape=(condition.shape[0], self.max_points, 4),
            num_timesteps=self.num_timesteps,
            cond=condition,
            mask=mask,
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
        return samples


class EmpiricalNSampler:
    """Sample the number of points given the incident energy by bootstrapping
    from the training set within log-spaced energy bins."""

    def __init__(self, num_bins: int = 40) -> None:
        self.num_bins = num_bins
        self.bin_edges: np.ndarray | None = None
        self.bin_values: list[np.ndarray] = []

    def fit(self, energies: np.ndarray, n_points: np.ndarray) -> None:
        log_e = np.log(energies)
        self.bin_edges = np.linspace(log_e.min(), log_e.max(), self.num_bins + 1)
        index = np.clip(
            np.digitize(log_e, self.bin_edges) - 1, 0, self.num_bins - 1
        )
        self.bin_values = [n_points[index == i] for i in range(self.num_bins)]
        for i in range(self.num_bins):
            if len(self.bin_values[i]) == 0:
                raise ValueError(f"empty energy bin {i}, reduce num_bins")

    def sample(self, energies: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.bin_edges is None:
            raise RuntimeError("call fit() first")
        index = np.clip(
            np.digitize(np.log(energies), self.bin_edges) - 1, 0, self.num_bins - 1
        )
        return np.array(
            [rng.choice(self.bin_values[i]) for i in index], dtype=np.int64
        )


def print_time(text):
    now = time.perf_counter()
    print(f"[{int(now - start):6d}s]: {text}")
    sys.stdout.flush()


def generate(
    generator: LArGenerator,
    energies: Tensor,
    num_points: Tensor,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> Tensor:
    if batch_size is None:
        batch_size = energies.shape[0]
    split_energies = torch.split(energies, batch_size, dim=0)
    split_num_points = torch.split(num_points, batch_size, dim=0)

    generator = generator.to(device)
    generator.eval()
    samples = []
    for i, (energies_l, num_points_l) in enumerate(
        zip(split_energies, split_num_points)
    ):
        print_time(f"start batch {i:3d}")
        samples_l = generator(energies_l.to(device), num_points_l.to(device)).cpu()
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
        choices=["truth", "empirical"],
        help="condition on the true number of points or sample it from "
        "the training-set empirical distribution P(N|E). default: truth",
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

    with h5py.File(parsed_args.cache_file, "r") as f:
        data_len = f["energy_MeV"].shape[0]
        first = data_len - parsed_args.num_samples
        if first < 0:
            raise ValueError(
                f"requested {parsed_args.num_samples} samples but cache only "
                f"has {data_len} events"
            )
        energies = f["energy_MeV"][first:]
        n_truth = f["n_points"][first:].astype(np.int64)
        cache_index = np.arange(first, data_len, dtype=np.int64)
        if parsed_args.n_source == "empirical":
            with open(os.path.join(parsed_args.run_dir, "conf.yaml")) as cf:
                run_conf = yaml.safe_load(cf)
            val_len = run_conf["data"]["val_len"]
            train_stop = data_len - val_len
            n_sampler = EmpiricalNSampler()
            n_sampler.fit(
                f["energy_MeV"][:train_stop], f["n_points"][:train_stop]
            )
            rng = np.random.default_rng(parsed_args.seed)
            n_used = n_sampler.sample(energies, rng)
        else:
            n_used = n_truth.copy()
    n_used = np.minimum(n_used, generator.max_points)

    samples = generate(
        generator,
        torch.from_numpy(energies.astype(np.float32)),
        torch.from_numpy(n_used),
        parsed_args.batch_size,
        device,
    )

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
        f.create_dataset("cache_index", data=cache_index)
        f.attrs["cache_file"] = parsed_args.cache_file
    with open(os.path.join(parsed_args.run_dir, name + ".yaml"), "w") as f:
        yaml.dump(vars(parsed_args), f)

    print(f"saved to {file_path}")
    print_time("all done")


if __name__ == "__main__":
    main()
