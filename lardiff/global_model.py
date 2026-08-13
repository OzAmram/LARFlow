"""Conditional flow matching for the per-event global quantities.

Diffusion/flow models over point clouds act locally on points, so an aggregate
of the whole cloud -- the fraction of the incident energy that ends up
deposited -- is only reproduced as well as the sum of many local decisions
allows.  In practice the mean comes out right and the *spread* comes out too
wide (see the electron response in the README).

This module learns that global structure directly instead: a small flow over

    (log N, log R)      N = hit count, R = E_deposited / E_incident

conditioned on the incident energy and the particle type.  N and R are
correlated, so they are modelled jointly rather than by two independent heads.
The point-cloud model then takes R as extra conditioning, which turns "produce
the right total" from something it must discover into something it is told.

Trained on the globals cache from scripts/preprocess_globals.py, which covers
every species in the raw file.  Minutes on one GPU.

Usage:
    python -m lardiff.global_model <globals.h5> --out <dir> [--pdg ...]
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor

from lardiff import ode_solvers
from lardiff.preprocessing import Transformation, compose

__all__ = ["GlobalFlow", "GlobalSampler", "PDG_CODES"]

# fixed order, so a checkpoint's particle embedding keeps its meaning
PDG_CODES = [-211, -13, -11, 11, 13, 22, 211, 2112, 2212]


class GlobalNetwork(nn.Module):
    """MLP velocity field over the 2-vector, conditioned on energy and type."""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_blocks: int = 3,
        num_particles: int = len(PDG_CODES),
        frequencies: int = 3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(2, dim_embedding)
        # cond: 2*frequencies time features + log-energy
        self.cond_embedding = nn.Linear(2 * frequencies + 1, dim_embedding)
        self.particle_embedding = nn.Embedding(num_particles, dim_embedding)
        layers = []
        for _ in range(num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Linear(dim_embedding, dim_embedding),
                    nn.GELU(),
                    nn.Linear(dim_embedding, dim_embedding),
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dim_embedding) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(dim_embedding, 2)

    def forward(self, t: Tensor, x: Tensor, cond: Tensor, label: Tensor) -> Tensor:
        h = self.embedding(x)
        h = h + self.cond_embedding(torch.cat([t, cond], dim=-1))
        h = h + self.particle_embedding(label)
        for block, norm in zip(self.blocks, self.norms):
            h = h + block(norm(h))
        return self.head(h)


class GlobalFlow(nn.Module):
    """CNF over a plain vector.

    Mirrors lardiff.flow_matching.CNF, minus the padding BlockMask: there is no
    sequence dimension here, so the attention machinery does not apply.
    """

    def __init__(
        self,
        network: GlobalNetwork,
        frequencies: int = 3,
        solver: str = "heun",
    ) -> None:
        super().__init__()
        self.frequencies = nn.Buffer(
            (torch.arange(1, frequencies + 1) * torch.pi).reshape(1, -1)
        )
        self.network = network
        self.solver = ode_solvers.integrators[solver]

    def forward(self, t: Tensor, x: Tensor, **kwargs) -> Tensor:
        t = self.frequencies * t.reshape(-1, 1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(x.shape[0], -1)
        return self.network(t, x, **kwargs)

    def loss(self, x: Tensor, **kwargs) -> Tensor:
        t = torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x
        return (self(t, y, **kwargs) - u).square()

    @torch.no_grad()
    def sample(self, num: int, num_timesteps: int = 100, **kwargs) -> Tensor:
        z = torch.randn(
            num, 2, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self.solver(self, z, 1.0, 0.0, num_timesteps, **kwargs)


def load_globals(
    path: str, pdg: list[int] | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (target=[N, R], energy, label) for events with at least one hit."""
    with h5py.File(path, "r") as f:
        codes = f["pdg"][:]
        energy = f["energy_MeV"][:]
        n_hits = f["n_hits"][:]
        edep = f["edep_total_MeV"][:]
    keep = (n_hits > 0) & (energy > 0) & (edep > 0)
    if pdg is not None:
        keep &= np.isin(codes, pdg)
    codes, energy, n_hits, edep = (a[keep] for a in (codes, energy, n_hits, edep))
    index = {code: i for i, code in enumerate(PDG_CODES)}
    unknown = set(np.unique(codes)) - set(index)
    if unknown:
        raise ValueError(f"PDG codes missing from PDG_CODES: {sorted(unknown)}")
    label = np.array([index[c] for c in codes])
    target = np.stack([n_hits.astype(np.float64), edep / energy], axis=-1)
    return (
        torch.from_numpy(target).float(),
        torch.from_numpy(energy.astype(np.float32)).unsqueeze(-1),
        torch.from_numpy(label.astype(np.int64)),
    )


class GlobalSampler:
    """Draws (N, R) for given incident energies and particle type."""

    def __init__(self, run_dir: str, device: str | torch.device = "cpu") -> None:
        with open(os.path.join(run_dir, "conf.yaml")) as f:
            self.conf = yaml.safe_load(f)
        self.device = torch.device(device)
        network = GlobalNetwork(**self.conf["model"])
        self.flow = GlobalFlow(
            network, frequencies=self.conf["model"].get("frequencies", 3)
        ).to(self.device)
        state = torch.load(
            os.path.join(run_dir, "best.pt"), map_location=self.device,
            weights_only=True,
        )
        self.flow.load_state_dict(state)
        self.flow.eval()
        trafos = torch.load(
            os.path.join(run_dir, "trafos.pt"), map_location="cpu", weights_only=True
        )
        self.target_trafo = compose(self.conf["data"]["target_trafo"])
        self.cond_trafo = compose(self.conf["data"]["cond_trafo"])
        self.target_trafo.load_state_dict(trafos["target_trafo"])
        self.cond_trafo.load_state_dict(trafos["cond_trafo"])
        self.target_trafo.to(self.device)
        self.cond_trafo.to(self.device)

    @torch.no_grad()
    def sample(
        self,
        energy: Tensor,
        pdg: int,
        num_timesteps: int = 100,
        max_num_points: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return (n_points int64, ratio float) for each incident energy."""
        energy = energy.to(self.device).reshape(-1, 1)
        label = torch.full(
            (energy.shape[0],), PDG_CODES.index(pdg), dtype=torch.long,
            device=self.device,
        )
        cond = self.cond_trafo(energy)
        out = self.flow.sample(
            energy.shape[0], num_timesteps=num_timesteps, cond=cond, label=label
        )
        out = self.target_trafo.inverse(out)
        n = out[:, 0].round().clamp_min(1).long()
        if max_num_points is not None:
            # truncation in the point caches only ever clamps the count, so
            # clamping the sampled count reproduces the capped distribution
            n = n.clamp_max(max_num_points)
        return n.cpu(), out[:, 1].clamp_min(0.0).cpu()


def train(args) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    target, energy, label = load_globals(args.globals_file, args.pdg)
    print(f"{len(target)} events, {len(torch.unique(label))} species")

    val_len = min(args.val_len, len(target) // 5)
    perm = torch.randperm(len(target), generator=torch.Generator().manual_seed(0))
    target, energy, label = target[perm], energy[perm], label[perm]
    splits = {
        "train": slice(0, len(target) - val_len),
        "val": slice(len(target) - val_len, len(target)),
    }

    target_trafo = compose(args.target_trafo)
    cond_trafo = compose(args.cond_trafo)
    target_trafo.fit(target[splits["train"]])
    cond_trafo.fit(energy[splits["train"]])
    y = {k: target_trafo(target[s]).to(device) for k, s in splits.items()}
    c = {k: cond_trafo(energy[s]).to(device) for k, s in splits.items()}
    lab = {k: label[s].to(device) for k, s in splits.items()}

    network = GlobalNetwork(
        dim_embedding=args.dim_embedding,
        num_blocks=args.num_blocks,
        frequencies=args.frequencies,
    )
    flow = GlobalFlow(network, frequencies=args.frequencies).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"parameters: {n_params}")
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    n_train = y["train"].shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate,
        total_steps=args.num_epochs * steps_per_epoch,
    )

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(args.num_epochs):
        flow.train()
        order = torch.randperm(n_train, device=device)
        total = 0.0
        for i in range(steps_per_epoch):
            idx = order[i * args.batch_size : (i + 1) * args.batch_size]
            loss = flow.loss(
                y["train"][idx], cond=c["train"][idx], label=lab["train"][idx]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        flow.eval()
        with torch.no_grad():
            val = flow.loss(y["val"], cond=c["val"], label=lab["val"]).mean().item()
        history.append((total / steps_per_epoch, val))
        if val < best:
            best = val
            torch.save(flow.state_dict(), os.path.join(args.out, "best.pt"))
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            print(f"epoch {epoch:4d}  train {total / steps_per_epoch:.5f}  "
                  f"val {val:.5f}{'  *' if val == best else ''}")

    torch.save(
        {
            "target_trafo": target_trafo.state_dict(),
            "cond_trafo": cond_trafo.state_dict(),
        },
        os.path.join(args.out, "trafos.pt"),
    )
    with open(os.path.join(args.out, "conf.yaml"), "w") as f:
        yaml.safe_dump(
            {
                "data": {
                    "globals_file": args.globals_file,
                    "pdg": args.pdg,
                    "target_trafo": args.target_trafo,
                    "cond_trafo": args.cond_trafo,
                },
                "model": {
                    "dim_embedding": args.dim_embedding,
                    "num_blocks": args.num_blocks,
                    "num_particles": len(PDG_CODES),
                    "frequencies": args.frequencies,
                },
                "train": {
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "best_val_loss": best,
                },
            },
            f,
        )
    np.savetxt(os.path.join(args.out, "losses.txt"), np.array(history))
    print(f"best val loss {best:.5f}; wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("globals_file", help="globals cache from preprocess_globals")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--pdg", type=int, nargs="*", default=None,
        help="restrict to these PDG codes (default: all species)",
    )
    parser.add_argument("--dim-embedding", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--frequencies", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-len", type=int, default=50_000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    # log N spans ~4 decades, R is O(1) with a long tail for hadrons
    args.target_trafo = [["Log", {"alpha": 1.0e-6}], ["StandardScaler", {"shape": [1, 2]}]]
    args.cond_trafo = [["Log", {"alpha": 0.0}], ["StandardScaler", {"shape": [1, 1]}]]
    train(args)


if __name__ == "__main__":
    main()
