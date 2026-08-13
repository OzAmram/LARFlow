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


def detect_atoms(
    energy: np.ndarray, edep: np.ndarray, label: np.ndarray, tol: float = 1e-5
) -> dict[int, float]:
    """Find each species' fully-contained deposit, if it has one.

    A contained event deposits a fixed amount above its incident kinetic
    energy: 0 for e-/gamma, 2*m_e = 1.022 MeV for e+ (annihilation), nothing
    fixed for hadrons or muons (the Michel electron spectrum is continuous).
    That makes p(R | E) a *mixed* distribution -- a point mass plus a
    continuous tail -- which no continuous flow can represent, however long it
    trains.  Detecting the offset from data rather than hard-coding it means
    species without an atom simply report a negligible fraction and fall back
    to the plain flow.

    Returns {label index: offset in MeV} for species whose atom holds >2% of
    events.
    """
    offsets = {}
    for i in np.unique(label):
        sel = label == i
        delta = edep[sel] - energy[sel]
        c = float(np.median(delta))
        fraction = float(np.mean(np.abs(delta - c) <= tol * energy[sel]))
        if fraction > 0.02:
            offsets[int(i)] = c
    return offsets


def atom_mask(
    energy: np.ndarray,
    edep: np.ndarray,
    label: np.ndarray,
    offsets: dict[int, float],
    tol: float = 1e-5,
) -> np.ndarray:
    mask = np.zeros(len(energy), dtype=bool)
    for i, c in offsets.items():
        sel = label == i
        mask[sel] = np.abs(edep[sel] - energy[sel] - c) <= tol * energy[sel]
    return mask


class ContainmentClassifier(nn.Module):
    """p(fully contained | incident energy, particle type)."""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_particles: int = len(PDG_CODES),
    ) -> None:
        super().__init__()
        self.particle_embedding = nn.Embedding(num_particles, dim_embedding)
        self.net = nn.Sequential(
            nn.Linear(dim_embedding + 1, dim_embedding),
            nn.GELU(),
            nn.Linear(dim_embedding, dim_embedding),
            nn.GELU(),
            nn.Linear(dim_embedding, 1),
        )

    def forward(self, cond: Tensor, label: Tensor) -> Tensor:
        h = torch.cat([cond, self.particle_embedding(label)], dim=-1)
        return self.net(h).squeeze(-1)


class GlobalNetwork(nn.Module):
    """MLP velocity field over the target vector, conditioned on energy/type."""

    def __init__(
        self,
        dim: int = 2,
        dim_embedding: int = 128,
        num_blocks: int = 3,
        num_particles: int = len(PDG_CODES),
        frequencies: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embedding = nn.Linear(dim, dim_embedding)
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
        self.head = nn.Linear(dim_embedding, dim)

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
            num,
            self.network.dim,
            device=self.frequencies.device,
            dtype=self.frequencies.dtype,
        )
        return self.solver(self, z, 1.0, 0.0, num_timesteps, **kwargs)


def load_globals(
    path: str, pdg: list[int] | None = None
) -> tuple[Tensor, Tensor, Tensor, np.ndarray]:
    """Return (target=[N, R], energy, label, edep) for events with >=1 hit."""
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
        np.stack([energy.astype(np.float64), edep.astype(np.float64)], axis=-1),
    )


class GlobalSampler:
    """Draws (N, R) for given incident energies and particle type."""

    def __init__(self, run_dir: str, device: str | torch.device = "cpu") -> None:
        with open(os.path.join(run_dir, "conf.yaml")) as f:
            self.conf = yaml.safe_load(f)
        self.device = torch.device(device)
        model_conf = dict(self.conf["model"])
        frequencies = model_conf.get("frequencies", 3)
        state = torch.load(
            os.path.join(run_dir, "best.pt"), map_location=self.device,
            weights_only=True,
        )
        trafos = torch.load(
            os.path.join(run_dir, "trafos.pt"), map_location="cpu", weights_only=True
        )

        def build(dim, key):
            flow = GlobalFlow(
                GlobalNetwork(dim=dim, **model_conf), frequencies=frequencies
            ).to(self.device)
            flow.load_state_dict(state[key])
            flow.eval()
            return flow

        self.flow = build(2, "flow_escaped")
        self.cond_trafo = compose(self.conf["data"]["cond_trafo"])
        self.cond_trafo.load_state_dict(trafos["cond_trafo"])
        self.cond_trafo.to(self.device)
        self.target_trafo = compose(self.conf["data"]["target_trafo"])
        self.target_trafo.load_state_dict(trafos["target_trafo"])
        self.target_trafo.to(self.device)

        self.containment = self.conf.get("containment", {})
        self.offsets = {
            int(k): float(v) for k, v in self.containment.get("offsets", {}).items()
        }
        if self.containment.get("enabled"):
            self.flow_contained = build(1, "flow_contained")
            self.classifier = ContainmentClassifier(
                dim_embedding=model_conf["dim_embedding"],
                num_particles=model_conf["num_particles"],
            ).to(self.device)
            self.classifier.load_state_dict(state["classifier"])
            self.classifier.eval()
            self.count_trafo = compose(self.conf["data"]["count_trafo"])
            self.count_trafo.load_state_dict(trafos["count_trafo"])
            self.count_trafo.to(self.device)

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
        n = out[:, 0]
        ratio = out[:, 1].clamp_min(0.0)

        has_atom = self.containment.get("enabled") and pdg in self.offsets
        if has_atom:
            # the contained population is a point mass the flow cannot express;
            # draw membership, then N from its own flow and R from the identity
            ceiling = (energy.squeeze(-1) + self.offsets[pdg]) / energy.squeeze(-1)
            probability = torch.sigmoid(self.classifier(cond, label))
            contained = torch.rand_like(probability) < probability
            n_c = self.count_trafo.inverse(
                self.flow_contained.sample(
                    energy.shape[0], num_timesteps=num_timesteps,
                    cond=cond, label=label,
                )
            )[:, 0]
            n = torch.where(contained, n_c, n)
            # escaped events deposited less than the contained value by
            # definition, so the continuous part lives strictly below it
            ratio = torch.where(contained, ceiling, torch.minimum(ratio, ceiling))

        n = n.round().clamp_min(1).long()
        if max_num_points is not None:
            # truncation in the point caches only ever clamps the count, so
            # clamping the sampled count reproduces the capped distribution
            n = n.clamp_max(max_num_points)
        return n.cpu(), ratio.cpu()


def _fit_flow(
    dim: int, y: dict, c: dict, lab: dict, args, device, name: str
) -> tuple[GlobalFlow, float, list]:
    """Train one flow component and return it with its best val loss."""
    network = GlobalNetwork(
        dim=dim,
        dim_embedding=args.dim_embedding,
        num_blocks=args.num_blocks,
        frequencies=args.frequencies,
    )
    flow = GlobalFlow(network, frequencies=args.frequencies).to(device)
    print(f"[{name}] {y['train'].shape[0]} events, "
          f"{sum(p.numel() for p in flow.parameters())} parameters")
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    n_train = y["train"].shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate,
        total_steps=args.num_epochs * steps_per_epoch,
    )
    best, best_state, history = float("inf"), None, []
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
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
        if epoch % 50 == 0 or epoch == args.num_epochs - 1:
            print(f"  [{name}] epoch {epoch:4d}  train {total / steps_per_epoch:.5f}"
                  f"  val {val:.5f}")
    flow.load_state_dict(best_state)
    print(f"  [{name}] best val {best:.5f}")
    return flow, best, history


def train(args) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    target, energy, label, raw = load_globals(args.globals_file, args.pdg)
    print(f"{len(target)} events, {len(torch.unique(label))} species")

    perm = torch.randperm(len(target), generator=torch.Generator().manual_seed(0))
    target, energy, label = target[perm], energy[perm], label[perm]
    raw = raw[perm.numpy()]

    offsets: dict[int, float] = {}
    contained = np.zeros(len(target), dtype=bool)
    if args.containment:
        offsets = detect_atoms(raw[:, 0], raw[:, 1], label.numpy())
        contained = atom_mask(raw[:, 0], raw[:, 1], label.numpy(), offsets)
        for i, c_off in sorted(offsets.items()):
            sel = label.numpy() == i
            print(f"  atom: {PDG_CODES[i]:>6d}  offset {c_off:+9.4f} MeV  "
                  f"{contained[sel].mean():.4f} of its events")
    print(f"contained overall: {contained.mean():.4f}")

    val_len = min(args.val_len, len(target) // 5)
    is_val = np.zeros(len(target), dtype=bool)
    is_val[-val_len:] = True

    cond_trafo = compose(args.cond_trafo)
    cond_trafo.fit(energy[~is_val])
    cond_all = {
        "train": cond_trafo(energy[~is_val]).to(device),
        "val": cond_trafo(energy[is_val]).to(device),
    }
    lab_all = {
        "train": label[~is_val].to(device),
        "val": label[is_val].to(device),
    }
    os.makedirs(args.out, exist_ok=True)
    state: dict[str, dict] = {}
    trafos: dict[str, dict] = {"cond_trafo": cond_trafo.state_dict()}
    losses: dict[str, float] = {}

    def subset(mask):
        m = torch.from_numpy(mask)
        return {
            "train": m[~torch.from_numpy(is_val)],
            "val": m[torch.from_numpy(is_val)],
        }

    # escaped events: joint (log N, log R)
    esc = subset(~contained)
    esc_trafo = compose(args.target_trafo)
    esc_trafo.fit(target[~is_val][esc["train"]])
    y = {k: esc_trafo(target[is_val if k == "val" else ~is_val][esc[k]]).to(device)
         for k in ("train", "val")}
    c = {k: cond_all[k][esc[k].to(device)] for k in ("train", "val")}
    lab = {k: lab_all[k][esc[k].to(device)] for k in ("train", "val")}
    flow, best, history = _fit_flow(2, y, c, lab, args, device, "escaped")
    state["flow_escaped"] = flow.state_dict()
    trafos["target_trafo"] = esc_trafo.state_dict()
    losses["escaped"] = best
    np.savetxt(os.path.join(args.out, "losses.txt"), np.array(history))

    # contained events: only N is free, R is pinned to the species offset
    if contained.any():
        con = subset(contained)
        con_trafo = compose(args.count_trafo)
        con_trafo.fit(target[~is_val][con["train"]][:, :1])
        y = {k: con_trafo(
                target[is_val if k == "val" else ~is_val][con[k]][:, :1]
             ).to(device) for k in ("train", "val")}
        c = {k: cond_all[k][con[k].to(device)] for k in ("train", "val")}
        lab = {k: lab_all[k][con[k].to(device)] for k in ("train", "val")}
        flow_c, best_c, _ = _fit_flow(1, y, c, lab, args, device, "contained")
        state["flow_contained"] = flow_c.state_dict()
        trafos["count_trafo"] = con_trafo.state_dict()
        losses["contained"] = best_c

        # p(contained | E, type)
        clf = ContainmentClassifier(dim_embedding=args.dim_embedding).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)
        t_all = {
            "train": torch.from_numpy(contained[~is_val]).float().to(device),
            "val": torch.from_numpy(contained[is_val]).float().to(device),
        }
        n_train = cond_all["train"].shape[0]
        steps = max(1, n_train // args.batch_size)
        bce = nn.BCEWithLogitsLoss()
        for epoch in range(args.num_epochs // 3):
            clf.train()
            order = torch.randperm(n_train, device=device)
            for i in range(steps):
                idx = order[i * args.batch_size : (i + 1) * args.batch_size]
                loss = bce(
                    clf(cond_all["train"][idx], lab_all["train"][idx]),
                    t_all["train"][idx],
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        clf.eval()
        with torch.no_grad():
            logit = clf(cond_all["val"], lab_all["val"])
            val_bce = bce(logit, t_all["val"]).item()
            acc = ((logit > 0).float() == t_all["val"]).float().mean().item()
        print(f"[classifier] val BCE {val_bce:.5f}  accuracy {acc:.4f}")
        state["classifier"] = clf.state_dict()
        losses["classifier_bce"] = val_bce
        losses["classifier_acc"] = acc

    torch.save(state, os.path.join(args.out, "best.pt"))
    torch.save(trafos, os.path.join(args.out, "trafos.pt"))
    with open(os.path.join(args.out, "conf.yaml"), "w") as f:
        yaml.safe_dump(
            {
                "data": {
                    "globals_file": args.globals_file,
                    "pdg": args.pdg,
                    "target_trafo": args.target_trafo,
                    "count_trafo": args.count_trafo,
                    "cond_trafo": args.cond_trafo,
                },
                "model": {
                    "dim_embedding": args.dim_embedding,
                    "num_blocks": args.num_blocks,
                    "num_particles": len(PDG_CODES),
                    "frequencies": args.frequencies,
                },
                "containment": {
                    "enabled": bool(args.containment and contained.any()),
                    # PDG code -> deposit above incident energy when contained
                    "offsets": {int(PDG_CODES[i]): float(v)
                                for i, v in sorted(offsets.items())},
                },
                "train": {
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "losses": losses,
                },
            },
            f,
        )
    print(f"wrote {args.out}")


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
    parser.add_argument(
        "--no-containment", dest="containment", action="store_false",
        help="model R with a single continuous flow, without the point mass "
        "at full containment (worse for e-/e+/gamma, identical elsewhere)",
    )
    args = parser.parse_args(argv)
    # log N spans ~4 decades, R is O(1) with a long tail for hadrons
    args.target_trafo = [["Log", {"alpha": 1.0e-6}], ["StandardScaler", {"shape": [1, 2]}]]
    args.count_trafo = [["Log", {"alpha": 1.0e-6}], ["StandardScaler", {"shape": [1, 1]}]]
    args.cond_trafo = [["Log", {"alpha": 0.0}], ["StandardScaler", {"shape": [1, 1]}]]
    train(args)


if __name__ == "__main__":
    main()
