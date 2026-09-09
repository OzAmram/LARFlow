"""Multivariate comparison of generated events against Geant4.

The per-observable KS tests in `evaluate.py` are one-dimensional: they miss
correlations, so a model can pass all of them while getting the joint
distribution wrong.  This adds two tests over a vector of per-event features:

* a **classifier two-sample test** -- train a classifier to tell real from
  generated and report its AUC on held-out events.  0.5 means the two samples
  are indistinguishable to it; the further above, the more structure separates
  them.  This is the more interpretable of the two, since the same classifier
  can be asked *which* feature it used.
* **FPD** and **KPD**, the Frechet and kernel physics distances of
  arXiv:2211.10295, which summarise the mismatch as a single number with an
  uncertainty.

The FPD and KPD implementations are transcribed from `jetnet.evaluation`
(https://github.com/jet-net/JetNet, MIT licence) so that values are comparable
with work that uses that package.  They are vendored rather than imported
because `jetnet.evaluation` pulls in energyflow, coffea and awkward at import
time for its Wasserstein metrics, none of which these two functions touch.

    python -m lardiff.metrics <samples.h5> <cache.h5> [--out results.json]

A note on sample size: FPD is defined by extrapolating the Frechet distance to
infinite batch size, and jetnet recommends at least 50,000 events with default
batch settings of 20,000-50,000.  The single-species caches here hold out only
`val_len` events, so `--fpd-min-samples` and `--fpd-max-samples` are exposed and
default to a tenth of jetnet's.  Absolute values are then not comparable with
published FPDs, but remain comparable between models and species measured the
same way -- which is what a comparison table needs.  The settings used are
recorded in the output.
"""

import argparse
import json
import os
import warnings

import h5py
import numpy as np
import torch
from scipy import linalg
from scipy.optimize import curve_fit
from scipy.stats import iqr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

__all__ = ["event_features", "fpd", "kpd", "classifier_auc", "FEATURE_NAMES"]

FEATURE_NAMES = [
    "log E_inc",
    "response",
    "log N",
    "centroid x",
    "centroid y",
    "centroid z",
    "width x",
    "width y",
    "width z",
    "extent x",
    "extent y",
    "extent z",
]


def event_features(points: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Per-event feature vector, one row per event.

    These are exactly the event-level quantities `evaluate.py` histograms, so
    the multivariate tests are measured on the same observables the reader sees
    plotted -- the extra information is the correlations between them.
    """
    edep = points[:, :, 3]
    hit = edep > 0
    w = np.where(hit, edep, 0.0)
    wsum = np.maximum(w.sum(1, keepdims=True), 1e-30)
    n = hit.sum(1)
    total = edep.sum(1)

    cols = [
        np.log(np.maximum(energy, 1e-6)),
        total / np.maximum(energy, 1e-30),
        np.log(np.maximum(n, 1)),
    ]
    centroids, widths, extents = [], [], []
    for axis in range(3):
        c = points[:, :, axis]
        cen = (w * c).sum(1, keepdims=True) / wsum
        centroids.append(cen[:, 0])
        widths.append(np.sqrt((w * (c - cen) ** 2).sum(1) / wsum[:, 0]))
        masked = np.where(hit, c, np.nan)
        with np.errstate(invalid="ignore"):
            extents.append(np.nanmax(masked, 1) - np.nanmin(masked, 1))
    cols += centroids + widths + extents
    feats = np.stack(cols, axis=-1).astype(np.float32)
    # an event with no hits at all leaves nan extents; drop it rather than
    # letting it poison a covariance
    return feats[np.isfinite(feats).all(axis=1)]


# --------------------------------------------------------------------------
# FPD and KPD, transcribed from jetnet.evaluation.gen_metrics (MIT licence).
# --------------------------------------------------------------------------


def _normalise_features(X, Y=None):
    maxes = np.max(np.abs(X), axis=0)
    maxes[maxes == 0] = 1  # a feature that is identically zero must not divide
    return (X / maxes, Y / maxes) if Y is not None else X / maxes


def _linear(x, intercept, slope):
    return intercept + slope * x


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between two multivariate Gaussians.

    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2 sqrt(C_1 C_2)).
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # the product can be near singular; nudge the diagonal
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not (
            np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
            or np.isclose(np.trace(covmean.imag) / np.trace(covmean.real), 0, atol=1e-3)
        ):
            warnings.warn(
                "large imaginary component in the covariance square root",
                RuntimeWarning,
                stacklevel=2,
            )
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def fpd(
    real_features,
    gen_features,
    min_samples: int = 2_000,
    max_samples: int = 10_000,
    num_batches: int = 20,
    num_points: int = 10,
    normalise: bool = True,
    seed: int = 42,
) -> tuple[float, float]:
    """Frechet physics distance: the Frechet distance extrapolated to infinite
    batch size, with the error on the extrapolated intercept.

    jetnet's defaults are min_samples=20000, max_samples=50000; the defaults
    here are a tenth of that because the held-out samples are smaller.  Values
    computed with different settings are not comparable with each other.
    """
    X, Y = np.asarray(real_features), np.asarray(gen_features)
    if min_samples > min(len(X), len(Y)):
        raise ValueError(
            f"min_samples {min_samples} exceeds the {min(len(X), len(Y))} events "
            "available; lower --fpd-min-samples"
        )
    if normalise:
        X, Y = _normalise_features(X, Y)

    # batch sizes at regular intervals in 1/N, so the fit below is linear
    batches = (
        1 / np.linspace(1.0 / min_samples, 1.0 / max_samples, num_points)
    ).astype("int32")
    rng = np.random.default_rng(seed)
    vals = []
    for batch_size in batches:
        point = []
        for _ in range(num_batches):
            a = X[rng.choice(len(X), size=batch_size)]
            b = Y[rng.choice(len(Y), size=batch_size)]
            point.append(
                _frechet_distance(
                    a.mean(0), np.cov(a, rowvar=False),
                    b.mean(0), np.cov(b, rowvar=False),
                )
            )
        vals.append(np.mean(point))

    params, covs = curve_fit(
        _linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf])
    )
    return float(params[0]), float(np.sqrt(np.diag(covs)[0]))


def _poly_kernel_pairwise(X, Y, degree):
    gamma = 1.0 / X.shape[-1]
    return (X @ Y.T * gamma + 1.0) ** degree


def _mmd_poly_quadratic_unbiased(X, Y, degree=4):
    XX = _poly_kernel_pairwise(X, X, degree)
    YY = _poly_kernel_pairwise(Y, Y, degree)
    XY = _poly_kernel_pairwise(X, Y, degree)
    m, n = XX.shape[0], YY.shape[0]
    return (
        (XX.sum() - np.trace(XX)) / (m * (m - 1))
        + (YY.sum() - np.trace(YY)) / (n * (n - 1))
        - 2 * np.mean(XY)
    )


def kpd(
    real_features,
    gen_features,
    num_batches: int = 10,
    batch_size: int = 5_000,
    normalise: bool = True,
    seed: int = 42,
) -> tuple[float, float]:
    """Kernel physics distance: median MMD with a degree-4 polynomial kernel
    over random batches, with half the 16--84 interquantile range as the error.

    Needs far fewer events than FPD, since nothing is extrapolated.
    """
    X, Y = np.asarray(real_features), np.asarray(gen_features)
    batch_size = min(batch_size, len(X), len(Y))
    if normalise:
        X, Y = _normalise_features(X, Y)
    vals = []
    for i in range(num_batches):
        rng = np.random.default_rng(seed + i * 1000)
        a = X[rng.choice(len(X), size=batch_size)]
        b = Y[rng.choice(len(Y), size=batch_size)]
        vals.append(_mmd_poly_quadratic_unbiased(a, b))
    return float(np.median(vals)), float(iqr(vals, rng=(16.275, 83.725)) / 2)


# --------------------------------------------------------------------------
# Classifier two-sample test
# --------------------------------------------------------------------------


class _MLP(torch.nn.Module):
    def __init__(self, dim, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        seq = [torch.nn.Linear(dim, hidden), torch.nn.LeakyReLU(), torch.nn.Dropout(dropout)]
        for _ in range(layers):
            seq += [torch.nn.Linear(hidden, hidden), torch.nn.LeakyReLU(),
                    torch.nn.Dropout(dropout)]
        seq += [torch.nn.Linear(hidden, 1)]  # logits; use BCEWithLogitsLoss
        self.net = torch.nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def classifier_auc(
    real_features,
    gen_features,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train a classifier to separate real from generated, return held-out AUC.

    An AUC of 0.5 means the two samples are indistinguishable in these
    features.  The split is 60/20/20 train/validation/test; the epoch with the
    best validation AUC is the one scored on test, so the reported number is
    not selected on the data it is quoted for.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.concatenate([np.asarray(real_features), np.asarray(gen_features)])
    y = np.concatenate([np.zeros(len(real_features)), np.ones(len(gen_features))])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_tr, n_va = int(0.6 * len(X)), int(0.2 * len(X))
    scaler = StandardScaler().fit(X[:n_tr])
    X = scaler.transform(X).astype(np.float32)
    splits = {
        "train": (X[:n_tr], y[:n_tr]),
        "val": (X[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va]),
        "test": (X[n_tr + n_va :], y[n_tr + n_va :]),
    }

    model = _MLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt = torch.tensor(splits["train"][0], device=device)
    yt = torch.tensor(splits["train"][1], dtype=torch.float32, device=device)

    # 0.5 is the no-information point, so any epoch is an improvement on it;
    # starting at 0.0 would make the |auc - 0.5| test below never fire
    best_val, best_state = 0.5, None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), batch_size):
            idx = order[i : i + batch_size]
            opt.zero_grad()
            loss_fn(model(xt[idx]).squeeze(-1), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(splits["val"][0], device=device)
            pv = model(xv).squeeze(-1).cpu().numpy()
        val_auc = roc_auc_score(splits["val"][1], pv)
        # AUC is symmetric about 0.5: a classifier that is reliably wrong is
        # just as much evidence of a difference as one that is reliably right
        if abs(val_auc - 0.5) > abs(best_val - 0.5):
            best_val = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xs = torch.tensor(splits["test"][0], device=device)
        ps = model(xs).squeeze(-1).cpu().numpy()
    return {
        "auc": float(roc_auc_score(splits["test"][1], ps)),
        "val_auc": float(best_val),
        "n_train": int(n_tr),
        "n_test": int(len(splits["test"][1])),
    }


def load_pair(samples_file: str, cache_file: str, pdg: int | None = None):
    """Generated and truth features for the events in a samples file."""
    with h5py.File(samples_file, "r") as f:
        gen = f["points"][:]
        energy = f["energy_MeV"][:]
        index = f["cache_index"][:]
        codes = f["pdg"][:] if "pdg" in f else None
    with h5py.File(cache_file, "r") as f:
        if "points" in f:
            truth = f["points"][int(index[0]) : int(index[-1]) + 1]
        else:
            from lardiff.evaluate import read_truth_points

            truth = read_truth_points(
                cache_file, int(index[0]), int(index[-1]) + 1, gen.shape[1]
            )
    if pdg is not None:
        if codes is None:
            raise SystemExit("--pdg needs samples that record a species per event")
        keep = codes == pdg
        gen, truth, energy = gen[keep], truth[keep], energy[keep]
    return event_features(gen, energy), event_features(truth, energy)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_file")
    parser.add_argument("cache_file")
    parser.add_argument("--pdg", type=int, default=None, help="restrict to one species")
    parser.add_argument("--out", default=None, help="write results as json here")
    parser.add_argument("--fpd-min-samples", type=int, default=2_000)
    parser.add_argument("--fpd-max-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parsed = parser.parse_args(args)

    gen, truth = load_pair(parsed.samples_file, parsed.cache_file, parsed.pdg)
    n = min(len(gen), len(truth))
    gen, truth = gen[:n], truth[:n]
    print(f"{n} events, {gen.shape[1]} features")

    fpd_max = min(parsed.fpd_max_samples, n)
    fpd_min = min(parsed.fpd_min_samples, fpd_max // 2)
    fpd_val, fpd_err = fpd(truth, gen, min_samples=fpd_min, max_samples=fpd_max,
                           seed=parsed.seed)
    kpd_val, kpd_err = kpd(truth, gen, seed=parsed.seed)
    cls = classifier_auc(truth, gen, epochs=parsed.epochs, seed=parsed.seed,
                         device=parsed.device)

    # a floor for how far from 0.5 an AUC can sit by chance alone: the same
    # classifier trained to separate two halves of the Geant4 sample
    half = len(truth) // 2
    null = classifier_auc(truth[:half], truth[half:], epochs=parsed.epochs,
                          seed=parsed.seed, device=parsed.device)

    result = {
        "samples_file": parsed.samples_file,
        "pdg": parsed.pdg,
        "n_events": n,
        "features": FEATURE_NAMES,
        "fpd": fpd_val, "fpd_err": fpd_err,
        "fpd_min_samples": fpd_min, "fpd_max_samples": fpd_max,
        "kpd": kpd_val, "kpd_err": kpd_err,
        "classifier_auc": cls["auc"],
        "classifier_auc_null": null["auc"],
        "classifier_n_test": cls["n_test"],
    }
    print(f"  FPD              {fpd_val * 1e3:8.4f} +- {fpd_err * 1e3:.4f}  x1e-3"
          f"   (batches {fpd_min}-{fpd_max})")
    print(f"  KPD              {kpd_val * 1e3:8.4f} +- {kpd_err * 1e3:.4f}  x1e-3")
    print(f"  classifier AUC   {cls['auc']:8.4f}   (Geant4 vs Geant4: {null['auc']:.4f})")

    if parsed.out:
        os.makedirs(os.path.dirname(os.path.abspath(parsed.out)), exist_ok=True)
        with open(parsed.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {parsed.out}")


if __name__ == "__main__":
    main()
