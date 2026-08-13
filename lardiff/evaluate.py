"""Compare generated samples with the matching held-out Geant4 events.

Usage:
    python -m lardiff.evaluate <run_dir>/samples00.h5 <cache.h5> [--out DIR]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def event_observables(points: np.ndarray) -> dict[str, np.ndarray]:
    """Per-event observables of a (N, P, 4) padded point array (hits: edep > 0)."""
    edep = points[:, :, 3]
    hit = edep > 0
    n_hits = hit.sum(axis=1)
    e_total = np.where(hit, edep, 0.0).sum(axis=1)

    obs = {"n_hits": n_hits, "e_total": e_total}
    safe_e_total = np.where(e_total > 0, e_total, 1.0)
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        coord = points[:, :, axis]
        weighted = np.where(hit, coord * edep, 0.0).sum(axis=1) / safe_e_total
        cmin = np.where(hit, coord, np.inf).min(axis=1)
        cmax = np.where(hit, coord, -np.inf).max(axis=1)
        extent = np.where(n_hits > 0, cmax - cmin, 0.0)
        obs[f"centroid_{name}"] = weighted
        obs[f"extent_{name}"] = extent
    return obs


def flat_hits(points: np.ndarray) -> dict[str, np.ndarray]:
    hit = points[:, :, 3] > 0
    return {
        "x": points[:, :, 0][hit],
        "y": points[:, :, 1][hit],
        "z": points[:, :, 2][hit],
        "edep": points[:, :, 3][hit],
    }


def plot_hist(
    gen: np.ndarray,
    g4: np.ndarray,
    label: str,
    path: str,
    logx: bool = False,
    logy: bool = True,
    bins: int = 80,
    xrange: tuple[float, float] | None = None,
    show_moments: bool = False,
):
    """Overlaid Geant4/model histogram.

    xrange       explicit binning range, for observables whose outliers would
                 otherwise stretch the axis and hide the bulk. Entries outside
                 it are piled into the end bins rather than dropped, so the
                 plot never hides a tail.
    show_moments append mean +- std to the legend labels, so a difference in
                 *width* between the two distributions is readable as a number
                 and not only as a shape. The moments are of the *unclipped*
                 data.
    """
    combined = np.concatenate([gen, g4])
    if logx:
        combined = combined[combined > 0]
    lo, hi = xrange if xrange is not None else (combined.min(), combined.max())
    edges = (
        np.geomspace(lo, hi, bins + 1) if logx else np.linspace(lo, hi, bins + 1)
    )
    plt.figure(figsize=(6, 4.5))
    labels = ["Geant4", "Model"]
    if show_moments:  # before clipping, so the numbers describe the real data
        labels = [
            f"{name} ({a.mean():.4g} $\\pm$ {a.std():.3g})"
            for name, a in zip(labels, [g4, gen])
        ]
    if xrange is not None:  # pile over/underflow into the end bins
        margin = (edges[-1] - edges[-2]) * 0.5
        gen, g4 = (np.clip(a, lo + margin, hi - margin) for a in (gen, g4))
    plt.hist(g4, bins=edges, histtype="step", density=True, label=labels[0])
    plt.hist(gen, bins=edges, histtype="step", density=True, label=labels[1])
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(label)
    plt.ylabel("density")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_energy_profile(
    gen_hits: dict[str, np.ndarray],
    g4_hits: dict[str, np.ndarray],
    n_events: int,
    axis: str,
    path: str,
    per: str = "event",
    bins: int = 80,
):
    """Average deposited energy vs. a coordinate.

    per="event": mean energy deposited per event in each coordinate bin
    per="hit":   mean energy of a single hit in each coordinate bin
    """
    combined = np.concatenate([gen_hits[axis], g4_hits[axis]])
    edges = np.linspace(combined.min(), combined.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(6, 4.5))
    for hits, label in [(g4_hits, "Geant4"), (gen_hits, "Model")]:
        e_sum, _ = np.histogram(hits[axis], bins=edges, weights=hits["edep"])
        if per == "event":
            profile = e_sum / n_events
        else:
            counts, _ = np.histogram(hits[axis], bins=edges)
            profile = np.divide(
                e_sum, counts, out=np.full(bins, np.nan), where=counts > 0
            )
        plt.step(centers, profile, where="mid", label=label)
    plt.yscale("log")
    plt.xlabel(f"hit {axis} [mm]")
    if per == "event":
        plt.ylabel("mean deposited energy per event [MeV / bin]")
    else:
        plt.ylabel("mean hit energy [MeV]")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_response_profile(
    e_inc: np.ndarray,
    ratio_gen: np.ndarray,
    ratio_g4: np.ndarray,
    path: str,
    bins: int = 20,
):
    edges = np.geomspace(e_inc.min(), e_inc.max(), bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    index = np.clip(np.digitize(e_inc, edges) - 1, 0, bins - 1)
    # multiplicative dodge (the axis is log): side-by-side bars instead of one
    # series drawn on top of the other, and caps so the spreads are comparable
    dodge = (edges[1] / edges[0]) ** 0.12
    plt.figure(figsize=(6, 4.5))
    for ratio, label, shift in [
        (ratio_g4, "Geant4", 1.0 / dodge),
        (ratio_gen, "Model", dodge),
    ]:
        mean = np.full(bins, np.nan)
        std = np.full(bins, np.nan)
        for i in range(bins):
            sel = ratio[index == i]
            if len(sel) > 1:
                mean[i], std[i] = sel.mean(), sel.std()
        plt.errorbar(
            centers * shift,
            mean,
            yerr=std,
            fmt="o",
            markersize=3,
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            label=label,
        )
    plt.xscale("log")
    plt.xlabel("incident energy [MeV]")
    plt.ylabel("total deposited / incident energy")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_event_displays(
    gen: np.ndarray, g4: np.ndarray, path: str, num_events: int = 4
):
    fig, axes = plt.subplots(
        num_events, 4, figsize=(20, 4 * num_events), constrained_layout=True
    )
    for row in range(num_events):
        for col, (points, proj, title) in enumerate(
            [
                (g4[row], (0, 1), "Geant4 x-y"),
                (gen[row], (0, 1), "Model x-y"),
                (g4[row], (0, 2), "Geant4 x-z"),
                (gen[row], (0, 2), "Model x-z"),
            ]
        ):
            axis = axes[row, col]
            hit = points[:, 3] > 0
            u, v, e = points[hit, proj[0]], points[hit, proj[1]], points[hit, 3]
            if len(e) > 0:
                sc = axis.scatter(
                    u, v, c=np.log10(e), s=4, cmap="viridis", alpha=0.8
                )
                fig.colorbar(sc, ax=axis, label="log10(Edep/MeV)", shrink=0.8)
            axis.set_title(f"event {row}: {title}")
        # shared axis limits between truth and model for each projection
        for pair in [(0, 1), (2, 3)]:
            xlim = [
                min(axes[row, p].get_xlim()[0] for p in pair),
                max(axes[row, p].get_xlim()[1] for p in pair),
            ]
            ylim = [
                min(axes[row, p].get_ylim()[0] for p in pair),
                max(axes[row, p].get_ylim()[1] for p in pair),
            ]
            for p in pair:
                axes[row, p].set_xlim(xlim)
                axes[row, p].set_ylim(ylim)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_file", help="generated samples h5")
    parser.add_argument("cache_file", help="preprocessed cache h5 with the truth")
    parser.add_argument("--out", default=None, help="output directory for plots")
    parsed = parser.parse_args(args)

    out_dir = parsed.out or os.path.join(
        os.path.dirname(parsed.samples_file),
        "eval_"
        + os.path.splitext(os.path.basename(parsed.samples_file))[0],
    )
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(parsed.samples_file, "r") as f:
        gen_points = f["points"][:]
        e_inc = f["energy_MeV"][:]
        cache_index = f["cache_index"][:]
    with h5py.File(parsed.cache_file, "r") as f:
        g4_points = f["points"][cache_index[0] : cache_index[-1] + 1]
    assert len(g4_points) == len(gen_points)

    gen_obs = event_observables(gen_points)
    g4_obs = event_observables(g4_points)
    gen_hits = flat_hits(gen_points)
    g4_hits = flat_hits(g4_points)

    def path(name):
        return os.path.join(out_dir, name)

    plot_hist(gen_obs["n_hits"], g4_obs["n_hits"], "hits per event",
              path("n_hits.pdf"))
    plot_hist(gen_obs["e_total"], g4_obs["e_total"],
              "total deposited energy [MeV]", path("e_total.pdf"), logx=True)
    ratio_gen = gen_obs["e_total"] / e_inc
    ratio_g4 = g4_obs["e_total"] / e_inc
    plot_response_profile(e_inc, ratio_gen, ratio_g4, path("response_profile.pdf"))
    # Bulk of *both* distributions, padded — asymmetric, because the muon
    # response is long-tailed and a range centred on the median wastes most of
    # the axis. The outer 0.5% lands in the end bins, so no tail is hidden.
    lo = min(np.percentile(ratio_g4, 0.5), np.percentile(ratio_gen, 0.5))
    hi = max(np.percentile(ratio_g4, 99.5), np.percentile(ratio_gen, 99.5))
    pad = 0.1 * max(hi - lo, 1e-3)
    plot_hist(
        ratio_gen,
        ratio_g4,
        "total deposited / incident energy",
        path("response.pdf"),
        logy=True,
        xrange=(max(0.0, lo - pad), hi + pad),
        show_moments=True,
    )
    plot_hist(gen_hits["edep"], g4_hits["edep"], "voxel energy [MeV]",
              path("hit_edep.pdf"), logx=True)
    for name in ["x", "y", "z"]:
        plot_hist(gen_hits[name], g4_hits[name], f"hit {name} [mm]",
                  path(f"hit_{name}.pdf"))
        plot_energy_profile(gen_hits, g4_hits, len(gen_points), name,
                            path(f"energy_profile_{name}.pdf"), per="event")
        plot_energy_profile(gen_hits, g4_hits, len(gen_points), name,
                            path(f"mean_hit_energy_{name}.pdf"), per="hit")
        plot_hist(gen_obs[f"extent_{name}"], g4_obs[f"extent_{name}"],
                  f"event {name} extent [mm]", path(f"extent_{name}.pdf"))
        plot_hist(gen_obs[f"centroid_{name}"], g4_obs[f"centroid_{name}"],
                  f"event energy-weighted {name} centroid [mm]",
                  path(f"centroid_{name}.pdf"))
    plot_event_displays(gen_points, g4_points, path("event_displays.pdf"))

    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
