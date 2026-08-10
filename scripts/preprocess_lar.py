"""Convert the raw LAr voxel file into a per-particle padded training cache.

The raw file stores all events of all particle types with CSR-style offsets
into flat per-hit arrays.  This script selects one particle type, gathers its
hits with sequential chunked reads (fast on network file systems), sorts each
event's hits by descending deposited energy, truncates to --max-points and
writes a dense float32 cache:

    points     (N_ev, max_points, 4)  [x_mm, y_mm, z_mm, edep_MeV], zero-padded
    energy_MeV (N_ev,)
    n_points   (N_ev,)                number of real hits (after truncation)
    orig_index (N_ev,)                event index in the raw file
"""

import argparse
import sys
import time

import h5py
import numpy as np

FLAT_KEYS = ["cube_x_mm_flat", "cube_y_mm_flat", "cube_z_mm_flat", "edep_MeV_flat"]


def gather_hits(dataset, hit_mask, chunk_size):
    """Sequentially read a flat hit array, keeping only masked entries."""
    parts = []
    n_total = dataset.shape[0]
    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        chunk = dataset[start:stop]
        parts.append(chunk[hit_mask[start:stop]].astype(np.float32))
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw LAr voxel h5 file")
    parser.add_argument("--output", required=True, help="cache h5 file to write")
    parser.add_argument("--pdg", type=int, default=13)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument(
        "--stop", type=int, default=None,
        help="only consider the first STOP events of the raw file (for testing)",
    )
    parser.add_argument("--chunk-hits", type=int, default=100_000_000)
    args = parser.parse_args()
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}")
        sys.stdout.flush()

    with h5py.File(args.input, "r") as f:
        pdg = f["pdgCode"][: args.stop]
        offsets = f["edep_offsets"][: None if args.stop is None else args.stop + 1]
        offsets = offsets.astype(np.int64)
        energy = f["energy_MeV"][: args.stop]

        counts = np.diff(offsets)
        selected = np.where((pdg == args.pdg) & (counts > 0))[0]
        if len(selected) == 0:
            raise SystemExit(f"no events with pdg {args.pdg} found")
        sel_counts = counts[selected]
        log(f"selected {len(selected)} events with pdg {args.pdg}, "
            f"{sel_counts.sum()} hits")

        # boolean mask over all hits belonging to selected events
        event_is_selected = np.zeros(len(counts), dtype=bool)
        event_is_selected[selected] = True
        hit_mask = np.repeat(event_is_selected, counts)
        n_raw_hits = int(offsets[-1])

        hits = {}
        for key in FLAT_KEYS:
            dataset = f[key]
            if args.stop is not None:
                dataset = dataset[:n_raw_hits]
            hits[key] = gather_hits(dataset, hit_mask, args.chunk_hits)
            log(f"gathered {key}")

    max_points = args.max_points
    n_events = len(selected)
    points = np.zeros((n_events, max_points, 4), dtype=np.float32)
    n_points = np.minimum(sel_counts, max_points).astype(np.int32)
    n_truncated = int((sel_counts > max_points).sum())

    sel_offsets = np.concatenate([[0], np.cumsum(sel_counts)])
    for i in range(n_events):
        lo, hi = sel_offsets[i], sel_offsets[i + 1]
        edep = hits["edep_MeV_flat"][lo:hi]
        order = np.argsort(-edep)[: n_points[i]]
        points[i, : n_points[i], 0] = hits["cube_x_mm_flat"][lo:hi][order]
        points[i, : n_points[i], 1] = hits["cube_y_mm_flat"][lo:hi][order]
        points[i, : n_points[i], 2] = hits["cube_z_mm_flat"][lo:hi][order]
        points[i, : n_points[i], 3] = edep[order]
    log(f"built dense array {points.shape}, {n_truncated} events truncated")

    with h5py.File(args.output, "w") as f:
        f.create_dataset(
            "points", data=points,
            chunks=(min(256, n_events), max_points, 4),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset("energy_MeV", data=energy[selected].astype(np.float32))
        f.create_dataset("n_points", data=n_points)
        f.create_dataset("orig_index", data=selected.astype(np.int64))
        f.attrs["pdg"] = args.pdg
        f.attrs["max_points"] = max_points
        f.attrs["voxel_pitch_mm"] = 5.0
        f.attrs["n_truncated"] = n_truncated
        f.attrs["source_file"] = args.input
    log(f"wrote {args.output}")

    edep = points[:, :, 3][points[:, :, 3] > 0]
    print()
    print(f"events:       {n_events}")
    print(f"n_points:     median {np.median(n_points):.0f}  mean {n_points.mean():.1f}  "
          f"p99 {np.percentile(n_points, 99):.0f}  max {n_points.max()}")
    print(f"edep [MeV]:   min {edep.min():.3g}  median {np.median(edep):.3g}  "
          f"max {edep.max():.3g}")
    print(f"energy [MeV]: min {energy[selected].min():.1f}  max {energy[selected].max():.1f}")
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        coords = points[:, :, axis][points[:, :, 3] > 0]
        print(f"{name} [mm]:       min {coords.min():.1f}  max {coords.max():.1f}")


if __name__ == "__main__":
    main()
