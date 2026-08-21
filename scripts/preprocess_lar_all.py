"""Convert the raw LAr voxel file into a multi-species training cache.

Unlike preprocess_lar.py, which writes one particle type as a dense padded
array, this keeps every species and stores the hits *packed* with CSR-style
offsets, the same way the raw file does.  Padding to max_points is what makes
the dense form unaffordable here: a million events at 8192 points is 131 GB,
which does not fit in a node's memory, while the hits themselves are only
~26 GB.  lardiff.lar_data.H5DataSet pads each batch as it reads it.

Events are stored in raw-file order and the shuffle lives in a `perm` dataset
instead, so that both the read here and the write stay sequential.  Consumers
treat `perm[j]` as logical event j, which is what makes a tail slice of the
cache a species-balanced random holdout.

    hits       (H, 4)    [x_mm, y_mm, z_mm, edep_MeV], sorted by -edep per event
    offsets    (E + 1,)  hits[offsets[i]:offsets[i + 1]] belongs to event i
    energy_MeV (E,)
    n_points   (E,)      hits kept for the event (after truncation)
    pdg        (E,)      PDG code
    label      (E,)      index into the `species` attribute
    perm       (E,)      logical order; logical event j is physical perm[j]
    orig_index (E,)      event index in the raw file
"""

import argparse
import sys
import time

import h5py
import numpy as np

FLAT_KEYS = ["cube_x_mm_flat", "cube_y_mm_flat", "cube_z_mm_flat", "edep_MeV_flat"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw LAr voxel h5 file")
    parser.add_argument("--output", required=True, help="cache h5 file to write")
    parser.add_argument(
        "--pdg", type=int, nargs="*", default=None,
        help="species to keep; default is every species in the file",
    )
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument(
        "--stop", type=int, default=None,
        help="only consider the first STOP events of the raw file (for testing)",
    )
    parser.add_argument(
        "--block-events", type=int, default=20_000,
        help="events per sequential read/write pass",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for perm")
    args = parser.parse_args()
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}")
        sys.stdout.flush()

    with h5py.File(args.input, "r") as f:
        pdg_all = f["pdgCode"][: args.stop]
        offsets_all = f["edep_offsets"][
            : None if args.stop is None else args.stop + 1
        ].astype(np.int64)
        energy_all = f["energy_MeV"][: args.stop]
    counts_all = np.diff(offsets_all)

    keep = counts_all > 0
    if args.pdg:
        keep &= np.isin(pdg_all, args.pdg)
    selected = np.where(keep)[0]
    if len(selected) == 0:
        raise SystemExit("no events selected")

    species = np.unique(pdg_all[selected])
    label_of = {int(p): i for i, p in enumerate(species)}
    sel_counts = counts_all[selected]
    n_points = np.minimum(sel_counts, args.max_points).astype(np.int32)
    n_truncated = int((sel_counts > args.max_points).sum())
    n_events = len(selected)
    out_offsets = np.concatenate([[0], np.cumsum(n_points, dtype=np.int64)])
    n_hits = int(out_offsets[-1])
    log(f"{n_events} events, {len(species)} species {list(species)}, "
        f"{n_hits} hits kept ({n_hits * 16 / 1e9:.1f} GB), "
        f"{n_truncated} events truncated")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_events)

    with h5py.File(args.output, "w") as out, h5py.File(args.input, "r") as f:
        # uncompressed: H5DataSet reads scattered events every batch, and
        # gzip would force a whole chunk to be decompressed for each one
        dset = out.create_dataset(
            "hits", shape=(n_hits, 4), dtype=np.float32, chunks=(65536, 4),
        )
        for block_start in range(0, n_events, args.block_events):
            block_stop = min(block_start + args.block_events, n_events)
            block = selected[block_start:block_stop]
            # events in a block are contiguous in the raw file only if the
            # selection is dense, so read the span and index into it
            lo, hi = int(offsets_all[block[0]]), int(offsets_all[block[-1] + 1])
            raw = np.empty((hi - lo, 4), dtype=np.float32)
            for j, key in enumerate(FLAT_KEYS):
                raw[:, j] = f[key][lo:hi]

            buf = np.empty((out_offsets[block_stop] - out_offsets[block_start], 4),
                           dtype=np.float32)
            base = out_offsets[block_start]
            for i, ev in enumerate(block, start=block_start):
                a = int(offsets_all[ev]) - lo
                b = int(offsets_all[ev + 1]) - lo
                ev_hits = raw[a:b]
                order = np.argsort(-ev_hits[:, 3])[: n_points[i]]
                buf[out_offsets[i] - base: out_offsets[i + 1] - base] = ev_hits[order]
            dset[out_offsets[block_start]: out_offsets[block_stop]] = buf
            log(f"events {block_stop}/{n_events} "
                f"({100 * block_stop / n_events:.1f}%)")

        out.create_dataset("offsets", data=out_offsets)
        out.create_dataset("energy_MeV", data=energy_all[selected].astype(np.float32))
        out.create_dataset("n_points", data=n_points)
        out.create_dataset("pdg", data=pdg_all[selected].astype(np.int32))
        out.create_dataset(
            "label",
            data=np.array([label_of[int(p)] for p in pdg_all[selected]], dtype=np.int8),
        )
        out.create_dataset("perm", data=perm.astype(np.int64))
        out.create_dataset("orig_index", data=selected.astype(np.int64))
        out.attrs["species"] = species.astype(np.int32)
        out.attrs["max_points"] = args.max_points
        out.attrs["voxel_pitch_mm"] = 5.0
        out.attrs["n_truncated"] = n_truncated
        out.attrs["source_file"] = args.input
        out.attrs["seed"] = args.seed
    log(f"wrote {args.output}")

    print()
    print(f"{'pdg':>8} {'label':>6} {'events':>8} {'medN':>7} {'meanN':>8} {'maxN':>7}")
    lab = np.array([label_of[int(p)] for p in pdg_all[selected]])
    for p in species:
        m = pdg_all[selected] == p
        print(f"{p:>8} {label_of[int(p)]:>6} {m.sum():>8} "
              f"{np.median(n_points[m]):>7.0f} {n_points[m].mean():>8.0f} "
              f"{n_points[m].max():>7}")
    print(f"\nperm seed {args.seed}: first 8 logical events are physical "
          f"{perm[:8]} with labels {lab[perm[:8]]}")


if __name__ == "__main__":
    main()
