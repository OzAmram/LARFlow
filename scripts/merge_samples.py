"""Concatenate sample files produced by parallel generator jobs.

Generation of a large sample is split into chunks so that the work runs in
parallel and a job that dies costs only its own chunk.  Each chunk is a normal
samples file covering a contiguous slice of the cache; this stitches them back
into one, in cache order, so `lardiff.evaluate` sees what it would have seen
from a single run.

    python scripts/merge_samples.py <out.h5> <chunk.h5> [<chunk.h5> ...]
"""

import argparse
import os

import h5py
import numpy as np

DATASETS = [
    "points", "energy_MeV", "n_points_used", "n_points_truth",
    "ratio_used", "ratio_truth", "cache_index", "pdg",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="merged samples h5 to write")
    parser.add_argument("chunks", nargs="+", help="chunk files to merge")
    parser.add_argument(
        "--allow-gaps", action="store_true",
        help="merge even if the chunks do not tile a contiguous range; the "
             "result is still usable but evaluate reads truth as one slice, so "
             "a gap there would misalign it",
    )
    args = parser.parse_args()

    chunks = []
    for path in args.chunks:
        if not os.path.exists(path):
            print(f"missing, skipping: {path}")
            continue
        with h5py.File(path, "r") as f:
            present = [k for k in DATASETS if k in f]
            chunks.append((int(f["cache_index"][0]), path,
                           {k: f[k][:] for k in present},
                           dict(f.attrs)))
    if not chunks:
        raise SystemExit("no chunk files found")
    chunks.sort(key=lambda c: c[0])

    index = np.concatenate([c[2]["cache_index"] for c in chunks])
    expected = np.arange(index[0], index[0] + len(index))
    contiguous = np.array_equal(index, expected)
    if not contiguous:
        missing = sorted(set(expected.tolist()) - set(index.tolist()))
        msg = (f"chunks do not tile a contiguous range: {len(missing)} events "
               f"missing, first few {missing[:5]}")
        if not args.allow_gaps:
            raise SystemExit(msg + "\n(re-run the failed chunk, or pass --allow-gaps)")
        print("WARNING: " + msg)

    keys = set(chunks[0][2])
    for _, path, data, _ in chunks:
        if set(data) != keys:
            raise SystemExit(f"{path} has different datasets than the first chunk")

    with h5py.File(args.output, "w") as out:
        for key in keys:
            out.create_dataset(
                key, data=np.concatenate([c[2][key] for c in chunks], axis=0)
            )
        for k, v in chunks[0][3].items():
            out.attrs[k] = v
        out.attrs["merged_from"] = [os.path.basename(c[1]) for c in chunks]

    print(f"merged {len(chunks)} chunks -> {args.output}")
    print(f"  {len(index)} events, cache index {index[0]}..{index[-1]}, "
          f"contiguous: {contiguous}")
    if "pdg" in keys:
        codes, counts = np.unique(
            np.concatenate([c[2]["pdg"] for c in chunks]), return_counts=True
        )
        print("  per species: " + ", ".join(
            f"{c}:{n}" for c, n in zip(codes.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
