"""Build the per-event global cache used to train the p(N, R | E, type) model.

The global model only needs four scalars per event -- particle type, incident
energy, hit count and total deposited energy -- so this script never
materializes a point cloud and covers all species in the raw file at once.
It reads only edep_MeV_flat, in event-aligned sequential blocks.

    pdg            (N_ev,)  PDG code
    energy_MeV     (N_ev,)  incident kinetic energy
    n_hits         (N_ev,)  hits in the event, untruncated
    edep_total_MeV (N_ev,)  summed deposited energy, untruncated

n_hits and edep_total are the *untruncated* values.  The point-cloud caches cap
the hit count, so a consumer that pairs this with a capped model clamps n_hits
to that cap -- which reproduces the capped distribution exactly, since
truncation only ever clamps the count.  The energy is a hair inconsistent in
the same situation (truncation drops the lowest-energy hits) but the dropped
fraction is ~0.05% of the mean for electrons at 8192 points.
"""

import argparse
import sys
import time

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw LAr voxel h5 file")
    parser.add_argument("--output", required=True, help="globals cache h5 to write")
    parser.add_argument(
        "--stop", type=int, default=None,
        help="only consider the first STOP events of the raw file (for testing)",
    )
    parser.add_argument(
        "--block-events", type=int, default=20_000,
        help="events per sequential read; bounds peak memory",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}")
        sys.stdout.flush()

    with h5py.File(args.input, "r") as f:
        pdg = f["pdgCode"][: args.stop]
        energy = f["energy_MeV"][: args.stop]
        offsets = f["edep_offsets"][
            : None if args.stop is None else args.stop + 1
        ].astype(np.int64)
        counts = np.diff(offsets)
        n_events = len(counts)
        log(f"{n_events} events, {offsets[-1]} hits, "
            f"{len(np.unique(pdg))} species")

        edep_total = np.zeros(n_events, dtype=np.float64)
        edep_flat = f["edep_MeV_flat"]
        for lo in range(0, n_events, args.block_events):
            hi = min(lo + args.block_events, n_events)
            o0, o1 = offsets[lo], offsets[hi]
            if o1 == o0:
                continue
            chunk = edep_flat[o0:o1]
            # hit -> local event index, so empty events stay at zero
            local = np.repeat(np.arange(hi - lo), counts[lo:hi])
            edep_total[lo:hi] = np.bincount(
                local, weights=chunk, minlength=hi - lo
            )
            if (lo // args.block_events) % 10 == 0:
                log(f"  events {lo}-{hi}")

    with h5py.File(args.output, "w") as f:
        f.create_dataset("pdg", data=pdg.astype(np.int32))
        f.create_dataset("energy_MeV", data=energy.astype(np.float32))
        f.create_dataset("n_hits", data=counts.astype(np.int32))
        f.create_dataset("edep_total_MeV", data=edep_total.astype(np.float32))
        f.attrs["source_file"] = args.input
        f.attrs["untruncated"] = True
    log(f"wrote {args.output}")

    print()
    ratio = np.divide(edep_total, energy, out=np.zeros_like(edep_total),
                      where=energy > 0)
    print(f"{'pdg':>7s} {'events':>8s} {'N median':>9s} {'N max':>7s} "
          f"{'R median':>9s} {'R mean':>8s} {'R p99':>8s}")
    for code in np.unique(pdg):
        sel = (pdg == code) & (counts > 0)
        print(f"{code:>7d} {sel.sum():>8d} {np.median(counts[sel]):>9.0f} "
              f"{counts[sel].max():>7d} {np.median(ratio[sel]):>9.4f} "
              f"{ratio[sel].mean():>8.4f} {np.percentile(ratio[sel], 99):>8.4f}")
    empty = int((counts == 0).sum())
    print(f"\nevents with no hits: {empty}")


if __name__ == "__main__":
    main()
