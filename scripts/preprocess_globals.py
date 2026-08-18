"""Build the per-event global cache used to train the p(N, R | E, type) model.

The global model only needs four scalars per event -- particle type, incident
energy, hit count and total deposited energy -- so this script never
materializes a point cloud and covers all species in the raw file at once.
It reads only edep_MeV_flat, in event-aligned sequential blocks.

    pdg                 (N_ev,)  PDG code
    energy_MeV          (N_ev,)  incident kinetic energy
    n_hits              (N_ev,)  hits in the event, untruncated
    edep_total_MeV      (N_ev,)  energy in the top --max-points hits
    n_hits_full         (N_ev,)  same as n_hits, kept for symmetry
    edep_total_full_MeV (N_ev,)  untruncated deposited energy

The two quantities need opposite treatment, because truncation acts on them
differently.  It only ever *clamps* the hit count, so leaving n_hits
untruncated and clamping at generation time reproduces the capped distribution
exactly -- point mass at the cap included.  Capping it here instead would make
the flow responsible for that point mass, which a continuous flow over log N
cannot express.

The energy has no such escape: truncation drops the lowest-edep hits, and no
clamp undoes that.  --max-points K therefore sums only the K highest-edep hits.
This matters more than the dropped energy fraction suggests -- that fraction is
~0.05% of the mean for electrons at 8192 points, but ~6% of events exceed the
cap, and dropping any hit from a fully contained event moves it off R = 1.  An
untruncated globals file puts 85% of electrons on the containment atom where
the capped cache has 82%, and a model fit on it over-produces contained events
by that much.

One cap covers the whole file, so it should match the point-cloud cache the
model will be paired with.  Muons are the only species where this is lossy
across caps (0.4% exceed 4096 against 0.01% exceeding 8192), and they carry no
containment atom, so a single 8192 file serves every cache in use.
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
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="cap hit counts here and sum only the K highest-edep hits, "
             "matching a point-cloud cache built with the same cap",
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

        edep_full = np.zeros(n_events, dtype=np.float64)
        edep_total = np.zeros(n_events, dtype=np.float64)
        edep_flat = f["edep_MeV_flat"]
        n_capped = 0
        for lo in range(0, n_events, args.block_events):
            hi = min(lo + args.block_events, n_events)
            o0, o1 = offsets[lo], offsets[hi]
            if o1 == o0:
                continue
            chunk = edep_flat[o0:o1]
            # hit -> local event index, so empty events stay at zero
            local = np.repeat(np.arange(hi - lo), counts[lo:hi])
            edep_full[lo:hi] = np.bincount(
                local, weights=chunk, minlength=hi - lo
            )
            edep_total[lo:hi] = edep_full[lo:hi]
            if args.max_points is not None:
                # only the over-cap events need redoing, ~6% outside muons
                over = np.nonzero(counts[lo:hi] > args.max_points)[0]
                n_capped += len(over)
                for j in over:
                    event = chunk[offsets[lo + j] - o0: offsets[lo + j + 1] - o0]
                    kth = event.size - args.max_points
                    edep_total[lo + j] = np.partition(event, kth)[kth:].sum()
            if (lo // args.block_events) % 10 == 0:
                log(f"  events {lo}-{hi}")

        # n_hits deliberately stays untruncated: truncation only ever *clamps*
        # the count, so GlobalSampler(max_num_points=K) reproduces the capped
        # distribution exactly, point mass at K included.  Capping it here
        # instead would force the flow to represent that point mass itself,
        # which a continuous flow over log N cannot do -- measured as KS(N)
        # rising from 0.008 to 0.033 on e+.  The energy has no such escape:
        # dropping hits changes it in a way no clamp can undo, so it is capped.
        counts_full = counts
        if args.max_points is not None:
            log(f"capped the energy of {n_capped} events "
                f"({n_capped / n_events:.2%}) to their top {args.max_points} hits")

    with h5py.File(args.output, "w") as f:
        f.create_dataset("pdg", data=pdg.astype(np.int32))
        f.create_dataset("energy_MeV", data=energy.astype(np.float32))
        f.create_dataset("n_hits", data=counts.astype(np.int32))
        f.create_dataset("edep_total_MeV", data=edep_total.astype(np.float32))
        f.create_dataset("n_hits_full", data=counts_full.astype(np.int32))
        f.create_dataset("edep_total_full_MeV", data=edep_full.astype(np.float32))
        f.attrs["source_file"] = args.input
        f.attrs["untruncated"] = args.max_points is None
        if args.max_points is not None:
            f.attrs["max_points"] = args.max_points
    log(f"wrote {args.output}")

    print()
    ratio = np.divide(edep_total, energy, out=np.zeros_like(edep_total),
                      where=energy > 0)
    ratio_full = np.divide(edep_full, energy, out=np.zeros_like(edep_full),
                           where=energy > 0)
    print(f"{'pdg':>7s} {'events':>8s} {'N median':>9s} {'N max':>7s} "
          f"{'R median':>9s} {'R mean':>8s} {'R p99':>8s} "
          f"{'R==1':>7s} {'uncapped':>9s}")
    for code in np.unique(pdg):
        sel = (pdg == code) & (counts > 0)
        # fraction sitting on the containment atom, before and after capping --
        # the gap is exactly the bias a mismatched globals file would inject
        atom = np.mean(np.abs(ratio[sel] - 1) <= 1e-5)
        atom_full = np.mean(np.abs(ratio_full[sel] - 1) <= 1e-5)
        print(f"{code:>7d} {sel.sum():>8d} {np.median(counts[sel]):>9.0f} "
              f"{counts[sel].max():>7d} {np.median(ratio[sel]):>9.4f} "
              f"{ratio[sel].mean():>8.4f} {np.percentile(ratio[sel], 99):>8.4f} "
              f"{atom:>7.3f} {atom_full:>9.3f}")
    empty = int((counts == 0).sum())
    print(f"\nevents with no hits: {empty}")


if __name__ == "__main__":
    main()
