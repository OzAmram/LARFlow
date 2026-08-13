# lardiff

Flow-matching point-cloud generative model for single-particle energy deposits
in liquid argon, adapted from
[AllShowers](https://github.com/FLC-QU-hep/AllShowers)
([arXiv:2601.11716](https://arxiv.org/abs/2601.11716)).

An event is a variable-length cloud of voxel hits `[x, y, z, edep]` (5 mm
pitch). A conditional-flow-matching (rectified flow) model with a
permutation-equivariant transformer backbone learns the distribution of hits
conditioned on the incident particle energy and the number of hits, and
generates events by integrating the learned ODE from Gaussian noise.

v1 targets negative muons (PDG 13) only and builds the complete pipeline:
preprocessing, training, generation, and validation plots. v2 adds an
independently trained electron (PDG 11) model using the same code and an
8192-point cache; the two are separate runs, not a joint multi-particle model.

## Relation to AllShowers

`ode_solvers.py`, `preprocessing.py`, `util.py`, and `data_loader.py` are
near-verbatim copies; `transformer.py`, `flow_matching.py`, and `train.py`
are simplified adaptations. The LAr-specific changes:

- **All three coordinates are generated.** AllShowers treats the shower depth
  as a discrete calorimeter-layer index supplied as conditioning (layer
  embedding + per-layer multiplicity histogram). Liquid argon has no layer
  structure, so points are 4D `[x, y, z, edep]` and the layer machinery is
  removed.
- **Full attention.** The calorimeter layer-neighborhood attention mask is
  replaced by plain padding-masked full attention. The padding `BlockMask` is
  built analytically from the per-event hit count via
  `BlockMask.from_kv_blocks` (possible because padding is always a contiguous
  suffix), avoiding `create_block_mask` entirely — see *torch 2.5 notes*.
- **Scalar conditioning.** The model conditions on `[log E_incident, log N]`
  through the global conditioning token, replacing the per-layer histogram
  MLP. At generation time N comes either from held-out truth (for one-to-one
  validation) or from an empirical bootstrap of P(N|E) (40 log-spaced energy
  bins over the training set) — the stand-in for AllShowers' external
  PointCountFM model.
- **Direct HDF5 data path.** The `showerdata` package is replaced by a
  one-time preprocessing script plus a plain h5py loader. The `rangerlite`
  optimizer dependency is dropped (AdamW default).

Training infrastructure (checkpoint/resume, DDP via torchrun, torch.compile,
schedulers, loss bookkeeping) is inherited unchanged from AllShowers.

## Repository layout

```
conf/lar_muon.yaml            # production config (full cache, 4096 points)
conf/lar_electron.yaml        # electron config (8192 points, batch 64, 40 epochs)
conf/lar_muon_mini.yaml       # small config for smoke tests
scripts/preprocess_lar.py     # raw voxel file -> per-particle padded cache
scripts/train_perlmutter.sh   # single-GPU sbatch script (NERSC Perlmutter)
lardiff/
  transformer.py              # flex-attention encoder + analytic padding BlockMask
  flow_matching.py            # CNF: flow-matching loss + ODE sampling
  ode_solvers.py              # euler / heun / midpoint integrators
  preprocessing.py            # Log / Affine / StandardScaler transformations
  data_loader.py              # in-RAM dataset / loader
  lar_data.py                 # cache reading, trafo fitting, train/val loaders
  train.py                    # Trainer + CLI
  generator.py                # LArGenerator + EmpiricalNSampler + CLI
  evaluate.py                 # Geant4-vs-model validation plots + CLI
```

## Data

Raw input: `/global/cfs/cdirs/m2612/ozamram/LAR_Diffu/lar_muon_voxels.h5` —
1M ToyG4 events (9 particle species, 10 MeV–10 GeV) with CSR-style
`edep_offsets` into flat per-hit arrays.

Preprocessing filters one PDG code, sorts each event's hits by descending
edep, truncates at `--max-points` (keeping the highest-energy hits; 4096
covers >99.5% of muon events intact) and writes a dense float32 cache:

| dataset      | shape               | content                              |
|--------------|---------------------|--------------------------------------|
| `points`     | (N_ev, max_points, 4) | `[x_mm, y_mm, z_mm, edep_MeV]`, zero-padded |
| `energy_MeV` | (N_ev,)             | incident kinetic energy              |
| `n_points`   | (N_ev,)             | real hits per event (after truncation) |
| `orig_index` | (N_ev,)             | event index in the raw file          |

Padding rows are exact zeros; `edep > 0` identifies real hits everywhere.
Because hits are sorted by descending edep, a smaller `max_num_points` at
load time keeps the most important hits for free.

Normalization (fitted on ≤100k training events, saved to
`results/<run>/preprocessing/trafos.pt`): per-axis StandardScaler for
coordinates, `log(edep + 1e-6)` + StandardScaler for the ~9-decade voxel
energy spectrum, elementwise `log` + StandardScaler for the `(E, N)`
condition.

## Usage

```bash
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python   # torch 2.5.1 + h5py
$PY -m pip install -e .                                # once

# 1. one-time preprocessing (a few minutes; run where CFS I/O is fast)
$PY scripts/preprocess_lar.py \
    --input /global/cfs/cdirs/m2612/ozamram/LAR_Diffu/lar_muon_voxels.h5 \
    --output /global/cfs/cdirs/m2612/ozamram/LAR_Diffu/cache/lar_pdg13_maxp4096.h5 \
    --pdg 13 --max-points 4096

# 2. training  (~4.5 min/epoch on one A100 at batch 128; auto-resumes from
#    checkpoints/last.pt if the result dir already exists in the conf copy)
$PY lardiff/train.py conf/lar_muon.yaml            # interactive
sbatch scripts/train_perlmutter.sh conf/lar_muon.yaml   # batch queue
$PY lardiff/train.py conf/lar_muon.yaml --fast-dev-run  # 2-epoch smoke test

# 3. generation — conditions on the validation tail of the cache
$PY -m lardiff.generator results/<run> <cache.h5> -n 2000 --n-source truth
$PY -m lardiff.generator results/<run> <cache.h5> -n 2000 --n-source empirical

# 4. validation plots -> results/<run>/eval_samplesNN/
$PY -m lardiff.evaluate results/<run>/samples00.h5 <cache.h5>
```

Multi-GPU: `torchrun --standalone --nproc_per_node=4 lardiff/train.py
conf/lar_muon.yaml --ddp` (global batch size is divided across ranks).

Evaluation plots: hit multiplicity, total deposited energy,
E_dep/E_inc response profile, per-voxel energy spectrum, hit position
distributions, per-axis event extents and energy-weighted centroids,
per-axis energy profiles (mean deposited energy per event, and mean hit
energy, vs. x/y/z), and side-by-side truth/model event displays.

## v1 results (muons, 100 epochs, ~1.3M parameters)

Validation on 2000 held-out events (train 0.136 / val 0.133 loss, no
overfitting): total deposited energy and per-voxel spectrum medians agree to
<1%, energy response 1.17 ± 0.5 in both, longitudinal energy profile matches
over ~8 decades. Empirical-N generation reproduces the truth multiplicity
distribution (KS p = 0.90) with unchanged physics observables. Known
imperfection: mean transverse (x/y) extents are ~5% low — a tail effect; the
medians agree.

## v2 results (electrons, 40 epochs, same ~1.3M-parameter architecture)

`conf/lar_electron.yaml`: cap 8192 points (5.8% of events truncated, ~0.05% of
mean energy lost), batch 64, 40 epochs — 18 h on one A100, ~27 min/epoch.
Train 0.2714 / val 0.2706 at the best epoch; validation tracked training
throughout. The loss is *not* comparable to the muon run's 0.13 (different
target distribution, ~3x the hits per event, much softer voxel spectrum).

Validation on 2000 held-out events, truth-N:

| observable                | model  | Geant4 |
|---------------------------|--------|--------|
| median total edep [MeV]   | 805.5  | 803.1  |
| median voxel edep [MeV]   | 0.1085 | 0.1085 |
| response E_dep/E_inc      | 0.9953 ± 0.043 | 0.9964 ± 0.014 |
| per-event E_total corr.   | 0.9997 | —      |
| median hit rms about centroid, x / z [mm] | 137.3 / 382.1 | 135.9 / 377.7 |

The voxel energy spectrum agrees to ~1% at every percentile from the 1st
(0.53 keV) to the 99.9th (9.3 MeV). Per-axis energy profiles agree to 4–7% rms
in the core bins. Empirical-N generation is nearly indistinguishable from
truth-N here (per-event energy correlation 0.9997 either way), because P(N|E)
is narrow for electrons — sampled N lands within 7.5% of the true N for half
the events. This is unlike muons, where the same swap cost a visible amount of
correlation (0.999 → 0.995).

Two known imperfections:

- **Response spread is ~3x too wide** (0.043 vs 0.014), consistently across the
  energy range. Geant4 electrons below ~57 MeV are fully contained and deposit
  essentially all their energy (spread ≈ 0), and the model does not reproduce
  how deterministic that containment is. The *mean* response is correct.
- **Event extents run 3–5% high in the median and further out in the tail**
  (p90 z-extent 6369 vs 5055 mm). The halo population itself is right — the
  fraction of hits more than 0.5/1/2 m from the event centroid matches Geant4
  to about a percent (0.251/0.0489/0.0052 vs 0.248/0.0490/0.0049) — but the
  slight excess in the >2 m tail is amplified by extent being a min–max
  statistic. Those hits carry 0.16% of the event energy.

### Generated coordinates are continuous, not voxelized

Worth knowing before using samples downstream, and true of **both** models: the
training data sits exactly on the 5 mm voxel grid, but the model emits
continuous coordinates with no learned grid structure — the offset from the
nearest voxel centre is uniformly distributed (mean 1.250 mm, versus 1.25 for
pure noise and 0.000 for Geant4). Snapping generated hits back onto the grid
therefore produces collisions: ~5% of electron hits and ~3% of muon hits land
on an already-occupied voxel, so a snap step must merge duplicates and sum
their energies rather than assume uniqueness.

## torch 2.5 notes

The pinned environment (torch 2.5.1+cu121) has three flex-attention pitfalls
this repo works around; all disappear with torch ≥ 2.6:

- `torch.compile(create_block_mask)` fails for mask functions that capture a
  tensor (vmap `.item()` error) on both CPU and CUDA, and the uncompiled
  version materializes a dense (B, P, P) mask (~19 GB at batch 128 × 4096).
  `compute_mask` therefore builds the BlockMask analytically (verified
  bit-exact against the reference).
- The compiled flex-attention kernel requires head_dim ≥ 16
  (`dim_embedding / num_head`).
- `torch.nn.utils.get_total_norm` does not exist yet (local fallback in
  `train.py`).

## Next steps (v2 candidates)

- Narrow the electron energy-response spread — the clearest quantitative gap.
  Candidates: larger model (dim 256 / 8 blocks, set aside for cost), per-event
  loss normalization so high-multiplicity events aren't down-weighted, or
  conditioning on total deposited energy in addition to N.
- Optional grid snapping with duplicate merging in `generator.py`, for
  consumers that need true voxel output.
- Learned multiplicity model P(N|E) replacing the bootstrap sampler. Lower
  priority for electrons (P(N|E) is narrow, so the bootstrap is already close
  to truth-N) than for muons.
- Fewer ODE steps at generation (200 Heun steps ≈ 25 s per 128-event batch at
  4096 points, ~125 s at 8192); distillation or a timestep study. This is now
  the main cost: generating 2000 electron events takes ~33 min.
- Condition on initial particle direction; extend to the other 8 species in
  the dataset via the (already present, disabled) particle embedding.
- Per-event loss normalization and OT noise–data matching (upstream
  `OT_match.py`) for training-efficiency studies.
