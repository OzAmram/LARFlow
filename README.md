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
conf/lar_electron_v3.yaml     # as above + energy-ratio conditioning (3-dim cond)
conf/lar_muon_mini.yaml       # small config for smoke tests
conf/lar_muon_mini_v3.yaml    # smoke test with 3-dim cond
scripts/preprocess_lar.py     # raw voxel file -> per-particle padded cache
scripts/preprocess_globals.py # all 9 species -> (E, N, E_dep) cache for the global model
scripts/train_perlmutter.sh   # single-GPU sbatch script (NERSC Perlmutter)
scripts/train_global_perlmutter.sh  # same, for the global model
lardiff/
  transformer.py              # flex-attention encoder + analytic padding BlockMask
  flow_matching.py            # CNF: flow-matching loss + ODE sampling
  ode_solvers.py              # euler / heun / midpoint integrators
  preprocessing.py            # Log / Affine / StandardScaler transformations
  data_loader.py              # in-RAM dataset / loader
  lar_data.py                 # cache reading, trafo fitting, train/val loaders
  train.py                    # Trainer + CLI
  global_model.py             # p(N, R | E, type) containment mixture + sampler + CLI
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
condition — `(E, N, R)` in v3 configs, where `R` is computed from the possibly
truncated cache so it is exactly what the model can reproduce.

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

# 3. global model p(N, R | E, type) — all 9 species, ~5 min on one A100
$PY scripts/preprocess_globals.py \
    --input /global/cfs/cdirs/m2612/ozamram/LAR_Diffu/lar_muon_voxels.h5 \
    --output /global/cfs/cdirs/m2612/ozamram/LAR_Diffu/cache/lar_globals.h5
sbatch scripts/train_global_perlmutter.sh results/global_all_species_v3

# 4. generation — conditions on the validation tail of the cache
$PY -m lardiff.generator results/<run> <cache.h5> -n 2000 --n-source truth
$PY -m lardiff.generator results/<run> <cache.h5> -n 2000 --n-source empirical
# v3: take N and R from the global model, and enforce R by rescaling
$PY -m lardiff.generator results/<run> <cache.h5> -n 2000 --n-source global \
    --global-model results/global_all_species_v3 --pdg 11 --renormalize

# 5. validation plots -> results/<run>/eval_samplesNN/
$PY -m lardiff.evaluate results/<run>/samples00.h5 <cache.h5>
```

Multi-GPU: `torchrun --standalone --nproc_per_node=4 lardiff/train.py
conf/lar_muon.yaml --ddp` (global batch size is divided across ranks).

Evaluation plots: hit multiplicity, total deposited energy,
E_dep/E_inc response (both a histogram, with mean ± std in the legend, and a
profile vs. incident energy), per-voxel energy spectrum, hit position
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

## v3: factorized global model

v2's clearest gap was the energy response: the point model's spread was ~3x too
wide (0.043 vs 0.014). Global quantities like a per-event energy ratio are hard
for a diffusion model that acts on points locally — nothing in the loss ties the
4096 independent hit energies together. v3 therefore factorizes the problem: a
small model predicts the event's global summary first, and the point model
receives it as conditioning.

The global model is `p(N, R | E, type)`, where `N` is the hit count and
`R = E_deposited / E_incident` the energy response. It is trained on all 9
species at once with a particle embedding (one model, ~205k parameters, 4–5 min
on one A100 for 300 epochs over 1M events — about 0.4% of the point model's
cost). The point models stay per-species and gain `log R` as a third
conditioning input alongside `log E` and `log N`. `generator.py --renormalize`
optionally rescales each generated event's hit energies so its total matches the
conditioned `R` exactly.

### Why a plain flow is not enough

83% of electrons are *fully contained*: they deposit exactly their incident
energy, so `R` is precisely 1.0 — not a narrow peak but a genuine point mass.
`p(R | E)` is therefore a **mixed** distribution, an atom plus a continuous
escape tail, and a continuous flow cannot represent an atom however long it
trains. A single joint flow over `(log N, log R)` smears it into a bump, which
put 35% of sampled electrons at `R > 1` (energy non-conservation) and left only
38% near 1.0 against a true 83%. Under `--renormalize` that defect would
propagate straight into the generated showers.

### Finding the atom

The atom is located in the data before any training — it is a labelling step,
not part of the model. The physical fact it exploits: a fully contained event
deposits a *fixed* amount relative to its incident energy, so
`delta = edep - E` takes the same value for every contained event of a species,
independent of energy. Detection is then just *does one value of `delta` get
shared by many of this species' events?* — find the densest window in `delta`
(each event's own `delta` as a candidate centre, counting neighbours within
`1e-5 * E`), recentre on the window, and declare an atom if it holds >2% of the
species.

Two details matter. The search must run in `delta`, not `delta / E`: the e+ atom
sits at a fixed *energy*, so in ratio space it smears across three decades of
incident energy. And the winning candidate is one event's own `delta`, which
carries ~1e-5 relative float64 summation noise, so the centre must be refined
(median over the window) before the fraction is measured — otherwise the narrow
low-energy tolerances all fall outside it and the atom is missed. An earlier
version located the atom by the median of `delta`, which only works when the
atom holds a *majority* of events; protons are 29% contained, so the median
landed in the escape tail and found nothing.

The detector is given no physics, only `edep` and `E`. It recovers:

| species        | offset      | contained | interpretation                  |
|----------------|-------------|-----------|---------------------------------|
| e-             | +0.0000 MeV | 0.832     | deposits exactly its kinetic energy |
| gamma          | +0.0000 MeV | 0.869     | same                            |
| e+             | +1.0220 MeV | 0.832     | 2 m_e c^2 — annihilation        |
| p              | +0.0000 MeV | 0.293     | stops below ~200 MeV            |
| mu±, pi±, n    | none        | <0.02     | no atom — plain flow            |

The positron offset landing on 2 m_e c^2 to four decimals is a useful check that
the detector finds real structure rather than fitting noise. Muons have no atom
(0.0000 of events within 1e-4 of the contained value) despite a very sharp
*continuous* peak — 41% of mu- land within 2% of `R = 1` — and are correctly
left to the plain flow.

### The mixture

```
p(N, R | E, type) = P(contained | E, type) * p(N | E, type, contained) * delta(R - ceiling)
                  + (1 - P)                * p(N, R | E, type, escaped)
```

with `ceiling = (E + offset) / E`. Three trained components:

1. **Containment classifier** — `(E, type) -> P(contained)`, a BCE-trained MLP.
2. **1-D flow over `log N`**, trained on contained events only.
3. **2-D joint flow over `(log N, log R)`**, trained on escaped events only.

At sampling time containment is drawn from the Bernoulli. If contained, `R` is
set to the ceiling deterministically — no sampling — but **`N` is still sampled**,
from flow (2). If escaped, both come from flow (3), with `R` clamped below the
ceiling since an escaping event deposits less than a contained one by
definition.

`N` needs its own flow because contained and escaped events have genuinely
different multiplicity distributions (contained events are the lower-energy ones
that stop in the volume), so reusing the joint flow would import the escaped
population's `N`. Sampling the joint flow for contained events does not work
either, because `R` is degenerate there — precisely the pathology being avoided.
The contained branch therefore drops the degenerate dimension and models only
the free one. Species with no detected atom bypass the mixture entirely.

### Global model results

50,000 held-out events, KS against Geant4. Two-sample KS has a ~0.026 critical
value at these sample sizes.

| species | KS(N) | KS(R) | atom (model / Geant4) |
|---------|-------|-------|-----------------------|
| pi-     | 0.014 | 0.016 | — |
| mu+     | 0.011 | 0.016 | — |
| e+      | 0.009 | 0.016 | 0.839 / 0.833 |
| e-      | 0.008 | 0.010 | 0.843 / 0.835 |
| mu-     | 0.011 | 0.083 | — |
| gamma   | 0.006 | 0.020 | 0.882 / 0.865 |
| pi+     | 0.014 | 0.013 | — |
| n       | 0.010 | 0.029 | — |
| p       | 0.012 | 0.015 | 0.297 / 0.298 |

The mixture took e- from KS(R) 0.487 to 0.010, gamma from 0.521 to 0.010, and
(once the detector was fixed) protons from 0.204 to 0.015, with no unphysical
`R > 1`. `N` was already excellent before the mixture and is unchanged.
Correlations `corr(log N, R)` are reproduced to ~0.01 for the strongly
correlated species (p -0.76, pi+ -0.65, e+ -0.63), so the joint structure
survives rather than just the two marginals.

The one weak entry is mu- at 0.083, which is a property of the statistic rather
than a physics error: the quantiles agree with Geant4 to ~0.2% throughout
(median 1.0258 vs 1.0249), but the CDF is nearly vertical through the sharp peak
at `R = 1`, so a 0.002 shift in `R` registers as KS 0.08. It is stable from 100
to 400 ODE steps, so it is peak placement, not integration error.

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

## Next steps

- A/B the v3 factorization on electrons: truth-`R` vs global-model `R`, each
  with and without `--renormalize`, against the v2 baseline. This is the open
  question — whether conditioning alone narrows the response spread, or whether
  enforcing `R` by rescaling is also needed.
- Optional grid snapping with duplicate merging in `generator.py`, for
  consumers that need true voxel output.
- Tighten the global model's mu- peak placement if it matters downstream (a
  5-minute retrain); see the v3 results above for why the KS overstates it.
- Fewer ODE steps at generation (200 Heun steps ≈ 25 s per 128-event batch at
  4096 points, ~125 s at 8192); distillation or a timestep study. This is now
  the main cost: generating 2000 electron events takes ~33 min.
- Condition on initial particle direction; extend to the other 8 species in
  the dataset via the (already present, disabled) particle embedding.
- Per-event loss normalization and OT noise–data matching (upstream
  `OT_match.py`) for training-efficiency studies.
