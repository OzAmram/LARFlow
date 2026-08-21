# lardiff

Flow-matching point-cloud generative model for single-particle energy deposits
in liquid argon, adapted from
[AllShowers](https://github.com/FLC-QU-hep/AllShowers)
([arXiv:2601.11716](https://arxiv.org/abs/2601.11716)).

An event is a variable-length cloud of voxel hits `[x, y, z, edep]` (5 mm
pitch). A conditional-flow-matching (rectified flow) model with a
permutation-equivariant transformer learns the distribution of hits given the
incident energy, the hit count and the deposited energy, and generates events by
integrating the learned ODE from Gaussian noise. A separate small model supplies
the per-event hit count and deposited energy, so the point model is told the
total it has to produce rather than having to discover it.

Results, validation plots and the history of what changed between versions are
in [results.md](results.md).

## Models

| purpose | run directory |
|---|---|
| electrons | `results/20260818_121435_LAr-electron-CNF-v4-cont` |
| muons | `results/20260810_154350_LAr-muon-CNF` |
| `p(N, R \| E, type)`, all species | `results/global_all_species_v5` |
| all species, type-conditioned | `results/*_LAr-allspecies-CNF-v6` (training) |

The global model covers all nine species and pairs with any point model.

## Layout

```
conf/                         # one yaml per run; *_mini are smoke tests
  lar_muon.yaml               #   muons, 4096 points
  lar_electron_v4.yaml        #   electrons, 8192 points
  lar_electron_v4_cont.yaml   #   warm start of the above, 80 more epochs
  lar_allspecies_v6.yaml      #   all 9 species, type-conditioned, 4-GPU
scripts/
  preprocess_lar.py           # raw file -> one species, dense padded cache
  preprocess_lar_all.py       # raw file -> all species, packed cache
  preprocess_globals.py       # raw file -> (E, N, E_dep) cache for the global model
  train_perlmutter.sh         # single-GPU sbatch
  train_resume_perlmutter.sh  # single-GPU, resumes an existing run
  train_ddp_perlmutter.sh     # 4-GPU DDP, resumes an existing run
  train_global_perlmutter.sh  # the global model
lardiff/
  transformer.py              # flex-attention encoder + analytic padding BlockMask
  flow_matching.py            # CNF: flow-matching loss + ODE sampling
  ode_solvers.py              # euler / heun / midpoint
  preprocessing.py            # Log / Affine / StandardScaler
  data_loader.py              # in-RAM dataset / loader
  lar_data.py                 # caches, trafo fitting, loaders (dense + packed)
  train.py                    # Trainer + CLI
  global_model.py             # p(N, R | E, type) + sampler + CLI
  generator.py                # LArGenerator + CLI
  evaluate.py                 # Geant4-vs-model plots + CLI
```

## Data

Raw input `/global/cfs/cdirs/m2612/ozamram/LAR_Diffu/lar_muon_voxels.h5`: 1M
ToyG4 events, 9 species, 10 MeV–10 GeV, CSR offsets into flat per-hit arrays.

Preprocessing sorts each event's hits by descending energy and truncates at
`--max-points`, so a smaller `max_num_points` at load time keeps the most
important hits for free. Two cache layouts, both accepted everywhere — the code
detects which by the presence of `hits`:

- **dense**, one species: `points (N_ev, max_points, 4)` zero-padded, plus
  `energy_MeV`, `n_points`, `orig_index`. Padding rows are exact zeros, so
  `edep > 0` identifies real hits.
- **packed**, all species: `hits (H, 4)` with `offsets (N_ev+1,)`, plus `pdg`
  and `label` per event. All nine species padded would be 131 GB, so they are
  stored packed (25 GB) and padded per batch.

Existing caches live in `/global/cfs/cdirs/m2612/ozamram/LAR_Diffu/cache/`.

## Usage

```bash
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python   # torch 2.5.1 + h5py
$PY -m pip install -e .                                # once
CACHE=/global/cfs/cdirs/m2612/ozamram/LAR_Diffu/cache
RAW=/global/cfs/cdirs/m2612/ozamram/LAR_Diffu/lar_muon_voxels.h5
```

### Generate

Needs no training. Samples land in `results/<run>/samplesNN.h5` at the next free
`NN`, with the arguments beside them as `samplesNN.yaml`.

```bash
# electrons, N and R from the global model, totals enforced (~60 min / 5000 events)
$PY -m lardiff.generator \
    results/20260818_121435_LAr-electron-CNF-v4-cont $CACHE/lar_pdg11_maxp8192.h5 \
    -n 5000 --n-source global --global-model results/global_all_species_v5 \
    --renormalize --solver heun --num-timesteps 200 --seed 0

# muons (predates the global model, so N comes from a bootstrap)
$PY -m lardiff.generator \
    results/20260810_154350_LAr-muon-CNF $CACHE/lar_pdg13_maxp4096.h5 \
    -n 2000 --n-source empirical

# all species: same command on the packed cache.  The species is read from the
# cache and fed to the model's particle embedding, so no extra flag is needed
$PY -m lardiff.generator \
    results/<allspecies_run> $CACHE/lar_all_species_maxp8192.h5 \
    -n 5000 --n-source global --global-model results/global_all_species_v5 \
    --renormalize
```

`--n-source` selects where the hit count `N` and response `R = E_dep / E_inc`
come from: `truth` (the held-out event itself, for one-to-one comparison),
`empirical` (bootstrap of `P(N|E)` over the training split), or `global` (a
trained global model; needs `--global-model`).

`--renormalize` rescales each event's hits so the total matches the drawn `R`
exactly. **You almost always want it**; it is not yet the default.

Generation draws from the **end** of the cache, which for the packed cache is
inside the held-out test region.

### Evaluate

```bash
# plots -> results/<run>/eval_samplesNN/
$PY -m lardiff.evaluate results/<run>/samples00.h5 $CACHE/lar_pdg11_maxp8192.h5

# one species out of a multi-species sample
$PY -m lardiff.evaluate results/<run>/samples00.h5 $CACHE/lar_all_species_maxp8192.h5 \
    --pdg 11 --out results/<run>/eval_electrons
```

Produces hit multiplicity, total deposited energy, response (histogram and
profile vs incident energy), per-voxel energy spectrum, hit positions, per-axis
extents and energy-weighted centroids, per-axis energy profiles, and truth/model
event displays.

### Train

```bash
# single GPU; resumes results/*_<run_name>/checkpoints/last.pt if it exists
sbatch scripts/train_resume_perlmutter.sh conf/lar_electron_v4.yaml

# four GPUs, same resume behaviour.  batch_size in the conf is the GLOBAL batch
sbatch scripts/train_ddp_perlmutter.sh conf/lar_allspecies_v6.yaml

# longer than the 24 h queue limit: chain jobs, each resuming the last
J=$(sbatch --parsable scripts/train_ddp_perlmutter.sh conf/lar_allspecies_v6.yaml)
for i in 1 2 3 4; do
    J=$(sbatch --parsable --dependency=afterany:$J \
        scripts/train_ddp_perlmutter.sh conf/lar_allspecies_v6.yaml)
done

# interactive, and a 2-epoch smoke test that writes to results/test
$PY lardiff/train.py conf/lar_electron_v4.yaml
$PY lardiff/train.py conf/lar_allspecies_v6.yaml --fast-dev-run
# note: the module entry point, not the `torchrun` script -- its shebang points
# at the interpreter the env was built with and does not resolve here
$PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    lardiff/train.py conf/... --ddp
```

Progress is `results/<run>/data/losses.txt` (train and validation, one row per
epoch) and `plots/losses.pdf`. Weights are `weights/best.pt` and
`weights/final.pt`, usually identical now that the validation loss is
deterministic.

**To continue a finished run, use `train.init_weights` — not a larger
`num_epochs`.** `CosineAnnealingLR.state_dict()` carries `T_max`, so resuming
restores the old horizon and the cosine climbs back up past its minimum.
`init_weights` takes only the weights, so the new run gets its own warmup and
decay. See `conf/lar_electron_v4_cont.yaml`. It applies only when no checkpoint
exists, so a preempted warm-started run still resumes normally.

### Rebuild the caches

They already exist; this is for reference.

```bash
$PY scripts/preprocess_lar.py --input $RAW \
    --output $CACHE/lar_pdg11_maxp8192.h5 --pdg 11 --max-points 8192

$PY scripts/preprocess_lar_all.py --input $RAW \
    --output $CACHE/lar_all_species_maxp8192.h5 --max-points 8192

# --max-points MUST match the point cache, or R means two different things
# in the two models
$PY scripts/preprocess_globals.py --input $RAW \
    --output $CACHE/lar_globals_maxp8192.h5 --max-points 8192
sbatch scripts/train_global_perlmutter.sh results/global_all_species_v5
```

## Gotchas

- **`val_len` should be a multiple of the batch size.** A ragged last batch
  changes the `BlockMask` shape and recompiles every epoch.
- **Generated coordinates are continuous, not on the 5 mm grid.** If you snap
  them, merge duplicates and sum their energies — about 5% of electron hits
  collide.
- **Generation costs more than training.** ~60 min per 5000 electron events at
  200 Heun steps on one A100.
- **`--renormalize` is opt-in but effectively required.** See results.md.

## torch 2.5 notes

The pinned environment (torch 2.5.1+cu121) has three flex-attention pitfalls
this repo works around; all disappear with torch >= 2.6:

- `torch.compile(create_block_mask)` fails for mask functions that capture a
  tensor, and the uncompiled version materializes a dense `(B, P, P)` mask
  (~19 GB at batch 128 x 4096). `compute_mask` builds the `BlockMask`
  analytically instead, verified bit-exact against the reference.
- The compiled flex-attention kernel requires `dim_embedding / num_head >= 16`.
- `torch.nn.utils.get_total_norm` does not exist yet (local fallback in
  `train.py`).
