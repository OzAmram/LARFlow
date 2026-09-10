# Results

Validation of the models in [README.md](README.md), and what changed between
versions. Two-sample KS throughout; the 5% critical value is ~0.027 at 5000
events and ~0.019 at 10,000.

**The current results are the two 10k evaluations**: electrons from
`v4-cont` and muons from `muon-v4`, both generated through the same pipeline
with `N` and `R` from the global model and the deposits rescaled to the drawn
total. See *Current results* below; the version history after it records how
each defect was found and fixed.

## Versions

| | what it added | outcome |
|---|---|---|
| v1 | muons; the whole pipeline | works, but superseded |
| v2 | electrons, 8192-point cache | response spread 3x too wide |
| v3 | factorised global model `p(N, R \| E, type)` | fixes the response |
| v4 | `E_dep` conditioning channel; deterministic validation loss | fixes both defects of v3 |
| v4-cont | v4 warm-started, 80 more epochs | converged; the electron model |
| v5 | global model refit to match the point cache's truncation | improves `KS(R)` for every species |
| muon-v4 | muons retrained through the v4 pipeline, 8192 points | converged; the muon model |
| v6 | one point model for all 9 species, type-conditioned | **cancelled**, see below |

Every model here is per-species. v6 would have been the first joint one; it was
cancelled in favour of finishing the single-species results for a deadline, and
never trained successfully — see *v6: what happened*.

## Current results

Both species, 10,000 held-out events each, evaluated identically.

### One-dimensional (KS, 5% critical value 0.019)

| observable | electrons | muons |
|---|---|---|
| hit multiplicity `N` | 0.0053 | 0.0069 |
| total deposited energy | 0.0018 | 0.0040 |
| response `R` | 0.0057 | **0.0284** |
| width x / y / z | 0.013 / 0.013 / 0.013 | 0.017 / 0.015 / 0.008 |
| extent x / y / z | 0.008 / 0.010 / 0.014 | 0.017 / 0.013 / **0.034** |
| hit x / y / z | 0.035 / 0.036 / 0.008 | 0.035 / 0.047 / 0.004 |
| hit energy | **0.033** | 0.019 |

Response mean/std: electrons 0.9968 ± 0.0133 against Geant4 0.9969 ± 0.0134,
with the `R = 1` atom at 0.813 against 0.807. Muons 1.160 ± 0.479 against
1.164 ± 0.493 — muons deposit *more* than their incident kinetic energy because
decay products contribute, and the distribution is broad, so the KS of 0.028
corresponds to a much smaller shift in `R` than the same number would for
electrons.

Muon `extent_z` at 0.034 is the largest event-level miss: the mean is 158 mm
long on an 8.3 m span, and extent is a min–max statistic set by the single most
distant deposit.

### Multivariate

Over twelve event-level features (log `E_inc`, `R`, log `N`, per-axis centroid,
width and extent) — the same quantities plotted as 1D histograms.

| metric | electrons | muons |
|---|---|---|
| classifier AUC | **0.616 ± 0.005** | **0.512** |
| null (Geant4 vs Geant4) | 0.504 ± 0.004 | 0.502 |
| FPD ×10⁻³ | 0.289 ± 0.071 | 0.265 ± 0.087 |
| KPD ×10⁻³ | −0.005 ± 0.025 | 0.000 ± 0.033 |

**This is the most informative result in the project.** Muons are very nearly
indistinguishable — AUC 0.512 against a 0.502 floor. Electrons are clearly
separable at 0.616, *despite every marginal passing its KS test*. Training the
same classifier on each feature alone gives at most 0.518, and for most
features something indistinguishable from 0.5. So the electron discrepancy is
entirely in the **correlations between observables**, which is exactly what
one-dimensional tests cannot see.

The natural reading is topology. A muon is minimum-ionising and deposits along
a track, ~640 voxels whose joint structure is largely fixed by direction and
length. An electron cascade is ~2000 voxels with far richer correlations
between multiplicity, containment and shape. The per-deposit spectrum agrees:
KS 0.019 for muons against 0.033 for electrons.

Run it with:

```bash
python -m lardiff.metrics <run>/samples_10k.h5 <cache.h5> --out <run>/metrics_10k.json
```

FPD and KPD are transcribed from `jetnet.evaluation` rather than imported,
because `jetnet.evaluation` pulls energyflow, coffea and awkward at module
scope for metrics these two functions never touch. Checked against the
reference with those stubbed out: FPD agrees bit for bit, KPD to 3e-4 relative
(jetnet's is njit-compiled, so float accumulation order differs), which is
~700x smaller than KPD's own uncertainty.

**FPD sample-size caveat.** jetnet's defaults are `min_samples=20000,
max_samples=50000` and it recommends ≥50,000 events. The dense caches hold out
only `val_len` = 10,240 per species, so batches of 2,000–10,000 are used
instead. Absolute values are not comparable with published FPDs — only between
the models measured here the same way. The settings are recorded in the output
JSON.

## Point models

### v1, muons (100 epochs, ~1.3M parameters)

2000 held-out events, train 0.136 / val 0.133. Total deposited energy and
per-voxel spectrum medians agree to <1%; response 1.17 ± 0.5 in both;
longitudinal profile matches over ~8 decades. Empirical-`N` generation
reproduces the truth multiplicity (KS p = 0.90). Mean transverse extents run
~5% low — a tail effect, the medians agree.

### v2, electrons (40 epochs)

8192-point cap (5.8% of events truncated, ~0.05% of mean energy lost), 18 h on
one A100. The per-voxel spectrum agrees to ~1% at every percentile from p1
(0.53 keV) to p99.9 (9.3 MeV). Two defects:

- **Response spread ~3x too wide**, 0.043 against 0.014, across the whole energy
  range. The mean is right. Geant4 electrons below ~57 MeV are fully contained
  and deposit essentially all their energy; the model does not reproduce how
  deterministic that is. This is what v3 exists to fix.
- **Extents 3–5% high in the median** and further in the tail. The halo
  population itself is right — the fraction of hits beyond 0.5/1/2 m from the
  centroid matches Geant4 to about a percent — but extent is a min–max
  statistic, so a slight excess past 2 m is amplified. Those hits carry 0.16% of
  the event energy.

### v4 and v4-cont, electrons

v4 is v3 plus the two fixes described under *Things worth knowing*. v4-cont
warm-starts from it for 80 more epochs, because v4 was still improving when its
40-epoch cosine ran out.

| run | best val | best epoch | validation noise floor* |
|---|---|---|---|
| v2 | 0.27064 | 39 | 0.00143 |
| v3 | 0.27070 | 28 | 0.00143 |
| v4 | 0.27040 | 40 | 0.00013 |
| **v4-cont** | **0.270122** | **80** | **0.000009** |

\* std of the residual to the running minimum over the last 20 epochs.

Both v4 and v4-cont have their argmin on the last epoch, so `best.pt` and
`final.pt` are bit-identical — what a converged run under a cosine decaying to
zero should look like. The last two epochs of v4-cont moved the loss by 1e-8.
The warm restart is slow: at 30% of the original peak LR it took until **epoch
46** to get back below v4's 0.27040.

What the extra training bought, on 5000 held-out events with the same seed and
the same global-v5 `N` and `R` in both arms — so response and multiplicity are
identical by construction and every difference is the point model:

| observable | v4 | v4-cont |
|---|---|---|
| centroid x / y / z | 0.0224 / 0.0284 / 0.0168 | **0.0156** / **0.0176** / 0.0228 |
| extent x / y / z | 0.0202 / 0.0202 / 0.0292 | **0.0122** / **0.0168** / **0.0218** |
| rms x / y / z | 0.0212 / 0.0316 / 0.0150 | 0.0220 / **0.0198** / **0.0136** |
| hit x / y / z | 0.0383 / 0.0384 / 0.0040 | **0.0354** / **0.0368** / 0.0070 |
| hit edep | 0.0335 | **0.0328** |

Most shape observables improved; `centroid_z` and `hit_z` regressed. The
per-hit spectrum is within 0.5–2.6% of Geant4 at every percentile p1–p99.9 in
both.

### muon-v4, muons (100 epochs)

The v1 muon model predated the global model, the renormalisation step and the
`E_dep` conditioning channel, so its numbers were not comparable to the electron
ones sitting beside them. `conf/lar_muon_v4.yaml` retrains muons through the
current pipeline at 8192 points.

Converged cleanly: best val **0.065976** at epoch 99 of 100, with the last five
epoch-to-epoch deltas at ~1e-7 against a noise floor of 1.1e-6. 412 s/epoch,
~11.4 h of compute. `best.pt` and `final.pt` differ only because the argmin
landed on epoch 99; the gap is 1.1e-7, inside the noise.

Truncation barely touches muons: 0.01% of events exceed 8192 points against
5.8% for electrons.

### v6, all nine species: what happened

Cancelled, and worth recording because it cost several days of queue time
without producing a model.

The code works — `preprocess_lar_all.py`, the packed cache, `H5DataSet`,
`PackedLoader` and the type embedding are all committed and tested, and a
4-rank smoke test trains. What killed it was two multi-GPU failures in a row,
each of which a single-rank test passed straight through:

1. **`torchrun` has a stale shebang.** The console script's interpreter path is
   the one the environment was *built* with (`/workspace/envs/ml/bin/python3.11`),
   which does not exist here, so every job died in 5 seconds. Fixed by invoking
   `python -m torch.distributed.run` instead. Note `sbatch` snapshots the batch
   script at submission, so fixing the file did **not** reach jobs already
   queued — the whole chain had to be resubmitted.
2. **DDPOptimizer cannot compile `flex_attention`.** It raises
   `NotImplementedError: Found a higher order op in the graph` before the first
   step. Fixed with `torch._dynamo.config.optimize_ddp = False`, which makes the
   graph one bucket — negligible here at ~1.1M parameters.

Both only bite when `world_size > 1`, and `--fast-dev-run --ddp` with
`--nproc_per_node=1` never wraps the model in DDP at all. **Test multi-GPU
paths with at least 2 ranks on real devices**, which on this system means a
GPU-node job, since two ranks on one login-node GPU fail in NCCL.

The chain was then cancelled in favour of the single-species results. To
restart: `sbatch scripts/train_ddp_perlmutter.sh conf/lar_allspecies_v6.yaml`,
chained, ~3.5 days.

## The global model

v2's response spread was the clearest gap. Global quantities are hard for a
model that acts on points locally — nothing in the loss ties thousands of
independent hit energies together. v3 factorises:

```
p(X | E, type) = p(N, R | E, type) . p(X | N, E_dep, E)
```

The global model is ~205k parameters, 4–5 min on one A100 for 300 epochs over
1M events — about 0.4% of the point model's cost. It covers all nine species
with a particle embedding.

### Why a plain flow is not enough

83% of electrons are fully contained: they deposit exactly their incident
energy, so `R` is precisely 1.0 — a genuine point mass, not a narrow peak.
`p(R | E)` is therefore **mixed**, an atom plus a continuous escape tail, and no
continuous flow can represent an atom however long it trains. A single joint
flow over `(log N, log R)` smears it into a bump: 35% of sampled electrons came
out at `R > 1` (energy non-conservation) and only 38% near 1.0 against a true
83%.

### Finding the atom

Located in the data before any training — a labelling step, not part of the
model. A fully contained event deposits a fixed amount relative to its incident
energy, so `delta = edep - E` takes the same value for every contained event of
a species, independent of energy. Detection is then just *does one value of
`delta` get shared by many events?*: find the densest window in `delta`,
recentre on it, and declare an atom if it holds >2% of the species.

Two details matter. The search must run in `delta`, not `delta / E` — the e+
atom sits at a fixed *energy*, so in ratio space it smears across three decades.
And the winning candidate is one event's own `delta`, carrying ~1e-5 float64
summation noise, so the centre must be refined before the fraction is measured.
An earlier version used the median of `delta`, which only works when the atom
holds a majority; protons are 29% contained, so the median landed in the escape
tail and found nothing.

Given only `edep` and `E`, it recovers:

| species | offset | contained | interpretation |
|---|---|---|---|
| e- | +0.0000 MeV | 0.832 | deposits exactly its kinetic energy |
| gamma | +0.0000 MeV | 0.869 | same |
| e+ | +1.0220 MeV | 0.832 | 2 m_e c^2 — annihilation |
| p | +0.0000 MeV | 0.293 | stops below ~200 MeV |
| mu±, pi±, n | none | <0.02 | no atom — plain flow |

The positron offset landing on 2 m_e c^2 to four decimals is a check that this
finds real structure rather than fitting noise. Muons have no atom despite a
very sharp *continuous* peak (41% within 2% of `R = 1`) and are correctly left
to the plain flow.

### The mixture

```
p(N, R | E, type) = P(contained) . p(N | contained) . delta(R - ceiling)
                  + (1 - P)      . p(N, R | escaped)
```

with `ceiling = (E + offset) / E`. Three components: a BCE-trained containment
classifier, a 1-D flow over `log N` for contained events, and a 2-D joint flow
over `(log N, log R)` for escaped ones. If contained, `R` is set to the ceiling
deterministically but `N` is still sampled; if escaped, both come from the joint
flow with `R` clamped below the ceiling.

`N` needs its own flow because contained and escaped events have genuinely
different multiplicity distributions — contained events are the lower-energy
ones that stop in the volume. Sampling the joint flow for contained events does
not work either, because `R` is degenerate there, which is the pathology being
avoided. Species with no detected atom bypass the mixture.

### Results

50,000 held-out events, scored against the capped `R` the point cache actually
holds with `N` clamped at 8192 on both sides.

| species | v3 KS(N) | v3 KS(R) | v5 KS(N) | v5 KS(R) | v5 atom / Geant4 |
|---------|----------|----------|----------|----------|------------------|
| pi-     | 0.011 | 0.008 | 0.016 | 0.010 | — |
| mu+     | 0.007 | 0.013 | 0.010 | 0.007 | — |
| e+      | 0.008 | 0.026 | 0.007 | 0.023 | 0.802 / 0.818 |
| e-      | 0.005 | 0.026 | 0.007 | 0.018 | 0.813 / 0.819 |
| mu-     | 0.015 | 0.077 | 0.014 | 0.033 | — |
| gamma   | 0.006 | 0.032 | 0.006 | 0.022 | 0.831 / 0.846 |
| pi+     | 0.011 | 0.011 | 0.017 | 0.013 | — |
| n       | 0.017 | 0.029 | 0.016 | 0.011 | — |
| p       | 0.019 | 0.027 | 0.008 | 0.014 | 0.295 / 0.298 |

Against an untruncated reference the mixture took e- from KS(R) 0.487 to 0.010,
gamma from 0.521 to 0.010 and protons from 0.204 to 0.015, with no unphysical
`R > 1`. `N` was already good and is unchanged. Correlations `corr(log N, R)`
are reproduced to ~0.01 for the strongly correlated species (p -0.76, pi+ -0.65,
e+ -0.63), so the joint structure survives rather than just the marginals.

The weak entry, mu-, is a property of the statistic: quantiles agree to ~0.2%
throughout (median 1.0258 vs 1.0249), but the CDF is nearly vertical through the
sharp peak at `R = 1`, so a 0.002 shift registers as a large KS. Stable from 100
to 400 ODE steps, so it is peak placement, not integration error.

**Known defect.** The containment classifier misses badly at the 3728–6105 MeV
turn-on: Geant4 escapes 0.7865 of the time, the model 0.6947 — a 14 sigma gap,
where the escape rate climbs steepest and the training set thins to ~3900 events
per bin against 21000 lower down. Every other energy bin agrees to ~0.01.

### Matching the truncation (v5)

`R` must mean the same thing in both models. The point caches keep only the
`max_num_points` highest-energy hits, so their `R` is the energy in *those* hits
over `E_inc`. The globals cache originally stored the untruncated sum. The
dropped energy is negligible in the mean (~0.05% for electrons at 8192) but that
is the wrong statistic: ~6% of events exceed the cap, and dropping any hit from
a contained event moves it off `R = 1`, so the untruncated file puts 85% of
electrons on the atom where the capped cache has 82%.

`N` needs the opposite treatment, and capping it too is a mistake worth naming.
Truncation only ever *clamps* the count, so leaving `n_hits` untruncated and
clamping at generation reproduces the capped distribution exactly, point mass
included. Capping it in the training data makes the flow responsible for that
point mass, and a continuous flow over `log N` cannot express one — the same
pathology that forced the containment mixture, reintroduced in the other
coordinate. Measured: `KS(N)` rose from 0.005–0.019 to 0.021–0.033 and came back
down when `N` was left untruncated.

### Does renormalization help?

Six runs on the same 5000-event tail (heun, 200 steps).

| run | response | Geant4 | frac `R = 1` |
|-----|----------|--------|--------------|
| v2 truth-N | 0.9953 ± 0.0429 | 0.9964 ± 0.0143 | 0.001 / 0.792 |
| v2 truth-N +renorm | 0.9964 ± 0.0143 | 0.9964 ± 0.0143 | 0.792 / 0.792 |
| v3 truth-R | 1.0429 ± 0.0621 | 0.9966 ± 0.0147 | 0.000 / 0.798 |
| v3 truth-R +renorm | 0.9966 ± 0.0147 | 0.9966 ± 0.0147 | 0.798 / 0.798 |
| v3 global-R | 1.0415 ± 0.0607 | 0.9966 ± 0.0147 | 0.000 / 0.798 |
| v3 global-R +renorm | 0.9973 ± 0.0152 | 0.9966 ± 0.0147 | 0.836 / 0.798 |

Renormalization is not optional, for the same reason the global model needs the
mixture: ~79% of Geant4 electrons deposit their full energy, and a total that is
the sum of thousands of independently emitted hits lands on that atom
essentially never (0.001 and 0.000 above). Rescaling is the only mechanism in
the pipeline that can restore it.

It is also close to free — on v2 the per-hit spectrum is within 0.3% of Geant4
at every percentile p1–p99.99 either way, because the correction factors are
small and randomly signed. What it cannot fix is a *shape* error: rescaling
slides a tilt down rather than flattening it.

## Things worth knowing

### Conditioning on `R` did not work; on `E_dep` it does

Given the exact truth ratio, the v3 point model reproduced it no better than the
v2 model that never saw it (per-event `|R_gen - R_truth| / R_truth` median 4.6%
vs 2.0%). Part of that is the training signal — hits are emitted independently
with no feedback from a running total — but the encoding was also defective.
Every `cond` channel goes through the same elementwise `Log` then a per-channel
`StandardScaler`:

| channel | log-space std | z-range after scaling |
|---------|---------------|------------------------|
| `log E_inc` | 1.5872 | [-2.47, +1.88] |
| `log N` | 1.5966 | [-3.14, +1.38] |
| `log E_dep` (v4) | 1.5855 | [-2.66, +1.87] |
| `log R` (v3) | 0.0154 | **[-54.92, +0.21]** |

`log R` is dominated by the containment atom, so standardising sent the escape
tail to z = -55 and handed `cond_embedding` one channel more than an order of
magnitude larger than the other two. v4 carries `E_dep` instead. No information
is lost: `cond_embedding` is `Linear`, so it can form
`log R = log E_dep - log E_inc` itself.

### The validation loss was measuring its own noise

`CNF.loss` draws both `t ~ U(0,1)` and the noise fresh on every call, so the
validation loss was a new Monte Carlo estimate each epoch. The CFM loss varies
far more with `t` than between neighbouring epochs: on the electron runs that
left an epoch-to-epoch spread of **0.0015** around a real v2–v3 difference of
**0.00007**, so `best.pt` was effectively a random epoch off the plateau (38 for
v2, 27 for v3) and any comparison between checkpoints was measuring the draw.

Validation now uses fixed times — each event gets the midpoint of its stratum,
`t_i = (i + 0.5) / N_val`, under a fixed permutation so `t` does not correlate
with cache order — and noise from a `Generator` reseeded each `evaluate()`,
which reproduces it bit for bit at no storage cost (a stored tensor would be
1.3 GB). Repeated calls on unchanged weights now return bit-identical losses.
Training is untouched. Validation losses are not comparable across this change.

### The per-voxel spectrum has discrete lines

Geant4's per-voxel energy spectrum is **not continuous**. ~6.8% of deposits sit
on a handful of discrete energies set by the transport physics:

| line | Geant4 | v4 | v4-cont |
|---|---|---|---|
| 3.2063 keV | 6.30% | 1.89% | **3.04%** |
| 326.5 eV | 0.484% | 0.112% | **0.185%** |
| 29.24 eV | 0.055% | 0.002% | **0.008%** |
| 511 keV | 0.226% | 0.225% | 0.216% |

Away from the lines the CDF agrees within 0.2% at every decade boundary from
1e-7 to 10 MeV, so essentially the whole 0.033 hit-level KS sits at 3.2 keV.

Muons corroborate this: their per-deposit KS is 0.019 against the electrons'
0.033, and muon deposits are dominated by continuous ionisation rather than the
soft shower secondaries that populate the 3.2 keV line.

This is the same shape of problem as the containment atom in `R` and the
truncation atom in `N`, but unlike those it is **not** clearly a hard limit:
training alone moved the 3.2 keV fraction up 60%. A continuous density cannot
place a true atom, but it can concentrate arbitrarily sharply, and more capacity
or more epochs evidently keeps buying that. Worth checking before anyone builds
a mixture for it. The 511 keV annihilation line is already almost exact.

### Generated coordinates are continuous, not voxelized

True of every model here. The training data sits exactly on the 5 mm grid, but
the model emits continuous coordinates with no learned grid structure — the
offset from the nearest voxel centre is uniformly distributed (mean 1.250 mm,
against 1.25 for pure noise and 0.000 for Geant4). Snapping back onto the grid
therefore produces collisions: ~5% of electron hits and ~3% of muon hits land on
an already-occupied voxel, so a snap step must merge duplicates and sum their
energies rather than assume uniqueness.

### Reading the packed cache

A fully shuffled index over an out-of-core cache means every batch is a scatter
of single-event reads, and on CFS a scattered read costs **~52 ms per event**
against **~0.02 ms** when the same bytes arrive in one request — 1486 ms to
assemble a batch of 64, comparable to the training step it is meant to be
feeding. `PackedLoader` reads consecutive blocks in one call and shuffles within
the buffer, shuffling block order too: **85 ms per batch**, a 17x speedup.

That is only legitimate because the cache's physical order is already random,
which was checked rather than assumed: physical index correlates with
log-energy, species and log-multiplicity at r ~ 5e-4, and every 100k-event slab
is uniform in species and mean log-E.

The v6 split withholds `data.holdout_frac` (0.3) from training entirely —
700,000 train / 10,240 validation / 289,760 test, each species-balanced to
~11.1%. Validation monitors only the first `val_len` of the holdout, since
scoring all 300k every epoch would cost a large fraction of an epoch, so the
rest stays clean for evaluation.

## Next steps

Roughly in order of value.

- **Time the Geant4 application.** Generation costs ~0.6 s/event at 200 Heun
  steps on one A100, but there is no measured Geant4 time for the same events,
  so no speed-up factor can be quoted. A surrogate paper is expected to make
  that comparison, and it is the one number the writeup is still missing.
- **Close the electron correlation gap.** Classifier AUC 0.616 with every
  marginal passing is the clearest quantitative target: the model places each
  observable correctly and their joint structure imperfectly. Worth asking the
  classifier *which pairs* it uses.
- **Make `--renormalize` the default.** There is no case where you want it off.
- Revisit the discrete voxel-energy lines now that training is known to move
  them; decide whether a mixture is needed or capacity is enough.
- Restart v6 with the two DDP fixes in place, and compare per-species against
  these single-species results.
- Fix the global model's containment classifier at the 3728-6105 MeV turn-on
  (14 sigma, the worst bin by a wide margin).
- Fewer ODE steps at generation, or distillation. Generation dominates the cost
  of using this: ~1.7 h per 10k events against ~11 h to train the muon model.
- A larger holdout. `val_len` = 10,240 caps the evaluation sample, forces FPD
  off its standard batch sizes, and means the evaluation set is also the model
  selection set. `holdout_frac` already exists for the packed caches.
- Optional grid snapping with duplicate merging in `generator.py`.
- Condition on initial particle direction.
