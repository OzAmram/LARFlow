# lardiff

Flow-matching point-cloud generative model for single-particle liquid-argon
energy deposits. Core model and training code adapted from
[AllShowers](https://github.com/FLC-QU-hep/AllShowers) (arXiv:2601.11716);
`ode_solvers.py`, `preprocessing.py`, `util.py`, `data_loader.py` are copied
nearly verbatim, `transformer.py` / `flow_matching.py` / `train.py` are
simplified for the LAr case:

- points are 4D `[x, y, z, edep]` with all three coordinates generated
  (no calorimeter-layer structure),
- full attention over the point cloud (padding mask only),
- conditioning on `[log E_incident, log N_points]` as a global token,
- data read directly from a padded HDF5 cache (no `showerdata` dependency).

## Pipeline

```bash
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python

# 1. one-time preprocessing: raw voxel file -> per-particle padded cache
$PY scripts/preprocess_lar.py \
    --input /global/cfs/cdirs/m2612/ozamram/LAR_Diffu/lar_muon_voxels.h5 \
    --output /global/cfs/cdirs/m2612/ozamram/LAR_Diffu/cache/lar_pdg13_maxp4096.h5 \
    --pdg 13 --max-points 4096

# 2. training (single GPU; add --ddp under torchrun for multi-GPU)
$PY lardiff/train.py conf/lar_muon.yaml

# 3. generation (conditions on the validation tail of the cache)
$PY -m lardiff.generator results/<run> <cache.h5> -n 10000 --n-source truth

# 4. evaluation plots
$PY -m lardiff.evaluate results/<run>/samples00.h5 <cache.h5>
```
