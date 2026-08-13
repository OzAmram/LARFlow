#!/bin/bash
# Single-GPU training job for the global p(N, R | E, type) model.
# Submit from the repo root:  sbatch scripts/train_global_perlmutter.sh <out_dir> [extra args]
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -J lardiff-global
#SBATCH -o slurm-%j.out

OUT=${1:?usage: train_global_perlmutter.sh <out_dir> [extra args]}
shift
GLOBALS=/global/cfs/cdirs/m2612/ozamram/LAR_Diffu/cache/lar_globals.h5
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python

exec $PY -u -m lardiff.global_model "$GLOBALS" --out "$OUT" "$@"
