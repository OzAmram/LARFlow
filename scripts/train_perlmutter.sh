#!/bin/bash
# Single-GPU training job for Perlmutter.
# Submit from the repo root:  sbatch scripts/train_perlmutter.sh [conf/lar_muon.yaml]
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -J lardiff-train
#SBATCH -o slurm-%j.out

CONF=${1:-conf/lar_muon.yaml}
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python

# resume automatically if a result_path is baked into the conf copy
exec $PY lardiff/train.py "$CONF"
