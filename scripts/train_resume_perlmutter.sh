#!/bin/bash
# Training job that continues an existing run if one is there, and otherwise
# starts a new one.  80 epochs takes ~36 h at 27 min/epoch, past the 24 h
# walltime cap, so submit this twice with --dependency=afterany between them:
# the first job starts the run, the second finds the result directory it made
# and resumes from checkpoints/last.pt.
#
#   sbatch scripts/train_resume_perlmutter.sh conf/lar_electron_v4_cont.yaml
#   sbatch --dependency=afterany:<jobid> scripts/train_resume_perlmutter.sh conf/lar_electron_v4_cont.yaml
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

CONF=${1:?usage: train_resume_perlmutter.sh <conf.yaml>}
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python
cd /global/u1/o/ozamram/personal/LAR_Diffu/lardiff

# a result dir carries the run_name in its name and a conf.yaml with
# result_path baked in, which is what makes Trainer pick up last.pt
RUN_NAME=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CONF")
LATEST=$(ls -d results/*_"$RUN_NAME" 2>/dev/null | sort | tail -1)

if [ -n "$LATEST" ] && [ -f "$LATEST/checkpoints/last.pt" ]; then
    echo "resuming $LATEST"
    CONF="$LATEST/conf.yaml"
else
    echo "starting a new run from $CONF"
fi

exec $PY lardiff/train.py "$CONF"
