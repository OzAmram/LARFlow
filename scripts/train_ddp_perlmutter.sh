#!/bin/bash
# Four-GPU training job that continues an existing run if one is there.
# 100 epochs of the all-species cache runs past the 24 h cap, so submit this
# repeatedly with --dependency=afterany between them:
#
#   sbatch scripts/train_ddp_perlmutter.sh conf/lar_allspecies_v6.yaml
#   sbatch --dependency=afterany:<jobid> scripts/train_ddp_perlmutter.sh conf/lar_allspecies_v6.yaml
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
#SBATCH -J lardiff-ddp
#SBATCH -o slurm-%j.out

CONF=${1:?usage: train_ddp_perlmutter.sh <conf.yaml>}
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

exec $(dirname $PY)/torchrun --standalone --nproc_per_node=4 \
    lardiff/train.py "$CONF" --ddp
