#!/bin/bash
# Generate a 10k held-out sample from a trained run and evaluate it.
#
#   scripts/generate_and_eval_10k.sh <run_dir> <cache.h5> [n_events]
#
# Works either as a batch job
#   sbatch -A m2612_g -C gpu -q shared -N 1 -n 1 -c 32 --gpus-per-task=1 \
#          -t 06:00:00 scripts/generate_and_eval_10k.sh <run_dir> <cache.h5>
# or on an interactive allocation, which usually starts far sooner:
#   salloc -A m2612_g -C gpu -q interactive -N 1 -t 04:00:00 \
#          --gpus-per-node=4 -c 128 srun -n 1 \
#          scripts/generate_and_eval_10k.sh <run_dir> <cache.h5>
#
# This lives in the repo rather than a scratch directory on purpose: /tmp is
# node-local on Perlmutter, so a script under /tmp is not visible from the
# compute node and the job dies with "No such file or directory".
set -o pipefail
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python
cd /global/u1/o/ozamram/personal/LAR_Diffu/lardiff

RUN=${1:?usage: generate_and_eval_10k.sh <run_dir> <cache.h5> [n_events]}
CACHE=${2:?usage: generate_and_eval_10k.sh <run_dir> <cache.h5> [n_events]}
N=${3:-10000}
[ -d "$RUN" ] || { echo "no such run directory: $RUN"; exit 1; }
[ -f "$CACHE" ] || { echo "no such cache: $CACHE"; exit 1; }

TOTAL=$($PY -c "import yaml;print(yaml.safe_load(open('$RUN/conf.yaml'))['train']['num_epochs'])")
DONE=$(wc -l < "$RUN/data/losses.txt" 2>/dev/null || echo 0)
echo "run:    $RUN"
echo "epochs: $DONE of $TOTAL"
echo "val:    $(tail -1 "$RUN/data/losses.txt" 2>/dev/null | awk '{print $2}')"
[ "$DONE" -lt "$TOTAL" ] && echo "WARNING: training did not finish; sampling an unconverged model"

echo
echo "############ generate $N events"
# the generator refuses to draw from the training region, so this is held out
$PY -m lardiff.generator "$RUN" "$CACHE" -n "$N" \
    --n-source global --global-model results/global_all_species_v5 \
    --renormalize --solver heun --num-timesteps 200 --seed 0 \
    --out "$RUN/samples_10k.h5" || exit 1

echo
echo "############ 1D observables"
$PY -m lardiff.evaluate "$RUN/samples_10k.h5" "$CACHE" --out "$RUN/eval_10k" || exit 1

echo
echo "############ multivariate metrics"
$PY -m lardiff.metrics "$RUN/samples_10k.h5" "$CACHE" \
    --out "$RUN/metrics_10k.json" || exit 1

echo
echo "all done: $RUN/eval_10k, $RUN/metrics_10k.json"
