#!/bin/bash
# Generate from the all-species run and evaluate it, overall and per species.
#
# Chained onto the end of the training chain so it runs unattended:
#
#   sbatch --dependency=afterany:<last training jobid> \
#          scripts/postprocess_v6_perlmutter.sh
#
# afterany, not afterok: the last training job in a chain normally ends in
# TIMEOUT rather than COMPLETED, and that is not a failure.  If training did not
# reach the end this still produces output from whatever checkpoint exists,
# which beats nothing; the epoch count is printed so it is obvious what was used.
#
# No copying step: results/ is a symlink to the group-readable directory on
# CFS, so everything written here is already published.  Do not "helpfully" add
# an `rsync -a` back over it -- that copies the source's permissions and strips
# the setgid bits that keep new files in group m2612.
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH -J lardiff-v6-post
#SBATCH -o slurm-%j.out

set -o pipefail
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python
HOME_DIR=/global/u1/o/ozamram/personal/LAR_Diffu/lardiff
SHARED=/global/cfs/cdirs/m2612/ozamram/LAR_Diffu
CACHE=$SHARED/cache/lar_all_species_maxp8192.h5
GLOBAL=results/global_all_species_v5
N_SAMPLES=${1:-5000}
cd $HOME_DIR

RUN=$(ls -d results/*_LAr-allspecies-CNF-v6 2>/dev/null | sort | tail -1)
if [ -z "$RUN" ]; then
    echo "no all-species run directory found; did training ever start?"
    exit 1
fi
TOTAL=$($PY -c "import yaml;print(yaml.safe_load(open('$RUN/conf.yaml'))['train']['num_epochs'])")
DONE=$(wc -l < $RUN/data/losses.txt 2>/dev/null || echo 0)
echo "run:     $RUN"
echo "epochs:  $DONE of $TOTAL"
echo "val:     $(tail -1 $RUN/data/losses.txt 2>/dev/null | awk '{print $2}')"
[ "$DONE" -lt "$TOTAL" ] && echo "WARNING: training did not finish; sampling an unconverged model"

echo
echo "############ generate $N_SAMPLES events"
$PY -m lardiff.generator "$RUN" "$CACHE" \
    -n "$N_SAMPLES" --n-source global --global-model "$GLOBAL" \
    --renormalize --solver heun --num-timesteps 200 --seed 0 || exit 1

SAMPLES=$(ls -t $RUN/samples*.h5 | head -1)
echo "samples: $SAMPLES"

echo
echo "############ evaluate, all species together"
$PY -m lardiff.evaluate "$SAMPLES" "$CACHE" || exit 1

echo
echo "############ evaluate, per species"
for pdg in -211 -13 -11 11 13 22 211 2112 2212; do
    echo "--- pdg $pdg ---"
    $PY -m lardiff.evaluate "$SAMPLES" "$CACHE" --pdg $pdg \
        --out "$RUN/eval_pdg$pdg" || echo "  (failed for $pdg, continuing)"
done

# belt and braces: umask and the setgid bits should already have done this
chgrp -R m2612 "$RUN" 2>/dev/null
chmod -R g+rX "$RUN" 2>/dev/null
find "$RUN" -type d -exec chmod g+s {} + 2>/dev/null

echo
echo "############ published under $SHARED/results"
ls -d $RUN/eval_* 2>/dev/null
du -sh "$RUN"
echo "all done"
