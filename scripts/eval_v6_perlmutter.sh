#!/bin/bash
# Merge the generated chunks and evaluate the all-species run, overall and for
# each of the nine species.
#
#   sbatch --dependency=afterany:<generate array jobid> \
#          scripts/eval_v6_perlmutter.sh
#
# afterany so a single failed chunk does not block the rest: the merge reports
# what is missing, and evaluating 8 species out of 9 beats evaluating none.
#
# Output goes to <run>/eval_* under results/, which is a symlink to the
# group-readable directory on CFS, so it is published as it is written.  Do not
# add an `rsync -a` back over that path -- it strips the setgid bits.
#
# Memory, not GPU, is what this needs: evaluate loads generated and truth points
# for the whole sample, ~12 GB each at 90k events by 8192 points.  The shared
# queue gives ~1.8 GB per core, so 32 cores is ~57 GB, comfortably above the
# ~30 GB peak.  Each species runs as its own process, so nothing accumulates.
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 06:00:00
#SBATCH -J lardiff-v6-eval
#SBATCH -o slurm-%j.out

set -o pipefail
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python
SHARED=/global/cfs/cdirs/m2612/ozamram/LAR_Diffu
CACHE=$SHARED/cache/lar_all_species_maxp8192.h5
cd /global/u1/o/ozamram/personal/LAR_Diffu/lardiff

RUN=$(ls -d results/*_LAr-allspecies-CNF-v6 2>/dev/null | sort | tail -1)
[ -z "$RUN" ] && { echo "no all-species run directory found"; exit 1; }
TOTAL=$($PY -c "import yaml;print(yaml.safe_load(open('$RUN/conf.yaml'))['train']['num_epochs'])")
DONE=$(wc -l < $RUN/data/losses.txt 2>/dev/null || echo 0)
echo "run:    $RUN"
echo "epochs: $DONE of $TOTAL"
echo "val:    $(tail -1 $RUN/data/losses.txt 2>/dev/null | awk '{print $2}')"
[ "$DONE" -lt "$TOTAL" ] && echo "WARNING: training did not finish; this samples an unconverged model"

echo
echo "############ merge chunks"
SAMPLES=$RUN/samples_all.h5
$PY scripts/merge_samples.py "$SAMPLES" $RUN/chunks/chunk_*.h5 || exit 1

echo
echo "############ evaluate, all species together"
$PY -m lardiff.evaluate "$SAMPLES" "$CACHE" --out "$RUN/eval_all" || exit 1

echo
echo "############ evaluate, per species"
for pdg in -211 -13 -11 11 13 22 211 2112 2212; do
    echo "--- pdg $pdg ---"
    $PY -m lardiff.evaluate "$SAMPLES" "$CACHE" --pdg $pdg \
        --out "$RUN/eval_pdg$pdg" || echo "  (failed for $pdg, continuing)"
done

# belt and braces: setgid and the 0007 umask should already have done this
chgrp -R m2612 "$RUN" 2>/dev/null
chmod -R g+rX "$RUN" 2>/dev/null
find "$RUN" -type d -exec chmod g+s {} + 2>/dev/null

echo
echo "############ published under $SHARED/results"
ls -d $RUN/eval_* 2>/dev/null
du -sh "$RUN"
echo "all done"
