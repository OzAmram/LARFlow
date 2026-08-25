#!/bin/bash
# Generate the all-species evaluation sample as parallel chunks.
#
# The target is ~10k events per species.  `-n` is a slice of the cache, not a
# per-species count, and the cache is species-balanced at ~11.1%, so 90k events
# gives 9878-10169 of each.  At the measured 0.617 s/event that is 15.4 h in one
# go, and the generator holds the whole sample in memory and writes once at the
# end, so a timeout would lose all of it.  Nine chunks of 10k run in parallel in
# ~1.7 h each and a failed chunk costs only itself -- rerun that array index and
# merge again.
#
#   sbatch --dependency=afterany:<last training jobid> \
#          scripts/generate_v6_array_perlmutter.sh
#
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -J lardiff-v6-gen
#SBATCH -a 0-8
#SBATCH -o slurm-%A_%a.out

set -o pipefail
PY=/global/u1/o/ozamram/personal/envs/ml/bin/python
SHARED=/global/cfs/cdirs/m2612/ozamram/LAR_Diffu
CACHE=$SHARED/cache/lar_all_species_maxp8192.h5
CHUNK=10000
TOTAL=90000
cd /global/u1/o/ozamram/personal/LAR_Diffu/lardiff

RUN=$(ls -d results/*_LAr-allspecies-CNF-v6 2>/dev/null | sort | tail -1)
if [ -z "$RUN" ]; then
    echo "no all-species run directory found; did training ever start?"
    exit 1
fi
# the last TOTAL events of the cache, which sit inside the test holdout
NEVENTS=$($PY -c "import h5py;print(len(h5py.File('$CACHE')['energy_MeV']))")
START=$(( NEVENTS - TOTAL + SLURM_ARRAY_TASK_ID * CHUNK ))
OUT=$RUN/chunks/chunk_$(printf %02d $SLURM_ARRAY_TASK_ID).h5

echo "run:   $RUN"
echo "chunk: $SLURM_ARRAY_TASK_ID -> cache [$START, $((START + CHUNK))) -> $OUT"

exec $PY -m lardiff.generator "$RUN" "$CACHE" \
    -n $CHUNK --start $START --out "$OUT" \
    --n-source global --global-model results/global_all_species_v5 \
    --renormalize --solver heun --num-timesteps 200 --seed 0
