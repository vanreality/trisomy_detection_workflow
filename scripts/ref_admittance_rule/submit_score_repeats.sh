#!/usr/bin/bash
# Submit compact 40+40 rescoring (default 100k repeats, seed=42).
#
#   cd scripts/ref_admittance_rule
#   bash submit_score_repeats.sh
#
# Optional:
#   TOTAL_REPEATS=20000 TAG=smoke bash submit_score_repeats.sh
#   POOL_SAMPLES=/path/admitted.txt TAG=admitted SEED=7 TOTAL_REPEATS=20000 bash submit_score_repeats.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule}
TOTAL_REPEATS=${TOTAL_REPEATS:-100000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-20000}
MAX_ARRAY_JOBS=${MAX_ARRAY_JOBS:-50}
SEED=${SEED:-42}
TAG=${TAG:-baseline96}
EZ_CUTOFF=${EZ_CUTOFF:-4.5}
POOL_SAMPLES=${POOL_SAMPLES:-}
POOL_SOURCE=${POOL_SOURCE:-dev_normal}
EXCLUDE_EVAL_SAMPLES=${EXCLUDE_EVAL_SAMPLES:-}
EVAL_SAMPLES=${EVAL_SAMPLES:-}
EXTRA_INPUT_DIR=${EXTRA_INPUT_DIR:-}
DRY_RUN=${DRY_RUN:-0}

N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
if [ "$N_JOBS" -gt "$MAX_ARRAY_JOBS" ]; then
    REPEATS_PER_JOB=$(( (TOTAL_REPEATS + MAX_ARRAY_JOBS - 1) / MAX_ARRAY_JOBS ))
    N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
fi
ARRAY_LAST=$((N_JOBS - 1))

echo "INPUT_DIR      : $INPUT_DIR"
echo "OUT_BASE       : $OUT_BASE"
echo "tag            : $TAG"
echo "total repeats  : $TOTAL_REPEATS"
echo "array          : 0-${ARRAY_LAST} (${REPEATS_PER_JOB}/job)"
echo "seed           : $SEED"
echo "ez-cutoff      : $EZ_CUTOFF"
echo "pool-samples   : ${POOL_SAMPLES:-<all 96 dev Normal>}"
echo "pool-source    : $POOL_SOURCE"
echo "exclude-eval   : ${EXCLUDE_EVAL_SAMPLES:-<none>}"
echo "eval-samples   : ${EVAL_SAMPLES:-<default mask>}"
echo "extra-input    : ${EXTRA_INPUT_DIR:-<none>}"

if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] skip submit"
    exit 0
fi

export TOTAL_REPEATS REPEATS_PER_JOB SEED TAG POOL_SAMPLES POOL_SOURCE EXCLUDE_EVAL_SAMPLES EVAL_SAMPLES EXTRA_INPUT_DIR EZ_CUTOFF
job=$(sbatch --parsable --job-name="admit_${TAG}" \
    --array="0-${ARRAY_LAST}" \
    --export=ALL \
    run_score_repeats.slurm "$INPUT_DIR" "$OUT_BASE")
echo "Submitted job_id=${job}"
echo "Shards -> ${OUT_BASE}/${TAG}/repeats_*.npz"
