#!/usr/bin/bash
#SBATCH -J admit_score
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

set -euo pipefail

INPUT_DIR=${1:?input dir}
OUTPUT_DIR=${2:?output dir}
TOTAL_REPEATS=${TOTAL_REPEATS:-100000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-20000}
REF_N=${REF_N:-40}
SEED=${SEED:-42}
EZ_CUTOFF=${EZ_CUTOFF:-4.5}
TAG=${TAG:-baseline96}
POOL_SAMPLES=${POOL_SAMPLES:-}
POOL_SOURCE=${POOL_SOURCE:-dev_normal}
EXCLUDE_EVAL_SAMPLES=${EXCLUDE_EVAL_SAMPLES:-}
EVAL_SAMPLES=${EVAL_SAMPLES:-}
EXTRA_INPUT_DIR=${EXTRA_INPUT_DIR:-}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

REPEAT_START=$((SLURM_ARRAY_TASK_ID * REPEATS_PER_JOB))
REPEAT_END=$((REPEAT_START + REPEATS_PER_JOB))
if [ "$REPEAT_END" -gt "$TOTAL_REPEATS" ]; then
    REPEAT_END=$TOTAL_REPEATS
fi
if [ "$REPEAT_START" -ge "$TOTAL_REPEATS" ]; then
    echo "nothing to do"
    exit 0
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

EXTRA=()
if [ -n "$POOL_SAMPLES" ]; then
    EXTRA+=(--pool-samples "$POOL_SAMPLES")
fi
if [ -n "$POOL_SOURCE" ]; then
    EXTRA+=(--pool-source "$POOL_SOURCE")
fi
if [ -n "$EXCLUDE_EVAL_SAMPLES" ]; then
    EXTRA+=(--exclude-eval-samples "$EXCLUDE_EVAL_SAMPLES")
fi
if [ -n "$EVAL_SAMPLES" ]; then
    EXTRA+=(--eval-samples "$EVAL_SAMPLES")
fi
if [ -n "$EXTRA_INPUT_DIR" ]; then
    EXTRA+=(--extra-input-dir "$EXTRA_INPUT_DIR")
fi

echo "[$(date)] score_repeats array=${SLURM_ARRAY_TASK_ID} [$REPEAT_START, $REPEAT_END) tag=$TAG"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 score_repeats.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --total-repeats "$TOTAL_REPEATS" \
        --repeat-start "$REPEAT_START" \
        --repeat-end "$REPEAT_END" \
        --ref-n "$REF_N" \
        --seed "$SEED" \
        --ez-cutoff "$EZ_CUTOFF" \
        --tag "$TAG" \
        "${EXTRA[@]}"

echo "[$(date)] Finished array=${SLURM_ARRAY_TASK_ID}"
