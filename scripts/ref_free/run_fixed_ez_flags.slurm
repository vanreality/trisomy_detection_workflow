#!/usr/bin/bash
#SBATCH -J fixed_ez_flags
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

set -euo pipefail

INPUT_DIR=${1:?input dir}
OUTPUT_BASE=${2:?output base}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-10000}
REF_N=${REF_N:-40}
SEED=${SEED:-42}
EP_THRESHOLD=${EP_THRESHOLD:-0.5}
EP_RECALL=${EP_RECALL:-0.65}
Z_THRESHOLD=${Z_THRESHOLD:-0.85}
Z_RECALL=${Z_RECALL:-0.95}
CUTOFF=${CUTOFF:-3.0}
EZ_CUTOFF=${EZ_CUTOFF:-4.5}
BLACKLIST=${BLACKLIST:-PTAY0577P9S1,PTAY0599P8S1,PTAY0666P7S1,PTAY0682P7S1,PTAY0689P8H1}
EZSCORE_REF_FILE=${EZSCORE_REF_FILE:-}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
REPEAT_START=$((TASK_ID * REPEATS_PER_JOB))
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

echo "[$(date)] fixed_ez_flags task=${TASK_ID} [$REPEAT_START, $REPEAT_END)"

PY_ARGS=(
    --input-dir "$INPUT_DIR"
    --output-base "$OUTPUT_BASE"
    --total-repeats "$TOTAL_REPEATS"
    --repeat-start "$REPEAT_START"
    --repeat-end "$REPEAT_END"
    --ref-n "$REF_N"
    --seed "$SEED"
    --cutoff "$CUTOFF"
    --ez-cutoff "$EZ_CUTOFF"
    --ep-threshold "$EP_THRESHOLD"
    --ep-recall "$EP_RECALL"
    --z-threshold "$Z_THRESHOLD"
    --z-recall "$Z_RECALL"
    --blacklist "$BLACKLIST"
)
if [ -n "$EZSCORE_REF_FILE" ]; then
    PY_ARGS+=(--ezscore-ref-file "$EZSCORE_REF_FILE")
fi

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 ref_free_fixed_ez_flags.py "${PY_ARGS[@]}"

echo "[$(date)] Finished task=${TASK_ID}"
