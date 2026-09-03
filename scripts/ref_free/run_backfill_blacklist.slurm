#!/usr/bin/bash
#SBATCH -J bl_backfill
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

set -euo pipefail

INPUT_DIR=${1:?input dir}
SWEEP_BASE=${2:?sweep base}
POOL_SIZES=${POOL_SIZES:-$(python3 - <<'PY'
print(",".join(str(p) for p in range(20, 161, 2)))
PY
)}
TOTAL_REPEATS=${TOTAL_REPEATS:-20000}
SEED=${SEED:-42}
FILL_SEED=${FILL_SEED:-7}
BLACKLIST=${BLACKLIST:-PTAY0577P9S1,PTAY0599P8S1,PTAY0666P7S1,PTAY0682P7S1,PTAY0689P8H1}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-48}}
FORCE=${FORCE:-0}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

IFS=',' read -r -a SIZE_ARR <<< "$POOL_SIZES"
POOL_SIZE=${SIZE_ARR[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export N_JOBS

echo "[$(date)] backfill_blacklist pool=${POOL_SIZE} repeats=${TOTAL_REPEATS} n_jobs=${N_JOBS} force=${FORCE}"

PY_ARGS=(
    --input-dir "$INPUT_DIR"
    --sweep-base "$SWEEP_BASE"
    --pool-sizes "$POOL_SIZES"
    --pool-size "$POOL_SIZE"
    --total-repeats "$TOTAL_REPEATS"
    --seed "$SEED"
    --fill-seed "$FILL_SEED"
    --n-jobs "$N_JOBS"
    --blacklist "$BLACKLIST"
)
if [ "$FORCE" = "1" ]; then
    PY_ARGS+=(--force)
fi

singularity exec -B /lustre1:/lustre1 "$SIF" \
    python3 backfill_blacklist_pool_ratios.py "${PY_ARGS[@]}"

echo "[$(date)] done pool=${POOL_SIZE}"
