#!/usr/bin/bash
#SBATCH -J setA_pool
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

# Array task: one pool size for one mode.
# Usage:
#   POOL_SIZES=... sbatch --array=0-N run_pool_size_setA.slurm modeA
#   sbatch --array=0-N run_pool_size_setA.slurm modeB

set -euo pipefail

MODE=${1:?modeA|modeB}
RESULT_ROOT=${RESULT_ROOT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_free_pool_plus_batch}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
FILL_SEED=${FILL_SEED:-7}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}

if [ -z "${POOL_SIZES:-}" ]; then
    POOL_SIZES=$(python3 - <<'PY'
print(",".join(str(p) for p in range(20, 161, 2)))
PY
)
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

IFS=',' read -r -a SIZE_ARR <<< "$POOL_SIZES"
POOL_SIZE=${SIZE_ARR[$SLURM_ARRAY_TASK_ID]}

INPUT_DIR="${RESULT_ROOT}/${MODE}/input"
OUTPUT_BASE="${RESULT_ROOT}/${MODE}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 N_JOBS

echo "[$(date)] SetA pool sweep mode=${MODE} pool=${POOL_SIZE} repeats=${TOTAL_REPEATS} n_jobs=${N_JOBS}"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 pool_size_setA_sweep.py \
        --mode "$MODE" \
        --input-dir "$INPUT_DIR" \
        --output-base "$OUTPUT_BASE" \
        --pool-sizes "$POOL_SIZES" \
        --pool-size "$POOL_SIZE" \
        --total-repeats "$TOTAL_REPEATS" \
        --seed "$SEED" \
        --fill-seed "$FILL_SEED" \
        --n-jobs "$N_JOBS"

echo "[$(date)] Finished pool=${POOL_SIZE}"
