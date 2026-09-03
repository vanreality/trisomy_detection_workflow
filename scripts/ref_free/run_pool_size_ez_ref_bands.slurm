#!/usr/bin/bash
#SBATCH -J ez_ref_bands
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

set -euo pipefail

OUTPUT_DIR=${1:?output dir}
POOL_SIZE_LIST=${POOL_SIZE_LIST:-$(python3 - <<'PY'
print(",".join(str(p) for p in range(20, 221, 2)))
PY
)}
POOL_SIZES=${POOL_SIZES:-20,220,2}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

IFS=',' read -r -a SIZE_ARR <<< "$POOL_SIZE_LIST"
POOL_SIZE=${SIZE_ARR[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[$(date)] pool_size_ez_ref_bands pool=${POOL_SIZE} repeats=${TOTAL_REPEATS}"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 pool_size_ez_ref_bands.py \
        --parquet "$PARQUET" \
        --meta "$META" \
        --toxic "$TOXIC" \
        --output-dir "$OUTPUT_DIR" \
        --pool-sizes "$POOL_SIZES" \
        --pool-size "$POOL_SIZE" \
        --total-repeats "$TOTAL_REPEATS" \
        --seed "$SEED" \
        --n-jobs "$N_JOBS" \
        --arms "${ARMS:-all}"

echo "[$(date)] finished pool=${POOL_SIZE}"
