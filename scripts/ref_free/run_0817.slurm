#!/usr/bin/bash
#SBATCH -J 0817_ref
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

set -euo pipefail

OUTPUT_DIR=${1:?output dir}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
CUTOFF=${CUTOFF:-3.0}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}

ARMS=(raw raw raw raw clean clean clean clean ref60 ref60 ref60 ref60)
MODES=(1 2 3 4 1 2 3 4 1 2 3 4)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

# Arm-specific pool size (ref60→40, else 80). Ignore a mismatched POOL_SIZE env.
if [[ "$ARM" == "ref60" ]]; then
    POOL_SIZE=40
else
    POOL_SIZE=80
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[$(date)] 0817 arm=${ARM} mode=${MODE} pool_size=${POOL_SIZE} repeats=${TOTAL_REPEATS} n_jobs=${N_JOBS}"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 run_0817.py score \
        --parquet "$PARQUET" \
        --meta "$META" \
        --toxic "$TOXIC" \
        --output-dir "$OUTPUT_DIR" \
        --arm "$ARM" \
        --mode "$MODE" \
        --total-repeats "$TOTAL_REPEATS" \
        --cutoff "$CUTOFF" \
        --seed "$SEED" \
        --n-jobs "$N_JOBS" \
        --pool-size "$POOL_SIZE"

echo "[$(date)] finished arm=${ARM} mode=${MODE}"
