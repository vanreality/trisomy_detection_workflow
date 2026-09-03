#!/usr/bin/bash
#SBATCH -J kfold_ez_ref
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail

OUTPUT_DIR=${1:?output dir}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
POOL_N=${POOL_N:-220}
K_VALUES=${K_VALUES:-2-220}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[$(date)] kfold_ez_ref pool_n=${POOL_N} repeats=${TOTAL_REPEATS} k=${K_VALUES} n_jobs=${N_JOBS}"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 kfold_ez_ref.py run \
        --parquet "$PARQUET" \
        --meta "$META" \
        --toxic "$TOXIC" \
        --output-dir "$OUTPUT_DIR" \
        --pool-n "$POOL_N" \
        --total-repeats "$TOTAL_REPEATS" \
        --seed "$SEED" \
        --k-values "$K_VALUES" \
        --n-jobs "$N_JOBS" \
        --self-check

echo "[$(date)] finished scoring"
