#!/usr/bin/bash
#SBATCH -J pred_vs_loo
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
OUTPUT_DIR=${1:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev/0817/fixed_ref_17_plus_25}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}
LOO_DIR=${LOO_DIR:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev/0817/clean/mode4_loo_fixed}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs
export PYTHONUNBUFFERED=1

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 plot_pred_vs_loo_test.py \
        --parquet "$PARQUET" \
        --meta "$META" \
        --toxic "$TOXIC" \
        --loo-dir "$LOO_DIR" \
        --output-dir "$OUTPUT_DIR"
