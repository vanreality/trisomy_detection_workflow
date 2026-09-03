#!/usr/bin/bash
#SBATCH -J jptay_t21search
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

# Reuse reference mu/sigma from previous grid-search parquets, search chr21
# epi/z combos, write per-batch ezscore tables.
#
# Usage:
#   sbatch run_search_t21_combo.slurm UNITS_CSV UNITS_DIR QUERY_EP_DIR QUERY_Z_DIR OUT_DIR

set -euo pipefail

UNITS=${1:?units csv}
UNITS_DIR=${2:?units dir}
QUERY_EP=${3:?query episcore dir}
QUERY_Z=${4:?query zscore dir}
OUT_DIR=${5:?output dir}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
GRID_INPUT=${GRID_INPUT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size/fixed_ez25/input_with_missing4}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs "$OUT_DIR"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 search_t21_combo.py \
        --units "$UNITS" \
        --units-dir "$UNITS_DIR" \
        --query-ep-dir "$QUERY_EP" \
        --query-z-dir "$QUERY_Z" \
        --output-dir "$OUT_DIR" \
        --grid-input "$GRID_INPUT"
