#!/usr/bin/bash
#SBATCH -J jptay_zgrid
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.log
#SBATCH -e logs/%x_%A_%a.log

# Array task: zscore percentage grid for one query unit.
# Usage:
#   sbatch --array=0-N run_query_zscore_grid.slurm UNITS COMBOS OUT_DIR [CPG_DIR]

set -euo pipefail

UNITS=${1:?units csv}
COMBOS=${2:?z combos csv}
OUT_DIR=${3:?output dir}
CPG_DIR=${4:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260525-grid_search_240k_panel_240k_model/recall_list_220k}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs "$OUT_DIR"

INDEX=${SLURM_ARRAY_TASK_ID:?need --array}

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 compute_query_zscore_grid.py \
        --units "$UNITS" \
        --combos "$COMBOS" \
        --output-dir "$OUT_DIR" \
        --cpg-recall-dir "$CPG_DIR" \
        --index "$INDEX"
