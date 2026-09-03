#!/usr/bin/bash
#SBATCH -J jptay_epgrid
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%A_%a.log
#SBATCH -e logs/%x_%A_%a.log

# Array task: episcore recall grid for one query unit at one BAM-split threshold.
# Usage:
#   sbatch --array=0-N run_query_episcore_grid.slurm \
#       UNITS COMBOS THRESHOLD OUT_DIR NF_OUTDIR [CPG_DIR]

set -euo pipefail

UNITS=${1:?units csv}
COMBOS=${2:?epi combos csv}
THRESHOLD=${3:?threshold}
OUT_DIR=${4:?output dir}
NF_OUTDIR=${5:?nextflow extract_beta outdir}
CPG_DIR=${6:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260525-grid_search_240k_panel_240k_model/recall_list_220k}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs "$OUT_DIR"

INDEX=${SLURM_ARRAY_TASK_ID:?need --array}

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 compute_query_episcore_grid.py \
        --units "$UNITS" \
        --combos "$COMBOS" \
        --threshold "$THRESHOLD" \
        --output-dir "$OUT_DIR" \
        --cpg-recall-dir "$CPG_DIR" \
        --nf-outdir "$NF_OUTDIR" \
        --index "$INDEX"
