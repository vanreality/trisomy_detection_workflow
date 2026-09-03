#!/usr/bin/bash
#SBATCH -J im_pct
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.log
#SBATCH -e logs/%x_%A_%a.log

# Array: compute one unit percentage.
# Usage:
#   sbatch --array=0-N%40 run_intermediate_pct.slurm UNITS OUT_DIR THRESHOLD RECALL

set -euo pipefail

UNITS=${1:?units csv}
OUT_DIR=${2:?output dir}
THRESHOLD=${3:?threshold}
RECALL=${4:?recall}
REPO=${REPO:-/lustre1/cqyi/AIPT_2.0/workflow/episcore}
SIF=${SIF:-${REPO}/containers/common_tools.sif}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REF_FREE="${REPO}/scripts/ref_free"

cd "$SCRIPT_DIR"
mkdir -p logs "$OUT_DIR"

INDEX=${SLURM_ARRAY_TASK_ID:?need --array}

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 "${REF_FREE}/compute_unit_percentages.py" \
        --units "$UNITS" \
        --output-dir "$OUT_DIR" \
        --threshold "$THRESHOLD" \
        --recall "$RECALL" \
        --index "$INDEX"
