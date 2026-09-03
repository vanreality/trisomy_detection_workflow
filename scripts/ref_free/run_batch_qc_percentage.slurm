#!/usr/bin/bash
#SBATCH -J batch_qc_pct
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

# Array task: compute percentage for one unit (INDEX = SLURM_ARRAY_TASK_ID).
# Usage:
#   sbatch --array=0-N run_batch_qc_percentage.slurm UNITS OUT_DIR THRESHOLD RECALL

set -euo pipefail

UNITS=${1:?units csv}
OUT_DIR=${2:?output dir}
THRESHOLD=${3:?threshold}
RECALL=${4:?recall}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs "$OUT_DIR"

INDEX=${SLURM_ARRAY_TASK_ID:?need --array}

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 compute_unit_percentages.py \
        --units "$UNITS" \
        --output-dir "$OUT_DIR" \
        --threshold "$THRESHOLD" \
        --recall "$RECALL" \
        --index "$INDEX"
