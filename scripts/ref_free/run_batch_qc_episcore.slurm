#!/usr/bin/bash
#SBATCH -J batch_qc_ep
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

# Array task: Mode-A episcore from production wide / beta@0.5.
# Usage:
#   sbatch --array=0-N run_batch_qc_episcore.slurm UNITS OUT_DIR THRESHOLD RECALL [CPG_LIST]

set -euo pipefail

UNITS=${1:?units csv}
OUT_DIR=${2:?output dir}
THRESHOLD=${3:?threshold}
RECALL=${4:?recall}
CPG_LIST=${5:-}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs "$OUT_DIR"

INDEX=${SLURM_ARRAY_TASK_ID:?need --array}

EXTRA=()
if [[ -n "$CPG_LIST" ]]; then
    EXTRA+=(--cpg-list "$CPG_LIST")
fi

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 compute_unit_episcore.py \
        --units "$UNITS" \
        --output-dir "$OUT_DIR" \
        --threshold "$THRESHOLD" \
        --recall "$RECALL" \
        --index "$INDEX" \
        "${EXTRA[@]}"
