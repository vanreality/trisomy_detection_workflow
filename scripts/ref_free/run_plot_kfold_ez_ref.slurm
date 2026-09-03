#!/usr/bin/bash
#SBATCH -J kfold_ez_plot
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
OUTPUT_DIR=${1:?output dir}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 kfold_ez_ref.py plot \
        --output-dir "$OUTPUT_DIR"
