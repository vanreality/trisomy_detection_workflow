#!/usr/bin/bash
#SBATCH -J admit_plot_redraw
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
ANALYSIS_DIR=${1:?analysis dir}
BASELINE=${2:?baseline96 score dir}
ADMITTED=${3:?admitted score dir}
RANDOM_N=${4:?random_n score dir}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 plot_admittance.py \
        --analysis-dir "$ANALYSIS_DIR" \
        --baseline-score-dir "$BASELINE" \
        --admitted-score-dir "$ADMITTED" \
        --random-score-dir "$RANDOM_N"
