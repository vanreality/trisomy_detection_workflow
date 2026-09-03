#!/usr/bin/bash
#SBATCH -J bqc_prep
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

# Wrapper: run TODAY_BASE/_run_after_scores.sh (written by submit_batch_qc_dual_mode.sh)

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
TODAY_BASE=${1:?TODAY_BASE}
bash "${TODAY_BASE}/_run_after_scores.sh"
