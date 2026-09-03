#!/usr/bin/bash
#SBATCH -J admit_mad16_pl
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 plot_mad_rank_dev_test.py \
        --dev-all "$DEV_ALL" \
        --dev-mad16 "$DEV_MAD16" \
        --dev-random16 "$DEV_RANDOM16" \
        --test-all "$TEST_ALL" \
        --test-mad16 "$TEST_MAD16" \
        --test-random16 "$TEST_RANDOM16" \
        --output-dir "$PLOT_OUT"
