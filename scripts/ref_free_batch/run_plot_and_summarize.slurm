#!/usr/bin/bash
#SBATCH -J setA_plot
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

# After the pool sweep: write 4 interactive HTMLs + README.md

set -euo pipefail

RESULT_ROOT=${RESULT_ROOT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_free_pool_plus_batch}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "[$(date)] plot + summarize -> $RESULT_ROOT"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 plot_pool_size_setA.py --result-root "$RESULT_ROOT"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 summarize_setA_pool.py --result-root "$RESULT_ROOT"

echo "[$(date)] Done"
ls -l "$RESULT_ROOT/plots" "$RESULT_ROOT/README.md"
