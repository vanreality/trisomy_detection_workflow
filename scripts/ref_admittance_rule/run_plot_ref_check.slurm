#!/usr/bin/bash
#SBATCH -J admit_chk_pl
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
CHECK_DIR=${1:?check dir}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
export PYTHONUNBUFFERED=1
PLOT_TAGS=${PLOT_TAGS:-all_96_test,toxic_16_excluded,random_16_excluded}
PLOT_OUT=${PLOT_OUT:-${CHECK_DIR}/plots}
EXTRA=()
if [ -n "${PLOT_TAGS:-}" ]; then
    EXTRA+=(--tags "$PLOT_TAGS")
fi
if [ -n "${PLOT_OUT:-}" ]; then
    EXTRA+=(--output-dir "$PLOT_OUT")
fi
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 plot_ref_admittance_check.py --check-dir "$CHECK_DIR" "${EXTRA[@]}"
