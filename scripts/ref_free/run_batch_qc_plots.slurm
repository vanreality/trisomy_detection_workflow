#!/usr/bin/bash
#SBATCH -J bqc_plots
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
#
# Refresh batch-QC scatters + Set D ezscore boxplots for one mode.
# Usage:
#   sbatch run_batch_qc_plots.slurm modeA
#   sbatch run_batch_qc_plots.slurm modeB

set -euo pipefail
MODE_NAME=${1:?modeA|modeB}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

ROOT=${RESULT_ROOT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
COHORT="$ROOT/cohort"
PLOTS="$ROOT/plots"

case "$MODE_NAME" in
    modeA)
        STATS="$PLOTS/modeA_per_chr_stats.tsv"
        INPUT="$ROOT/mode_A_ep0.5_0.65_z0.85_0.95/input_fixed"
        RESULT="$ROOT/mode_A_ep0.5_0.65_z0.85_0.95/fixed_combo"
        ;;
    modeB)
        STATS="$PLOTS/modeB_per_chr_stats.tsv"
        INPUT="$ROOT/mode_B_ep0.1_0.61_z0.9_0.92/input_fixed"
        RESULT="$ROOT/mode_B_ep0.1_0.61_z0.9_0.92/fixed_combo"
        ;;
    *)
        echo "MODE_NAME must be modeA or modeB" >&2
        exit 2
        ;;
esac

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 plot_batch_qc_sets.py \
        --cohort-dir "$COHORT" \
        --stats-tsv "$STATS" \
        --mode-name "$MODE_NAME" \
        --output-dir "$PLOTS" \
        --input-dir "$INPUT" \
        --result-dir "$RESULT"
