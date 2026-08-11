#!/usr/bin/bash
#SBATCH -J me31_epi
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/%x.log
#SBATCH -e logs/%x.log

# Conventional episcore (early female early_ref) for one recall.
# Args: <recall> [samples_meta] [out_dir]

set -euo pipefail
recall=${1:?Usage: sbatch run_episcore_recall.slurm <recall> [meta] [outdir]}
SAMPLES_META=${2:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260731-240k_middle_normal_samples_plus_early_allosomes_samples/episcore_samples_meta.csv}
OUT_BASE=${3:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260731-240k_middle_normal_samples_plus_early_allosomes_samples/episcore_recall_conventional}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

CPG_DIR=/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260525-grid_search_240k_panel_240k_model/recall_list_240k
SIF=/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif
CPG_LIST="${CPG_DIR}/240k_cpg_recall_${recall}.txt"
OUT_DIR="${OUT_BASE}/recall_${recall}"

if [ ! -f "$SAMPLES_META" ] || [ ! -f "$CPG_LIST" ]; then
    echo "ERROR: missing meta or cpg list" >&2
    exit 1
fi
if [ -f "${OUT_DIR}/_analyze_zscore.tsv.gz" ]; then
    echo "[$(date)] recall=${recall} already done, skipping"
    exit 0
fi
mkdir -p "$OUT_DIR"
echo "[$(date)] episcore recall=${recall}"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 run_episcore_recall.py \
        --samples-meta "$SAMPLES_META" \
        --cpg-list "$CPG_LIST" \
        --output-prefix "${OUT_DIR}/" \
        --ncpus "${SLURM_CPUS_PER_TASK:-4}"

echo "[$(date)] Finished episcore recall=${recall}"
