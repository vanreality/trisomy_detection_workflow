#!/usr/bin/bash
#SBATCH -J me31_feat
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o logs/%x.log
#SBATCH -e logs/%x.log

set -euo pipefail
recall=${1:?Usage: sbatch run_features_recall.slurm <recall>}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT=/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260731-240k_middle_normal_samples_plus_early_allosomes_samples
OUTPUT=/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260731-240k_middle_normal_samples_plus_early_allosomes_samples
CPG_DIR=/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260525-grid_search_240k_panel_240k_model/recall_list_240k
SIF=/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif

META="${INPUT}/features_samples_meta.csv"
CPG_LIST="${CPG_DIR}/240k_cpg_recall_${recall}.txt"
OUT_DIR="${OUTPUT}/features_recall/recall_${recall}"

[ -f "$META" ] || { echo "missing $META"; exit 1; }
[ -f "$CPG_LIST" ] || { echo "missing $CPG_LIST"; exit 1; }
if [ -f "${OUT_DIR}/features.tsv.gz" ]; then
    echo "[$(date)] recall=${recall} features exist, skipping"
    exit 0
fi
mkdir -p "$OUT_DIR"
echo "[$(date)] features recall=${recall}"

# Ensure beta_to_zscore importable
export PYTHONPATH="${SCRIPT_DIR}/episcore:${SCRIPT_DIR}/../grid_search_240k:${PYTHONPATH:-}"

singularity exec -B /lustre1,/lustre2,/appsnew \
    --env PYTHONPATH="/lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/240k_middle_early_20260731/episcore:/lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/grid_search_240k" \
    "$SIF" python3 run_features_recall.py \
        --samples-meta "$META" \
        --cpg-list "$CPG_LIST" \
        --output-prefix "${OUT_DIR}/" \
        --cutoff 0.85 \
        --ncpus "${SLURM_CPUS_PER_TASK:-4}"

echo "[$(date)] Finished features recall=${recall}"
