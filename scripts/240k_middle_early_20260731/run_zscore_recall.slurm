#!/usr/bin/bash
#SBATCH -J me31_z
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o logs/%x.log
#SBATCH -e logs/%x.log

# Conventional / middle-ref zscore for one recall.
# Args: <recall> [samples_meta] [out_dir]

set -euo pipefail
recall=${1:?Usage: sbatch run_zscore_recall.slurm <recall> [meta] [outdir]}
SAMPLES_META=${2:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260731-240k_middle_normal_samples_plus_early_allosomes_samples/zscore_samples_meta.csv}
OUT_BASE=${3:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260731-240k_middle_normal_samples_plus_early_allosomes_samples/zscore_recall_conventional}

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

# Skip only if valid chrX_percentage
if [ -f "${OUT_DIR}/_analyze_zscore.tsv.gz" ]; then
    if singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" python3 - "${OUT_DIR}/_analyze_zscore.tsv.gz" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], sep="\t")
raise SystemExit(0 if "chrX_percentage" in df.columns and float(df["chrX_percentage"].max()) > 0 else 1)
PY
    then
        echo "[$(date)] recall=${recall} already valid, skipping"
        exit 0
    fi
    rm -f "${OUT_DIR}/_analyze_zscore.tsv.gz" "${OUT_DIR}/_reference_percentage.tsv.gz"
fi

mkdir -p "$OUT_DIR"
echo "[$(date)] zscore recall=${recall} cutoff=0.85"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 run_zscore_recall.py \
        --samples-meta "$SAMPLES_META" \
        --cpg-list "$CPG_LIST" \
        --output-prefix "${OUT_DIR}/" \
        --cutoff 0.85 \
        --ncpus "${SLURM_CPUS_PER_TASK:-4}"

echo "[$(date)] Finished zscore recall=${recall}"
