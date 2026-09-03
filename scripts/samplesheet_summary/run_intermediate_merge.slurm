#!/usr/bin/bash
#SBATCH -J im_merge
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%A_%a.log
#SBATCH -e logs/%x_%A_%a.log

# Array: merge multi-batch deconvs for one sample, then thr=0 + after % for modes A/B.
# Usage:
#   sbatch --array=0-N%20 run_intermediate_merge.slurm MULTI_CSV CACHE_DIR

set -euo pipefail

MULTI_CSV=${1:?multi_merge_units.csv}
CACHE_DIR=${2:?intermediate_cache}
REPO=${REPO:-/lustre1/cqyi/AIPT_2.0/workflow/episcore}
SIF=${SIF:-${REPO}/containers/common_tools.sif}
CPG_DIR=${CPG_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260525-grid_search_240k_panel_240k_model/recall_list_220k}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REF_FREE="${REPO}/scripts/ref_free"
MERGE_BIN="${REPO}/bin/merge_deconv_res_full.py"

cd "$SCRIPT_DIR"
mkdir -p logs

INDEX=${SLURM_ARRAY_TASK_ID:?need --array}
PY=${PY:-$HOME/softwares/miniconda3/envs/custom_bert/bin/python}

# shellcheck disable=SC1091
eval "$("$PY" - <<PY
import pandas as pd
df = pd.read_csv("${MULTI_CSV}")
r = df.iloc[${INDEX}]
print(f"SAMPLE={r['sample']!r}")
print(f"DECONV_LIST={r['deconv_list']!r}")
PY
)"

MERGE_DIR="${CACHE_DIR}/merged_deconv"
mkdir -p "$MERGE_DIR"
MERGED="${MERGE_DIR}/${SAMPLE}.merged.deconv.parquet"

if [[ ! -f "$MERGED" ]]; then
    singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
        python3 "$MERGE_BIN" \
            --inputs "$DECONV_LIST" \
            --output "$MERGED"
fi

# one-row units for percentage scripts
TMP_UNITS=$(mktemp "${CACHE_DIR}/_merge_unit_${SAMPLE}_XXXX.csv")
UNIT_ID="${SAMPLE}__MERGED"
echo "unit_id,deconv_res" > "$TMP_UNITS"
echo "${UNIT_ID},${MERGED}" >> "$TMP_UNITS"

run_pct () {
    local thr=$1 recall=$2 outdir=$3
    mkdir -p "$outdir"
    local dest="${outdir}/${UNIT_ID}.percentage.tsv"
    if [[ -f "$dest" ]]; then
        echo "skip pct thr=${thr} ${UNIT_ID}"
        return 0
    fi
    singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
        python3 "${REF_FREE}/compute_unit_percentages.py" \
            --units "$TMP_UNITS" \
            --output-dir "$outdir" \
            --threshold "$thr" \
            --recall "$recall" \
            --cpg-recall-dir "$CPG_DIR" \
            --unit-id "$UNIT_ID"
}

# before thr=0 for both modes; after at mode pins
run_pct 0.0 0.95 "${CACHE_DIR}/merged_percentage_thr0_modeA"
run_pct 0.0 0.92 "${CACHE_DIR}/merged_percentage_thr0_modeB"
run_pct 0.85 0.95 "${CACHE_DIR}/merged_percentage_thr0.85_modeA"
run_pct 0.9 0.92 "${CACHE_DIR}/merged_percentage_thr0.9_modeB"

rm -f "$TMP_UNITS"
echo "OK merge+pct ${SAMPLE}"
