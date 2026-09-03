#!/usr/bin/bash
# Recompute missing intermediate % / episcore when harvest is not enough.
#
#   ./submit_regenerate_missing_intermediate.sh [-n|--dry-run]
#
# Then:
#   python fill_missing_intermediate_modeA.py --cmd apply
#   python update_db_intermediate_chr.py

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/lustre1/cqyi/AIPT_2.0/workflow/episcore
OUTDIR=${OUTDIR:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary}
CACHE=${OUTDIR}/intermediate_cache
JOBS=${CACHE}/jobs
BQC=${BQC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
EP_DIR=${BQC}/mode_A_ep0.5_0.65_z0.85_0.95/scores/episcore
PCT_AFTER=${BQC}/mode_A_ep0.5_0.65_z0.85_0.95/scores/percentage
PCT_BEFORE=${CACHE}/percentage_thr0_modeA
CPG065=${REPO}/assets/CpG_recall0.65.txt
PY=${PY:-$HOME/softwares/miniconda3/envs/custom_bert/bin/python}
NF=${NF:-/appsnew/home/syfan/softwares/nextflow/nextflow}
NF_OUT=${NF_OUT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260828-placeholder_nf/nf_extract_thr0.5}
PROFILE=${PROFILE:-early,grid_search,alioth_slurm,singularity}
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done

cd "$SCRIPT_DIR"
mkdir -p logs "$PCT_BEFORE" "$EP_DIR" "$PCT_AFTER"
"$PY" fill_missing_intermediate_modeA.py --outdir "$OUTDIR" --cmd jobs

submit_array () {
    local name=$1 units=$2 out=$3 thr=$4 recall=$5 script=$6
    shift 6 || true
    local extra=("$@")
    [[ -f "$units" ]] || return 0
    local n; n=$(tail -n +2 "$units" | wc -l)
    [[ "$n" -gt 0 ]] || { echo "skip $name — 0 rows"; return 0; }
    local last=$((n - 1))
    echo "=== $name n=$n ==="
    if [[ "$DRY_RUN" = 1 ]]; then
        echo "[DRY-RUN] sbatch --array=0-${last} $script $units $out $thr $recall ${extra[*]-}"
        return 0
    fi
    sbatch --parsable --job-name="$name" --array="0-${last}%12" \
        "$script" "$units" "$out" "$thr" "$recall" ${extra[@]+"${extra[@]}"}
}

# missing_pct_compute.csv has mixed before/after; split if present
PCT="${JOBS}/missing_pct_compute.csv"
if [[ -f "$PCT" ]] && [[ $(tail -n +2 "$PCT" | wc -l) -gt 0 ]]; then
    "$PY" - <<PY
import pandas as pd
p="$PCT"
df=pd.read_csv(p)
if "which" in df.columns:
    df[df.which=="before"].to_csv("${JOBS}/missing_pct_before.csv", index=False)
    df[df.which=="after"].to_csv("${JOBS}/missing_pct_after.csv", index=False)
else:
    df.to_csv("${JOBS}/missing_pct_after.csv", index=False)
PY
    submit_array ph_miss_before "${JOBS}/missing_pct_before.csv" "$PCT_BEFORE" 0.0 0.95 run_intermediate_pct.slurm
    submit_array ph_miss_after "${JOBS}/missing_pct_after.csv" "$PCT_AFTER" 0.85 0.95 run_intermediate_pct.slurm
fi

submit_array ph_miss_ep "${JOBS}/missing_ep_from_beta.csv" "$EP_DIR" 0.5 0.65 run_intermediate_ep.slurm "$CPG065"

NF_SHEET="${JOBS}/missing_ep_nf_extract.csv"
if [[ -f "$NF_SHEET" ]] && [[ $(tail -n +2 "$NF_SHEET" | wc -l) -gt 0 ]]; then
    echo "=== NF EXTRACT_BETA n=$(tail -n +2 "$NF_SHEET" | wc -l) ==="
    if [[ "$DRY_RUN" = 1 ]]; then
        echo "[DRY-RUN] nextflow --threshold 0.5 --input $NF_SHEET"
    else
        mkdir -p "$NF_OUT/work"
        sbatch --parsable --job-name=ph_miss_nf --partition=cn-long --cpus-per-task=4 --mem=8G --time=72:00:00 \
            --output="${SCRIPT_DIR}/logs/ph_miss_nf_%j.log" \
            --wrap="cd '${NF_OUT}' && '${NF}' run '${REPO}/main.nf' -profile '${PROFILE}' -name 'ph_miss_nf' -w '${NF_OUT}/work' --step grid_search --input '${NF_SHEET}' --outdir '${NF_OUT}' --threshold 0.5"
    fi
fi
