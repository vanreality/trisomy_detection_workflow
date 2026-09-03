#!/usr/bin/bash
# Harvest NF EXTRACT_BETA betas into BQC per-unit episcore TSVs.
#
# Usage:
#   ./harvest_intermediate_episcore.sh [--force]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/lustre1/cqyi/AIPT_2.0/workflow/episcore
OUTDIR=${OUTDIR:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary}
JOBS=${OUTDIR}/intermediate_cache/jobs
BQC=${BQC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
NF_BASE=${NF_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260814-intermediate_nf}
SIF=${SIF:-${REPO}/containers/common_tools.sif}
FORCE=()
if [[ "${1:-}" = "--force" ]]; then
    FORCE=(--force)
fi

harvest () {
    local units=$1
    local nf_out=$2
    local ep_dir=$3
    local thr=$4
    local recall=$5
    echo "=== harvest thr=$thr recall=$recall from $nf_out -> $ep_dir ==="
    mkdir -p "$ep_dir"
    singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
        python3 "${REPO}/scripts/ref_free/harvest_batch_qc_episcore.py" \
            --units "$units" \
            --nf-outdir "$nf_out" \
            --ep-dir "$ep_dir" \
            --threshold "$thr" \
            --recall "$recall" \
            "${FORCE[@]}"
}

harvest \
    "${JOBS}/missing_after_ep_nf_A.csv" \
    "${NF_BASE}/nf_extract_thr0.5" \
    "${BQC}/mode_A_ep0.5_0.65_z0.85_0.95/scores/episcore" \
    0.5 0.65

harvest \
    "${JOBS}/missing_after_ep_nf_B.csv" \
    "${NF_BASE}/nf_extract_thr0.1" \
    "${BQC}/mode_B_ep0.1_0.61_z0.9_0.92/scores/episcore" \
    0.1 0.61
