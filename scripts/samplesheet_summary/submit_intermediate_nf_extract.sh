#!/usr/bin/bash
# Submit Nextflow grid_search EXTRACT_BETA for missing intermediate episcore units.
#
# Usage:
#   ./submit_intermediate_nf_extract.sh [--mode A|B|both] [-n]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/lustre1/cqyi/AIPT_2.0/workflow/episcore
OUTDIR=${OUTDIR:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary}
JOBS=${JOBS:-${OUTDIR}/intermediate_cache/jobs}
NF_BASE=${NF_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260814-intermediate_nf}
MODE=${MODE:-both}
DRY_RUN=${DRY_RUN:-0}
PROFILE=${PROFILE:-early,grid_search,alioth_slurm,singularity}
NF=${NF:-/appsnew/home/syfan/softwares/nextflow/nextflow}

for arg in "$@"; do
    case "$arg" in
        A|B|both) MODE=$arg ;;
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

mkdir -p "${SCRIPT_DIR}/logs" "${NF_BASE}/nf_logs"

run_one () {
    local thr=$1
    local sheet=$2
    local name=$3
    local out="${NF_BASE}/nf_extract_thr${thr}"
    if [[ ! -f "$sheet" ]]; then
        echo "Missing samplesheet: $sheet" >&2
        return 1
    fi
    local n
    n=$(tail -n +2 "$sheet" | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo "Skip $name — empty samplesheet"
        return 0
    fi
    echo "=== $name thr=$thr n=$n out=$out ==="
    if [[ "$DRY_RUN" = 1 ]]; then
        echo "[DRY-RUN] sbatch nextflow --threshold $thr --input $sheet"
        return 0
    fi
    mkdir -p "$out"
    local jid
    jid=$(sbatch --parsable --job-name="${name}" \
        --partition=cn-long --cpus-per-task=4 --mem=8G --time=72:00:00 \
        --output="${SCRIPT_DIR}/logs/${name}_%j.log" \
        --wrap="mkdir -p '${out}/work' && \
            cd '${out}' && \
            '${NF}' run '${REPO}/main.nf' \
              -profile '${PROFILE}' \
              -name '${name}' \
              -w '${out}/work' \
              --step grid_search \
              --input '${sheet}' \
              --outdir '${out}' \
              --threshold ${thr}")
    echo "$jid" > "${NF_BASE}/nf_logs/${name}.jobid"
    echo "Submitted launcher job_id=${jid}"
}

case "$MODE" in
    A) run_one 0.5 "${JOBS}/nf_extract_missing_A.csv" im_nf_thr05 ;;
    B) run_one 0.1 "${JOBS}/nf_extract_missing_B.csv" im_nf_thr01 ;;
    both)
        run_one 0.5 "${JOBS}/nf_extract_missing_A.csv" im_nf_thr05
        run_one 0.1 "${JOBS}/nf_extract_missing_B.csv" im_nf_thr01
        ;;
    *)
        echo "MODE must be A|B|both" >&2
        exit 1
        ;;
esac
