#!/usr/bin/bash
# Prepare manifests and submit SLURM arrays to fill intermediate matrix gaps.
#
# Stages:
#   1) prepare_intermediate_jobs.py (units + manifests + contamination audit)
#   2) before_mq % thr=0 (modeA recall 0.95, modeB 0.92) — all missing units
#   3) after_mq % for missing units (modeA 0.85/0.95, modeB 0.9/0.92)
#   4) modeA episcore from production wide/beta
#   5) multi-batch merge + merged percentages
#   6) optionally NF EXTRACT_BETA for remaining episcore (see --nf)
#
# Usage:
#   ./submit_intermediate_fill.sh [-n|--dry-run] [--nf] [--skip-prepare]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/lustre1/cqyi/AIPT_2.0/workflow/episcore
OUTDIR=${OUTDIR:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary}
CACHE=${OUTDIR}/intermediate_cache
JOBS=${CACHE}/jobs
BQC=${BQC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
CPG065=${CPG065:-${REPO}/assets/CpG_recall0.65.txt}
PY=${PY:-$HOME/softwares/miniconda3/envs/custom_bert/bin/python}
DRY_RUN=0
DO_NF=0
SKIP_PREPARE=0
ARRAY_THROTTLE=${ARRAY_THROTTLE:-40}

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        --nf) DO_NF=1 ;;
        --skip-prepare) SKIP_PREPARE=1 ;;
        -h|--help)
            sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

mkdir -p "${SCRIPT_DIR}/logs" "$JOBS"
cd "$SCRIPT_DIR"

if [[ "$SKIP_PREPARE" != 1 ]]; then
    echo "=== prepare manifests ==="
    "$PY" prepare_intermediate_jobs.py --outdir "$OUTDIR"
fi

submit_array () {
    local name=$1
    local units=$2
    local out=$3
    local thr=$4
    local recall=$5
    local script=$6
    shift 6 || true
    local extra=("$@")
    if [[ ! -f "$units" ]]; then
        echo "skip $name — missing $units"
        return 0
    fi
    local n
    n=$(tail -n +2 "$units" | wc -l)
    if [[ "$n" -le 0 ]]; then
        echo "skip $name — 0 rows"
        return 0
    fi
    local last=$((n - 1))
    echo "=== $name n=$n array=0-${last}%${ARRAY_THROTTLE} thr=$thr recall=$recall -> $out ==="
    if [[ "$DRY_RUN" = 1 ]]; then
        echo "[DRY-RUN] sbatch --array=0-${last}%${ARRAY_THROTTLE} $script $units $out $thr $recall ${extra[*]-}"
        return 0
    fi
    mkdir -p "$out"
    local jid
    jid=$(sbatch --parsable --job-name="$name" \
        --array="0-${last}%${ARRAY_THROTTLE}" \
        "$script" "$units" "$out" "$thr" "$recall" ${extra[@]+"${extra[@]}"})
    echo "$jid" > "${JOBS}/${name}.jobid"
    echo "Submitted $name jobid=$jid"
}

# 2) before_mq percentages
submit_array im_before_A \
    "${JOBS}/missing_before_pct_modeA.csv" \
    "${CACHE}/percentage_thr0_modeA" \
    0.0 0.95 run_intermediate_pct.slurm

submit_array im_before_B \
    "${JOBS}/missing_before_pct_modeB.csv" \
    "${CACHE}/percentage_thr0_modeB" \
    0.0 0.92 run_intermediate_pct.slurm

# 3) after_mq percentages → BQC score dirs
submit_array im_after_A \
    "${JOBS}/missing_after_pct_modeA.csv" \
    "${BQC}/mode_A_ep0.5_0.65_z0.85_0.95/scores/percentage" \
    0.85 0.95 run_intermediate_pct.slurm

submit_array im_after_B \
    "${JOBS}/missing_after_pct_modeB.csv" \
    "${BQC}/mode_B_ep0.1_0.61_z0.9_0.92/scores/percentage" \
    0.9 0.92 run_intermediate_pct.slurm

# 4) modeA episcore from production
submit_array im_epA_prod \
    "${JOBS}/missing_after_ep_A_from_prod.csv" \
    "${BQC}/mode_A_ep0.5_0.65_z0.85_0.95/scores/episcore" \
    0.5 0.65 run_intermediate_ep.slurm \
    "$CPG065"

# 5) multi-batch merge
MULTI="${JOBS}/multi_merge_units.csv"
if [[ -f "$MULTI" ]]; then
    n=$(tail -n +2 "$MULTI" | wc -l)
    if [[ "$n" -gt 0 ]]; then
        last=$((n - 1))
        echo "=== im_merge n=$n array=0-${last}%20 ==="
        if [[ "$DRY_RUN" = 1 ]]; then
            echo "[DRY-RUN] sbatch --array=0-${last}%20 run_intermediate_merge.slurm $MULTI $CACHE"
        else
            jid=$(sbatch --parsable --job-name=im_merge \
                --array="0-${last}%20" \
                run_intermediate_merge.slurm "$MULTI" "$CACHE")
            echo "$jid" > "${JOBS}/im_merge.jobid"
            echo "Submitted im_merge jobid=$jid"
        fi
    fi
fi

# 6) NF extract
if [[ "$DO_NF" = 1 ]]; then
    if [[ "$DRY_RUN" = 1 ]]; then
        ./submit_intermediate_nf_extract.sh --dry-run both
    else
        ./submit_intermediate_nf_extract.sh both
    fi
fi

echo "Done submit. Jobids under ${JOBS}/*.jobid"
echo "Monitor: squeue -u \$USER | grep -E 'im_|nf_'"
echo "After arrays + NF harvest finish:"
echo "  $PY ${SCRIPT_DIR}/build_intermediate_matrices.py --skip-compute --compute-merged"
