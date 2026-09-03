#!/usr/bin/bash
# Fill placeholder rows in intermediate_each_batch_modeA.parquet.
#
# Rebuilds intermediate features (FF, %, episcore z_intra), not ezscore.
# Pins match the existing modeA score tree:
#   episcore 0.5/0.65   percentage after 0.85/0.95   percentage before 0.0/0.95
#
#   ./submit_fill_placeholder_modeA.sh [-n|--dry-run] [--skip-prepare] [--assemble-only]
#
# After arrays finish:
#   python fill_placeholder_intermediate_modeA.py --cmd assemble

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
CPG065=${CPG065:-${REPO}/assets/CpG_recall0.65.txt}
PY=${PY:-$HOME/softwares/miniconda3/envs/custom_bert/bin/python}
NF=${NF:-/appsnew/home/syfan/softwares/nextflow/nextflow}
NF_OUT=${NF_OUT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260828-placeholder_nf/nf_extract_thr0.5}
PROFILE=${PROFILE:-early,grid_search,alioth_slurm,singularity}
ARRAY_THROTTLE=${ARRAY_THROTTLE:-12}
DRY_RUN=0
SKIP_PREPARE=0
ASSEMBLE_ONLY=0

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        --skip-prepare) SKIP_PREPARE=1 ;;
        --assemble-only) ASSEMBLE_ONLY=1 ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

mkdir -p "${SCRIPT_DIR}/logs" "$JOBS" "$PCT_BEFORE" "$EP_DIR" "$PCT_AFTER"
cd "$SCRIPT_DIR"

if [[ "$ASSEMBLE_ONLY" = 1 ]]; then
    "$PY" fill_placeholder_intermediate_modeA.py --outdir "$OUTDIR" --cmd assemble
    exit 0
fi

if [[ "$SKIP_PREPARE" != 1 ]]; then
    echo "=== prepare placeholder manifests ==="
    "$PY" fill_placeholder_intermediate_modeA.py --outdir "$OUTDIR" --cmd prepare
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

# 1) after_mq episcore from production wide / beta@0.5 (modeA 0.5/0.65)
submit_array ph_epA \
    "${JOBS}/placeholder_ep.csv" \
    "$EP_DIR" \
    0.5 0.65 run_intermediate_ep.slurm \
    "$CPG065"

# 2) percentages
submit_array ph_beforeA \
    "${JOBS}/placeholder_pct.csv" \
    "$PCT_BEFORE" \
    0.0 0.95 run_intermediate_pct.slurm

submit_array ph_afterA \
    "${JOBS}/placeholder_pct.csv" \
    "$PCT_AFTER" \
    0.85 0.95 run_intermediate_pct.slurm

# 3) NF EXTRACT_BETA for units with no beta@0.5
NF_SHEET="${JOBS}/placeholder_nf_extract.csv"
if [[ -f "$NF_SHEET" ]] && [[ $(tail -n +2 "$NF_SHEET" | wc -l) -gt 0 ]]; then
    n=$(tail -n +2 "$NF_SHEET" | wc -l)
    echo "=== ph_nf_thr05 n=$n out=$NF_OUT ==="
    if [[ "$DRY_RUN" = 1 ]]; then
        echo "[DRY-RUN] nextflow grid_search --threshold 0.5 --input $NF_SHEET"
    else
        mkdir -p "$NF_OUT" "${NF_OUT}/work" "${NF_OUT}/../nf_logs"
        jid=$(sbatch --parsable --job-name=ph_nf_thr05 \
            --partition=cn-long --cpus-per-task=4 --mem=8G --time=72:00:00 \
            --output="${SCRIPT_DIR}/logs/ph_nf_thr05_%j.log" \
            --wrap="cd '${NF_OUT}' && \
                '${NF}' run '${REPO}/main.nf' \
                  -profile '${PROFILE}' \
                  -name 'ph_nf_thr05' \
                  -w '${NF_OUT}/work' \
                  --step grid_search \
                  --input '${NF_SHEET}' \
                  --outdir '${NF_OUT}' \
                  --threshold 0.5")
        echo "$jid" > "${JOBS}/ph_nf_thr05.jobid"
        echo "Submitted ph_nf_thr05 jobid=$jid"
    fi
fi

# Assemble after ep + percentage arrays (NF harvest is a second pass)
if [[ "$DRY_RUN" != 1 ]]; then
    deps=()
    for name in ph_epA ph_beforeA ph_afterA; do
        if [[ -f "${JOBS}/${name}.jobid" ]]; then
            deps+=("$(cat "${JOBS}/${name}.jobid")")
        fi
    done
    if [[ ${#deps[@]} -gt 0 ]]; then
        dep=$(IFS=:; echo "${deps[*]}")
        jid=$(sbatch --parsable --job-name=ph_assemble \
            --partition=cn-long --cpus-per-task=2 --mem=8G --time=1:00:00 \
            --dependency="afterok:${dep}" \
            --output="${SCRIPT_DIR}/logs/ph_assemble_%j.log" \
            --wrap="cd '${SCRIPT_DIR}' && '${PY}' fill_placeholder_intermediate_modeA.py --outdir '${OUTDIR}' --cmd assemble")
        echo "$jid" > "${JOBS}/ph_assemble.jobid"
        echo "Submitted ph_assemble jobid=$jid afterok:$dep"
    fi
fi

echo "Done submit. Jobids under ${JOBS}/ph_*.jobid"
echo "Monitor: squeue -u \$USER | grep -E 'ph_'"
echo "Status: $PY ${SCRIPT_DIR}/fill_placeholder_intermediate_modeA.py --outdir $OUTDIR --cmd status"
echo "After NF EXTRACT_BETA finishes, harvest then assemble:"
echo "  singularity exec -B /lustre1,/lustre2,/appsnew ${REPO}/containers/common_tools.sif \\"
echo "    python3 ${REPO}/scripts/ref_free/harvest_batch_qc_episcore.py \\"
echo "      --units ${JOBS}/placeholder_need_nf.csv --nf-outdir ${NF_OUT} \\"
echo "      --ep-dir ${EP_DIR} --threshold 0.5 --recall 0.65"
echo "  $PY ${SCRIPT_DIR}/fill_placeholder_intermediate_modeA.py --outdir $OUTDIR --cmd assemble"
