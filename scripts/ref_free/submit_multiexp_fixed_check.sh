#!/usr/bin/bash
# Quick fixed-combo (40+40) ref_free check from an mqres samplesheet.
# Skips FF / nextflow — reuses production episcore + zscore artifacts.
#
#   MQRES=... TODAY_BASE=... ./submit_multiexp_fixed_check.sh [-n|--dry-run]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

TODAY_BASE=${TODAY_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260803-ref_free_multiexp_sample_check}
MQRES=${MQRES:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260803-ref_free_multiexp_sample_check/mqres.csv}
MAIN_INPUT=${MAIN_INPUT:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
INPUT_DIR="${TODAY_BASE}/input_fixed"
FIXED_BASE="${TODAY_BASE}/fixed_combo"
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-2000}
MAX_ARRAY_JOBS=${MAX_ARRAY_JOBS:-50}
REF_N=${REF_N:-40}
SEED=${SEED:-42}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
DRY_RUN=${DRY_RUN:-0}
JOB_NAME=${JOB_NAME:-ref_free_chk}

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
if [ "$N_JOBS" -gt "$MAX_ARRAY_JOBS" ]; then
    REPEATS_PER_JOB=$(( (TOTAL_REPEATS + MAX_ARRAY_JOBS - 1) / MAX_ARRAY_JOBS ))
    N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
fi
ARRAY_LAST=$((N_JOBS - 1))

echo "Today base     : $TODAY_BASE"
echo "mqres          : $MQRES"
echo "Input (slim)   : $INPUT_DIR"
echo "Fixed out      : $FIXED_BASE"
echo "Array          : 0-${ARRAY_LAST} (${REPEATS_PER_JOB}/job) total=${TOTAL_REPEATS}"
echo "Ref split      : ${REF_N}+${REF_N}"

if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] prepare + submit fixed ${TOTAL_REPEATS}"
    exit 0
fi

mkdir -p "$FIXED_BASE/plots" logs
rm -rf "${FIXED_BASE}/ref_free_ezscore"
mkdir -p "${FIXED_BASE}/ref_free_ezscore"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 prepare_multiexp_fixed_assets.py \
        --main-input "$MAIN_INPUT" \
        --mqres "$MQRES" \
        --output-dir "$INPUT_DIR" \
        --ep-threshold 0.5 --ep-recall 0.65 \
        --z-threshold 0.85 --z-recall 0.95

export TOTAL_REPEATS REPEATS_PER_JOB REF_N SEED SIF COMPRESS=1
export COMBO_MODE=fixed STORE_PAIR_COUNTS=0
export EP_THRESHOLD=0.5 EP_RECALL=0.65
export Z_THRESHOLD=0.85 Z_RECALL=0.95
export CUTOFF=3.0 MIN_FF=0
# Grid 3.0..4.5; aggregate/plots use primary ez=4.5 for fixed combo
export EZ_CUTOFF_MIN=3.0 EZ_CUTOFF_MAX=4.5 EZ_CUTOFF_STEP=0.1
unset EP_THRESHOLD_MIN EP_THRESHOLD_MAX EP_RECALL_MIN EP_RECALL_MAX
unset Z_THRESHOLD_MIN Z_THRESHOLD_MAX Z_RECALL_MIN Z_RECALL_MAX

fixed_job=$(sbatch --parsable --job-name="${JOB_NAME}" \
    --array="0-${ARRAY_LAST}" \
    run_ref_free_ezscore.slurm "$INPUT_DIR" "$FIXED_BASE")
echo "Submitted fixed array job_id=${fixed_job}"

agg_job=$(sbatch --parsable --job-name="agg_${JOB_NAME}" \
    --dependency="afterok:${fixed_job}" \
    run_aggregate_and_plot.slurm \
    "$FIXED_BASE" \
    "40+40 fixed combo check (${TOTAL_REPEATS})" \
    "0.01" \
    0)
echo "Submitted aggregate+plot job_id=${agg_job}"
echo "Outputs: $TODAY_BASE"
