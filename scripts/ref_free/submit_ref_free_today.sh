#!/usr/bin/bash
# Submit 40+40 ref_free @ 100k repeats (fixed + filtered), with val set,
# compressed slices, ez-pair subset search, dual plots, best-mode report.
#
# Layout:
#   ${TODAY_BASE}/input_with_val/          # main + val fixed-combo rows
#   ${TODAY_BASE}/fixed_combo/{ref_free_ezscore,plots}
#   ${TODAY_BASE}/filtered_combos/{ref_free_ezscore,plots}
#   ${TODAY_BASE}/best_mode_report.json

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

MAIN_INPUT=${MAIN_INPUT:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
VAL_META=${VAL_META:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260702-ref_40_20260625_samples}
TODAY_BASE=${TODAY_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260721-ref_free}
TOTAL_REPEATS=${TOTAL_REPEATS:-100000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-2000}
MAX_ARRAY_JOBS=${MAX_ARRAY_JOBS:-50}
REF_N=${REF_N:-40}
SEED=${SEED:-42}
FF_MIN=${FF_MIN:-0.01}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
DRY_RUN=${DRY_RUN:-0}
CLEAN_OLD=${CLEAN_OLD:-1}

N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
if [ "$N_JOBS" -gt "$MAX_ARRAY_JOBS" ]; then
    REPEATS_PER_JOB=$(( (TOTAL_REPEATS + MAX_ARRAY_JOBS - 1) / MAX_ARRAY_JOBS ))
    N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
fi
ARRAY_LAST=$((N_JOBS - 1))

FIXED_BASE="${TODAY_BASE}/fixed_combo"
FILTERED_BASE="${TODAY_BASE}/filtered_combos"
VAL_INPUT="${TODAY_BASE}/input_with_val"
mkdir -p "$FIXED_BASE" "$FILTERED_BASE" "$VAL_INPUT"

echo "Today base     : $TODAY_BASE"
echo "Ref split      : ${REF_N}+${REF_N} from 96-sample pool"
echo "Array          : 0-${ARRAY_LAST} (${REPEATS_PER_JOB} repeats/job) total=${TOTAL_REPEATS}"
echo "Plot ff filter : ${FF_MIN}"
echo "Val meta       : $VAL_META"

if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] would prepare val, clean old, submit fixed+filtered"
    exit 0
fi

# --- prepare val-augmented input (fixed combo rows for val) ---
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 prepare_val_fixed_assets.py \
        --main-input "$MAIN_INPUT" \
        --val-meta-dir "$VAL_META" \
        --output-dir "$VAL_INPUT" \
        --ep-threshold 0.5 --ep-recall 0.65 \
        --z-threshold 0.85 --z-recall 0.95 \
        --use-existing-wide

# Fixed uses val-augmented input; filtered uses main only (val lacks full grid)
FIXED_INPUT="$VAL_INPUT"
FILTERED_INPUT="$MAIN_INPUT"

if [ "$CLEAN_OLD" = 1 ]; then
    echo "Cleaning previous ref_free_ezscore / plots under fixed+filtered"
    rm -rf "${FIXED_BASE}/ref_free_ezscore" "${FIXED_BASE}/plots"
    rm -rf "${FILTERED_BASE}/ref_free_ezscore" "${FILTERED_BASE}/plots"
    mkdir -p "${FIXED_BASE}/plots" "${FILTERED_BASE}/plots"
fi

export TOTAL_REPEATS REPEATS_PER_JOB REF_N SEED SIF FF_MIN COMPRESS=1
# Grid 3.0..4.5; fixed combo primary ez=4.5, filtered primary ez=3.0
export EZ_CUTOFF_MIN=3.0 EZ_CUTOFF_MAX=4.5 EZ_CUTOFF_STEP=0.1
export CUTOFF=3.0 MIN_FF=0

# --- fixed combo (with val) ---
export COMBO_MODE=fixed STORE_PAIR_COUNTS=0
export EP_THRESHOLD=0.5 EP_RECALL=0.65
export Z_THRESHOLD=0.85 Z_RECALL=0.95
unset EP_THRESHOLD_MIN EP_THRESHOLD_MAX EP_RECALL_MIN EP_RECALL_MAX
unset Z_THRESHOLD_MIN Z_THRESHOLD_MAX Z_RECALL_MIN Z_RECALL_MAX

fixed_job=$(sbatch --parsable --job-name=ref_free_fixed \
    --array="0-${ARRAY_LAST}" \
    run_ref_free_ezscore.slurm "$FIXED_INPUT" "$FIXED_BASE")
echo "Submitted fixed array job_id=${fixed_job}"

fixed_plot=$(sbatch --parsable --job-name=plot_fixed \
    --dependency="afterok:${fixed_job}" \
    run_aggregate_and_plot.slurm \
    "$FIXED_BASE" \
    "40+40 fixed combo (ep 0.5/0.65, z 0.85/0.95)" \
    "$FF_MIN" \
    0)
echo "Submitted fixed aggregate+plot job_id=${fixed_plot}"

# --- filtered combos (main eval only; store pair counts for subset search) ---
export COMBO_MODE=all STORE_PAIR_COUNTS=1
unset EP_THRESHOLD EP_RECALL Z_THRESHOLD Z_RECALL
export EP_THRESHOLD_MIN=0.1 EP_THRESHOLD_MAX=0.5
export EP_RECALL_MIN=0.5 EP_RECALL_MAX=0.75
export Z_THRESHOLD_MIN=0.8 Z_THRESHOLD_MAX=0.95
export Z_RECALL_MIN=0.9 Z_RECALL_MAX=0.99

filtered_job=$(sbatch --parsable --job-name=ref_free_filt \
    --array="0-${ARRAY_LAST}" \
    run_ref_free_ezscore.slurm "$FILTERED_INPUT" "$FILTERED_BASE")
echo "Submitted filtered array job_id=${filtered_job}"

filtered_plot=$(sbatch --parsable --job-name=plot_filt \
    --dependency="afterok:${filtered_job}" \
    run_aggregate_and_plot.slurm \
    "$FILTERED_BASE" \
    "40+40 filtered combos" \
    "$FF_MIN" \
    1)
echo "Submitted filtered aggregate+plot+search job_id=${filtered_plot}"

# --- best-mode report after both plot jobs ---
report_job=$(sbatch --parsable --job-name=best_mode \
    --dependency="afterok:${fixed_plot}:${filtered_plot}" \
    --wrap="cd '$SCRIPT_DIR' && singularity exec -B /lustre1,/lustre2,/appsnew '$SIF' python3 report_best_mode.py --today-base '$TODAY_BASE'")
echo "Submitted best-mode report job_id=${report_job}"

echo "Outputs: $TODAY_BASE"
echo "Note: filtered val panel empty until full val grid parquets are built;"
echo "      fixed mode includes val (set=val) Normal/Trisomy samples."
