#!/usr/bin/bash
# Fixed-combo pool-size sweep (step=2) → MCC plot on core eval + FP+FN @ ez=4.5.
#
# Layout:
#   ${OUT_BASE}/fixed/pool_{P}/...
#   ${OUT_BASE}/plots/pool_size_mcc.html
#   ${OUT_BASE}/fixed_flags_ez45/flags_*.npz
#   ${OUT_BASE}/fp_fn_density/fp_fn_density.html

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size}
POOL_STEP=${POOL_STEP:-2}
POOL_MIN=${POOL_MIN:-20}
POOL_MAX=${POOL_MAX:-160}
if [ -z "${POOL_SIZES:-}" ] || [ "${USE_POOL_STEP:-1}" = 1 ]; then
    POOL_SIZES=$(POOL_MIN=$POOL_MIN POOL_MAX=$POOL_MAX POOL_STEP=$POOL_STEP python3 - <<'PY'
import os
lo, hi, step = int(os.environ["POOL_MIN"]), int(os.environ["POOL_MAX"]), int(os.environ["POOL_STEP"])
print(",".join(str(p) for p in range(lo, hi + 1, step)))
PY
)
fi
FIXED_REPEATS=${FIXED_REPEATS:-10000}
FLAG_REPEATS=${FLAG_REPEATS:-1000000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-20000}
MAX_ARRAY_JOBS=${MAX_ARRAY_JOBS:-50}
SEED=${SEED:-42}
FILL_SEED=${FILL_SEED:-7}
FF_MIN=${FF_MIN:-0.01}
RATIO_CUTOFF=${RATIO_CUTOFF:-0.5}
BLACKLIST=${BLACKLIST:-PTAY0577P9S1,PTAY0599P8S1,PTAY0666P7S1,PTAY0682P7S1,PTAY0689P8H1}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
SKIP_POOL_SWEEP=${SKIP_POOL_SWEEP:-0}
SKIP_FP_FN=${SKIP_FP_FN:-0}
CLEAN_OLD=${CLEAN_OLD:-0}
DRY_RUN=${DRY_RUN:-0}

IFS=',' read -r -a SIZE_ARR <<< "$POOL_SIZES"
N_SIZES=${#SIZE_ARR[@]}
ARRAY_LAST=$((N_SIZES - 1))

N_FLAG_JOBS=$(( (FLAG_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
if [ "$N_FLAG_JOBS" -gt "$MAX_ARRAY_JOBS" ]; then
    REPEATS_PER_JOB=$(( (FLAG_REPEATS + MAX_ARRAY_JOBS - 1) / MAX_ARRAY_JOBS ))
    N_FLAG_JOBS=$(( (FLAG_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
fi
FLAG_ARRAY_LAST=$((N_FLAG_JOBS - 1))
FLAG_BASE="${OUT_BASE}/fixed_flags_ez45"

mkdir -p "$OUT_BASE"
echo "OUT_BASE          : $OUT_BASE"
echo "pool sizes        : n=$N_SIZES (step=${POOL_STEP}, ${SIZE_ARR[0]}..${SIZE_ARR[-1]})"
echo "pool repeats      : ${FIXED_REPEATS} (FIXED_REPEATS)"
echo "MCC cutoff        : signal_ratio>=${RATIO_CUTOFF} (core eval only)"
echo "FP+FN flags       : ${FLAG_REPEATS} reps @ ez=4.5 -> ${FLAG_BASE} (skip=${SKIP_FP_FN})"
echo "blacklist         : $BLACKLIST"

if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] skip_pool=${SKIP_POOL_SWEEP} skip_fp_fn=${SKIP_FP_FN}"
    exit 0
fi

if [ "$CLEAN_OLD" = 1 ]; then
    echo "Cleaning previous fixed/plots"
    rm -rf "${OUT_BASE}/fixed" "${OUT_BASE}/filtered" "${OUT_BASE}/plots"
    if [ "$SKIP_FP_FN" != 1 ]; then
        echo "Cleaning previous fp_fn_density/fixed_flags_ez45"
        rm -rf "${OUT_BASE}/fp_fn_density" "${FLAG_BASE}"
    fi
fi

deps_for_plot=""
if [ "$SKIP_POOL_SWEEP" = 0 ]; then
    export POOL_SIZES SEED FILL_SEED FF_MIN SIF CUTOFF=3.0 BLACKLIST
    export COMBO_MODE=fixed TOTAL_REPEATS=$FIXED_REPEATS EZ_CUTOFF=4.5
    fixed_job=$(sbatch --parsable --job-name=pool_fixed \
        --array="0-${ARRAY_LAST}" \
        run_pool_size_auc.slurm "$INPUT_DIR" "$OUT_BASE")
    echo "Submitted fixed array job_id=${fixed_job}"
    deps_for_plot="afterany:${fixed_job}"
else
    echo "SKIP_POOL_SWEEP=1 — reusing existing ${OUT_BASE}/fixed/pool_*"
fi

PLOT_MCC_WRAP="cd '$SCRIPT_DIR' && singularity exec -B /lustre1,/lustre2,/appsnew '$SIF' \
    python3 plot_pool_size_mcc.py --sweep-base '$OUT_BASE' --output-dir '$OUT_BASE/plots' \
        --ff-min '$FF_MIN' --ratio-cutoff '$RATIO_CUTOFF' --blacklist '$BLACKLIST'"
PLOT_AUC_WRAP="cd '$SCRIPT_DIR' && singularity exec -B /lustre1,/lustre2,/appsnew '$SIF' \
    python3 plot_pool_size_auc.py --sweep-base '$OUT_BASE' --output-dir '$OUT_BASE/plots' \
        --ff-min '$FF_MIN'"
if [ -n "$deps_for_plot" ]; then
    plot_mcc_job=$(sbatch --parsable --job-name=plot_pool_mcc \
        --dependency="$deps_for_plot" --wrap="$PLOT_MCC_WRAP")
    plot_auc_job=$(sbatch --parsable --job-name=plot_pool_auc \
        --dependency="$deps_for_plot" --wrap="$PLOT_AUC_WRAP")
else
    plot_mcc_job=$(sbatch --parsable --job-name=plot_pool_mcc --wrap="$PLOT_MCC_WRAP")
    plot_auc_job=$(sbatch --parsable --job-name=plot_pool_auc --wrap="$PLOT_AUC_WRAP")
fi
echo "Submitted MCC plot job_id=${plot_mcc_job}"
echo "Submitted AUC plot job_id=${plot_auc_job}"

if [ "$SKIP_FP_FN" != 1 ]; then
    # --- FP+FN at ez=4.5: generate flags then summarize ---
    export TOTAL_REPEATS=$FLAG_REPEATS REPEATS_PER_JOB REF_N=40 SEED SIF
    export EP_THRESHOLD=0.5 EP_RECALL=0.65 Z_THRESHOLD=0.85 Z_RECALL=0.95
    export CUTOFF=3.0 EZ_CUTOFF=4.5
    mkdir -p "$FLAG_BASE"
    flag_job=$(sbatch --parsable --job-name=fixed_flags45 \
        --array="0-${FLAG_ARRAY_LAST}" \
        run_fixed_flags.slurm "$INPUT_DIR" "$FLAG_BASE")
    echo "Submitted ez=4.5 flags array job_id=${flag_job}"

    fp_job=$(sbatch --parsable --job-name=fp_fn_dens \
        --dependency="afterok:${flag_job}" \
        --wrap="cd '$SCRIPT_DIR' && singularity exec -B /lustre1,/lustre2,/appsnew '$SIF' \
            python3 summarize_fp_fn_density.py \
                --flag-dir '${FLAG_BASE}/fixed_flags' \
                --output-dir '$OUT_BASE/fp_fn_density' \
                --ff-min '$FF_MIN' \
                --score all \
                --blacklist '$BLACKLIST'")
    echo "Submitted FP+FN dens job_id=${fp_job}"
else
    echo "SKIP_FP_FN=1 — leaving existing fp_fn_density / fixed_flags_ez45 untouched"
fi

echo "Outputs:"
echo "  $OUT_BASE/plots/pool_size_mcc.html"
echo "  $OUT_BASE/plots/pool_size_auc.html"
if [ "$SKIP_FP_FN" != 1 ]; then
    echo "  $OUT_BASE/fp_fn_density/fp_fn_density.html"
fi
