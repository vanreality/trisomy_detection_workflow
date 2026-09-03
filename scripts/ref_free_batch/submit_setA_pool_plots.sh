#!/usr/bin/bash
# Set A pool-size exploration: prepare assets, sweep modeA+modeB, plot + markdown.
#
#   ./submit_setA_pool_plots.sh [-n|--dry-run]
#   TOTAL_REPEATS=20000 ./submit_setA_pool_plots.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

RESULT_ROOT=${RESULT_ROOT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_free_pool_plus_batch}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
FILL_SEED=${FILL_SEED:-7}
DRY_RUN=${DRY_RUN:-0}
SKIP_PREPARE=${SKIP_PREPARE:-0}
SKIP_SWEEP=${SKIP_SWEEP:-0}

POOL_MIN=${POOL_MIN:-20}
POOL_MAX=${POOL_MAX:-160}
POOL_STEP=${POOL_STEP:-2}
if [ -z "${POOL_SIZES:-}" ]; then
    POOL_SIZES=$(POOL_MIN=$POOL_MIN POOL_MAX=$POOL_MAX POOL_STEP=$POOL_STEP python3 - <<'PY'
import os
lo, hi, step = int(os.environ["POOL_MIN"]), int(os.environ["POOL_MAX"]), int(os.environ["POOL_STEP"])
print(",".join(str(p) for p in range(lo, hi + 1, step)))
PY
)
fi

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        --skip-prepare) SKIP_PREPARE=1 ;;
        --skip-sweep) SKIP_SWEEP=1 ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a SIZE_ARR <<< "$POOL_SIZES"
ARRAY_LAST=$((${#SIZE_ARR[@]} - 1))

echo "RESULT_ROOT   : $RESULT_ROOT"
echo "pool sizes    : n=${#SIZE_ARR[@]} (${SIZE_ARR[0]}..${SIZE_ARR[-1]})"
echo "repeats       : $TOTAL_REPEATS"
echo "dry-run       : $DRY_RUN"

if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] prepare both modes, array 0-${ARRAY_LAST} x modeA/modeB, then plot"
    exit 0
fi

mkdir -p "$RESULT_ROOT/plots" logs

if [ "$SKIP_PREPARE" != 1 ]; then
    echo "build Set A from meta_samplesheet"
    singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
        python3 build_setA_cohort.py --output-dir "$RESULT_ROOT/cohort"
    echo "Cleaning previous pool_* outputs"
    rm -rf "$RESULT_ROOT"/modeA/pool_* "$RESULT_ROOT"/modeB/pool_*
    rm -f "$RESULT_ROOT"/plots/*ff_lt*
    for mode in modeA modeB; do
        echo "prepare $mode"
        singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
            python3 prepare_setA_assets.py \
                --mode "$mode" \
                --cohort-dir "$RESULT_ROOT/cohort" \
                --output-dir "$RESULT_ROOT/$mode/input"
    done
fi

export RESULT_ROOT SIF TOTAL_REPEATS SEED FILL_SEED POOL_SIZES

deps=()
if [ "$SKIP_SWEEP" != 1 ]; then
    job_a=$(sbatch --parsable --job-name=setA_pA \
        --array="0-${ARRAY_LAST}" \
        run_pool_size_setA.slurm modeA)
    job_b=$(sbatch --parsable --job-name=setA_pB \
        --array="0-${ARRAY_LAST}" \
        run_pool_size_setA.slurm modeB)
    echo "Submitted modeA=${job_a} modeB=${job_b}"
    deps+=("$job_a" "$job_b")
fi

if [ "${#deps[@]}" -gt 0 ]; then
    IFS=:
    dep_arg="--dependency=afterok:${deps[*]}"
    unset IFS
    plot_job=$(sbatch --parsable --job-name=setA_plot $dep_arg \
        run_plot_and_summarize.slurm)
else
    plot_job=$(sbatch --parsable --job-name=setA_plot \
        run_plot_and_summarize.slurm)
fi
echo "Submitted plot+summary job=${plot_job}"
echo "HTMLs -> $RESULT_ROOT/plots/"
echo "README -> $RESULT_ROOT/README.md"
