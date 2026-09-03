#!/usr/bin/bash
# Batch-quality dual-mode reference-free workflow.
#
# Modes:
#   A: ep thr=0.5 / recall=0.65 ; z thr=0.85 / recall=0.95
#   B: ep thr=0.1 / recall=0.61 ; z thr=0.9  / recall=0.92
#
# Steps:
#   1) Expand mqres → unit_samplesheet (one row per sample×batch_key)
#   2) Compute z percentages from deconv for both modes (SLURM array)
#   3) Compute Mode-A episcore from production wide/beta@0.5
#   4) Prepare slim assets + submit ref_free_ezscore for each mode
#   5) Mode-B episcore needs beta@0.1 — emits nf_split_bam_samplesheet +
#      launch hint (see submit_batch_qc_nf_extract.sh)
#
# Usage:
#   ./submit_batch_qc_dual_mode.sh [-n|--dry-run] [--multi-batch-only]
#   TODAY_BASE=... TOTAL_REPEATS=5000 ./submit_batch_qc_dual_mode.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

TODAY_BASE=${TODAY_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
MQRES=${MQRES:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/mqres_samplesheet.csv}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
MAIN_INPUT=${MAIN_INPUT:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-2000}
MAX_ARRAY_JOBS=${MAX_ARRAY_JOBS:-50}
REF_N=${REF_N:-40}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-0}
MULTI_ONLY=${MULTI_ONLY:-0}
SKIP_SCORES=${SKIP_SCORES:-0}
SKIP_REF_FREE=${SKIP_REF_FREE:-0}

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        --multi-batch-only) MULTI_ONLY=1 ;;
        --skip-scores) SKIP_SCORES=1 ;;
        --skip-ref-free) SKIP_REF_FREE=1 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

UNITS_DIR="${TODAY_BASE}/units"
MODE_A="${TODAY_BASE}/mode_A_ep0.5_0.65_z0.85_0.95"
MODE_B="${TODAY_BASE}/mode_B_ep0.1_0.61_z0.9_0.92"
SCORE_A_EP="${MODE_A}/scores/episcore"
SCORE_A_Z="${MODE_A}/scores/percentage"
SCORE_B_EP="${MODE_B}/scores/episcore"
SCORE_B_Z="${MODE_B}/scores/percentage"
INPUT_A="${MODE_A}/input_fixed"
INPUT_B="${MODE_B}/input_fixed"
FIXED_A="${MODE_A}/fixed_combo"
FIXED_B="${MODE_B}/fixed_combo"

N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
if [ "$N_JOBS" -gt "$MAX_ARRAY_JOBS" ]; then
    REPEATS_PER_JOB=$(( (TOTAL_REPEATS + MAX_ARRAY_JOBS - 1) / MAX_ARRAY_JOBS ))
    N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
fi
ARRAY_LAST=$((N_JOBS - 1))

echo "Today base : $TODAY_BASE"
echo "mqres      : $MQRES"
echo "multi-only : $MULTI_ONLY"
echo "ref_free   : ${REF_N}+${REF_N} x ${TOTAL_REPEATS} (array 0-${ARRAY_LAST})"

BUILD_FLAGS=()
if [ "$MULTI_ONLY" = 1 ]; then
    BUILD_FLAGS+=(--multi-batch-only)
fi

if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] build units + score arrays + dual ref_free"
    exit 0
fi

mkdir -p "$UNITS_DIR" "$SCORE_A_EP" "$SCORE_A_Z" "$SCORE_B_EP" "$SCORE_B_Z" \
    "$FIXED_A/plots" "$FIXED_B/plots" logs

# --- 1) units ---
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 build_batch_qc_units.py \
        --mqres "$MQRES" \
        --meta "$META" \
        --output-dir "$UNITS_DIR" \
        "${BUILD_FLAGS[@]}"

UNITS_CSV="${UNITS_DIR}/unit_samplesheet.csv"
N_UNITS=$(singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 -c "import pandas as pd; print(len(pd.read_csv('$UNITS_CSV')))")
UNIT_LAST=$((N_UNITS - 1))
echo "Units: $N_UNITS (array 0-${UNIT_LAST})"

score_deps=()

if [ "$SKIP_SCORES" != 1 ]; then
    # --- 2) percentages (both modes) ---
    job_za=$(sbatch --parsable --job-name=bqc_za \
        --array="0-${UNIT_LAST}%80" \
        run_batch_qc_percentage.slurm "$UNITS_CSV" "$SCORE_A_Z" 0.85 0.95)
    job_zb=$(sbatch --parsable --job-name=bqc_zb \
        --array="0-${UNIT_LAST}%80" \
        run_batch_qc_percentage.slurm "$UNITS_CSV" "$SCORE_B_Z" 0.9 0.92)
    echo "Submitted z% Mode A job=${job_za} Mode B job=${job_zb}"
    score_deps+=("$job_za" "$job_zb")

    # --- 3) Mode A episcore from production ---
    job_ea=$(sbatch --parsable --job-name=bqc_ea \
        --array="0-${UNIT_LAST}%80" \
        run_batch_qc_episcore.slurm "$UNITS_CSV" "$SCORE_A_EP" 0.5 0.65)
    echo "Submitted ep Mode A job=${job_ea}"
    score_deps+=("$job_ea")
fi

dep_arg=""
if [ "${#score_deps[@]}" -gt 0 ]; then
    IFS=:
    # afterok: every array task must succeed. Huge TXT deconv units can TIMEOUT
    # and leave DependencyNeverSatisfied — raise percentage walltime and/or
    # resubmit missing indices then re-run run_batch_qc_after_scores.slurm.
    dep_arg="--dependency=afterok:${score_deps[*]}"
    unset IFS
fi

# --- 4) prepare + ref_free after scores ---
cat > "${TODAY_BASE}/_run_after_scores.sh" <<EOF
#!/usr/bin/bash
set -euo pipefail
cd "$SCRIPT_DIR"
SIF="$SIF"
UNITS_CSV="$UNITS_CSV"
MAIN_INPUT="$MAIN_INPUT"

singularity exec -B /lustre1,/lustre2,/appsnew "\$SIF" \\
    python3 prepare_batch_qc_assets.py \\
        --main-input "\$MAIN_INPUT" \\
        --units "\$UNITS_CSV" \\
        --ep-dir "$SCORE_A_EP" \\
        --z-dir "$SCORE_A_Z" \\
        --output-dir "$INPUT_A" \\
        --ep-threshold 0.5 --ep-recall 0.65 \\
        --z-threshold 0.85 --z-recall 0.95 \\
        --allow-partial

# Mode B: only units that already have ep@0.1 (usually none until NF extract)
if ls "$SCORE_B_EP"/*.episcore.tsv >/dev/null 2>&1; then
    singularity exec -B /lustre1,/lustre2,/appsnew "\$SIF" \\
        python3 prepare_batch_qc_assets.py \\
            --main-input "\$MAIN_INPUT" \\
            --units "\$UNITS_CSV" \\
            --ep-dir "$SCORE_B_EP" \\
            --z-dir "$SCORE_B_Z" \\
            --output-dir "$INPUT_B" \\
            --ep-threshold 0.1 --ep-recall 0.61 \\
            --z-threshold 0.9 --z-recall 0.92 \\
            --allow-partial
else
    echo "[warn] Mode B episcore dir empty — run submit_batch_qc_nf_extract.sh first"
fi

export TOTAL_REPEATS=$TOTAL_REPEATS REPEATS_PER_JOB=$REPEATS_PER_JOB REF_N=$REF_N SEED=$SEED
export SIF COMPRESS=1 COMBO_MODE=fixed STORE_PAIR_COUNTS=0
export CUTOFF=3.0 MIN_FF=0
export EZ_CUTOFF_MIN=3.0 EZ_CUTOFF_MAX=4.5 EZ_CUTOFF_STEP=0.1
unset EP_THRESHOLD_MIN EP_THRESHOLD_MAX EP_RECALL_MIN EP_RECALL_MAX
unset Z_THRESHOLD_MIN Z_THRESHOLD_MAX Z_RECALL_MIN Z_RECALL_MAX

submit_mode () {
    local name=\$1 input=\$2 fixed=\$3 ep_t=\$4 ep_r=\$5 z_t=\$6 z_r=\$7
    if [[ ! -f "\$input/meta.csv" ]]; then
        echo "[skip] no input for \$name"
        return 0
    fi
    rm -rf "\${fixed}/ref_free_ezscore"
    mkdir -p "\${fixed}/ref_free_ezscore" "\${fixed}/plots"
    export EP_THRESHOLD=\$ep_t EP_RECALL=\$ep_r Z_THRESHOLD=\$z_t Z_RECALL=\$z_r
    local j
    j=\$(sbatch --parsable --job-name="rf_\${name}" \\
        --array="0-${ARRAY_LAST}" \\
        run_ref_free_ezscore.slurm "\$input" "\$fixed")
    echo "Submitted ref_free \$name job=\$j"
    sbatch --parsable --job-name="agg_\${name}" \\
        --dependency="afterok:\${j}" \\
        run_aggregate_and_plot.slurm \\
        "\$fixed" \\
        "batch_qc \${name} ${REF_N}+${REF_N}" \\
        "0.01" \\
        0
}

submit_mode modeA "$INPUT_A" "$FIXED_A" 0.5 0.65 0.85 0.95
submit_mode modeB "$INPUT_B" "$FIXED_B" 0.1 0.61 0.9 0.92
EOF
chmod +x "${TODAY_BASE}/_run_after_scores.sh"

if [ "$SKIP_REF_FREE" = 1 ]; then
    echo "Skipped ref_free submit. After scores finish, run:"
    echo "  ${TODAY_BASE}/_run_after_scores.sh"
    exit 0
fi

if [ -n "$dep_arg" ]; then
    # shellcheck disable=SC2086
    sbatch --job-name=bqc_prep $dep_arg \
        run_batch_qc_after_scores.slurm "$TODAY_BASE"
    echo "Submitted prepare+ref_free after score deps: ${score_deps[*]}"
else
    bash "${TODAY_BASE}/_run_after_scores.sh"
fi

echo "NF extract (Mode B ep / missing Mode A beta):"
echo "  ./submit_batch_qc_nf_extract.sh"
echo "Outputs: $TODAY_BASE"
