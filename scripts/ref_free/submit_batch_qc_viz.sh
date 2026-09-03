#!/usr/bin/bash
# Expand batch-QC scores to Set A–D viz units, re-run dual-mode ref_free (MIN_FF=-1),
# then compute per-chr stats for notebook plots.
#
#   ./submit_batch_qc_viz.sh [-n|--dry-run]

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

TODAY_BASE=${TODAY_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
MAIN_INPUT=${MAIN_INPUT:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
MQRES=${MQRES:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/mqres_samplesheet.csv}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-2000}
REF_N=${REF_N:-40}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-0}

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown: $arg" >&2; exit 2 ;;
    esac
done

COHORT="$TODAY_BASE/cohort"
MODE_A="$TODAY_BASE/mode_A_ep0.5_0.65_z0.85_0.95"
MODE_B="$TODAY_BASE/mode_B_ep0.1_0.61_z0.9_0.92"
SCORE_A_EP="$MODE_A/scores/episcore"
SCORE_A_Z="$MODE_A/scores/percentage"
SCORE_B_EP="$MODE_B/scores/episcore"
SCORE_B_Z="$MODE_B/scores/percentage"
INPUT_A="$MODE_A/input_fixed"
INPUT_B="$MODE_B/input_fixed"
FIXED_A="$MODE_A/fixed_combo"
FIXED_B="$MODE_B/fixed_combo"
PLOTS="$TODAY_BASE/plots"

N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
ARRAY_LAST=$((N_JOBS - 1))

echo "TODAY_BASE=$TODAY_BASE"
if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] cohort + score expand + dual ref_free + per-chr stats"
    exit 0
fi

mkdir -p "$COHORT" "$SCORE_A_EP" "$SCORE_A_Z" "$SCORE_B_EP" "$SCORE_B_Z" "$PLOTS" logs

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 build_batch_qc_viz_cohort.py \
        --mqres "$MQRES" --meta "$META" --output-dir "$COHORT"

UNITS_CSV="$COHORT/viz_units.csv"
N_UNITS=$(singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 -c "import pandas as pd; print(len(pd.read_csv('$UNITS_CSV')))")
UNIT_LAST=$((N_UNITS - 1))
echo "viz units=$N_UNITS"

job_za=$(sbatch --parsable --job-name=bqc_viz_za \
    --array="0-${UNIT_LAST}%80" \
    run_batch_qc_percentage.slurm "$UNITS_CSV" "$SCORE_A_Z" 0.85 0.95)
job_ea=$(sbatch --parsable --job-name=bqc_viz_ea \
    --array="0-${UNIT_LAST}%80" \
    run_batch_qc_episcore.slurm "$UNITS_CSV" "$SCORE_A_EP" 0.5 0.65)
job_zb=$(sbatch --parsable --job-name=bqc_viz_zb \
    --array="0-${UNIT_LAST}%80" \
    run_batch_qc_percentage.slurm "$UNITS_CSV" "$SCORE_B_Z" 0.9 0.92)
echo "score jobs za=$job_za ea=$job_ea zb=$job_zb"

cat > "$TODAY_BASE/_run_viz_after_scores.sh" <<EOF
#!/usr/bin/bash
set -euo pipefail
cd "$SCRIPT_DIR"
export SIF="$SIF"
export TOTAL_REPEATS=$TOTAL_REPEATS REPEATS_PER_JOB=$REPEATS_PER_JOB REF_N=$REF_N SEED=$SEED
export COMPRESS=1 COMBO_MODE=fixed STORE_PAIR_COUNTS=0
export CUTOFF=3.0 MIN_FF=-1
export EZ_CUTOFF_MIN=3.0 EZ_CUTOFF_MAX=4.5 EZ_CUTOFF_STEP=0.1
unset EP_THRESHOLD_MIN EP_THRESHOLD_MAX EP_RECALL_MIN EP_RECALL_MAX
unset Z_THRESHOLD_MIN Z_THRESHOLD_MAX Z_RECALL_MIN Z_RECALL_MAX

singularity exec -B /lustre1,/lustre2,/appsnew "\$SIF" \\
    python3 prepare_batch_qc_assets.py \\
        --main-input "$MAIN_INPUT" \\
        --units "$UNITS_CSV" \\
        --ep-dir "$SCORE_A_EP" --z-dir "$SCORE_A_Z" \\
        --output-dir "$INPUT_A" \\
        --ep-threshold 0.5 --ep-recall 0.65 \\
        --z-threshold 0.85 --z-recall 0.95 \\
        --allow-partial

if ls "$SCORE_B_EP"/*.episcore.tsv >/dev/null 2>&1; then
    singularity exec -B /lustre1,/lustre2,/appsnew "\$SIF" \\
        python3 prepare_batch_qc_assets.py \\
            --main-input "$MAIN_INPUT" \\
            --units "$UNITS_CSV" \\
            --ep-dir "$SCORE_B_EP" --z-dir "$SCORE_B_Z" \\
            --output-dir "$INPUT_B" \\
            --ep-threshold 0.1 --ep-recall 0.61 \\
            --z-threshold 0.9 --z-recall 0.92 \\
            --allow-partial
fi

submit_mode () {
    local name=\$1 input=\$2 fixed=\$3 ep_t=\$4 ep_r=\$5 z_t=\$6 z_r=\$7 out_tsv=\$8
    [[ -f "\$input/meta.csv" ]] || { echo "[skip] \$name"; return 0; }
    rm -rf "\${fixed}/ref_free_ezscore"
    mkdir -p "\${fixed}/ref_free_ezscore" "\${fixed}/plots"
    export EP_THRESHOLD=\$ep_t EP_RECALL=\$ep_r Z_THRESHOLD=\$z_t Z_RECALL=\$z_r
    local j
    j=\$(sbatch --parsable --job-name="rf_\${name}" \\
        --array="0-${ARRAY_LAST}" \\
        run_ref_free_ezscore.slurm "\$input" "\$fixed")
    echo "ref_free \$name=\$j"
    sbatch --parsable --job-name="agg_\${name}" \\
        --dependency="afterok:\${j}" \\
        run_aggregate_and_plot.slurm "\$fixed" "batch_qc \${name}" "0.01" 0
    sbatch --parsable --job-name="pch_\${name}" \\
        --dependency="afterok:\${j}" \\
        run_batch_qc_per_chr.slurm "\$input" "\$fixed" "\$out_tsv"
}

submit_mode modeA "$INPUT_A" "$FIXED_A" 0.5 0.65 0.85 0.95 "$PLOTS/modeA_per_chr_stats.tsv"
submit_mode modeB "$INPUT_B" "$FIXED_B" 0.1 0.61 0.9 0.92 "$PLOTS/modeB_per_chr_stats.tsv"
EOF
chmod +x "$TODAY_BASE/_run_viz_after_scores.sh"

sbatch --job-name=bqc_viz_prep \
    --dependency="afterok:${job_za}:${job_ea}:${job_zb}" \
    --partition=cn-long --cpus-per-task=4 --mem=32G --time=2:00:00 \
    -o logs/bqc_viz_prep_%j.log -e logs/bqc_viz_prep_%j.log \
    --wrap="bash '$TODAY_BASE/_run_viz_after_scores.sh'"

echo "Submitted score expand + viz prepare chain"
echo "Per-chr stats -> $PLOTS/{modeA,modeB}_per_chr_stats.tsv"
echo "Notebook -> notebooks/aipt_2.0/ref_free_batch_qc.ipynb"
