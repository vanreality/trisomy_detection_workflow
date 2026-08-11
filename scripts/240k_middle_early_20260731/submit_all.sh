#!/usr/bin/bash
# Orchestrate 20260731 early-allosomes + middle-normal analysis.
#
# Usage:
#   ./submit_all.sh prepare
#   ./submit_all.sh nextflow [early|middle|both]
#   ./submit_all.sh prepare              # after NF: refresh betas
#   ./submit_all.sh chry                 # FF + chrY ratios
#   ./submit_all.sh assign_gender        # middle fetal gender from early fits
#   ./submit_all.sh prepare              # refresh middle-ref metas
#   ./submit_all.sh episcore [--test]    # conventional early female ref
#   ./submit_all.sh zscore   [--test]
#   ./submit_all.sh features [--test]    # percentage + z_intra matrix jobs
#   ./submit_all.sh male_ref_episcore|male_ref_zscore|female_ref_episcore|female_ref_zscore [--test]
#   ./submit_all.sh collect              # tables + plots
#
set -euo pipefail

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$WORKDIR"
mkdir -p logs

SIF=/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif
# Bind host fonts so matplotlib PDFs can render CJK labels.
PY=(singularity exec -B /lustre1,/lustre2,/appsnew,/usr/share/fonts
    --env PYTHONPATH="${WORKDIR}/episcore:/lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/grid_search_240k"
    "$SIF" python3)

cmd=${1:-}
shift || true

case "$cmd" in
    prepare)
        "${PY[@]}" prepare_inputs.py
        ;;
    nextflow)
        ./submit_nextflow.sh "${1:-both}" "${@:2}"
        ;;
    chry)
        "${PY[@]}" collect_chry_ff.py
        ;;
    assign_gender)
        "${PY[@]}" assign_middle_gender.py "$@"
        ;;
    episcore)
        ./submit_recall_grid.sh episcore "$@"
        ;;
    zscore)
        ./submit_recall_grid.sh zscore "$@"
        ;;
    features)
        # submit feature jobs via thin wrapper
        DRY_RUN=0; TEST=0
        for arg in "$@"; do
            case "$arg" in -n|--dry-run) DRY_RUN=1 ;; -t|--test) TEST=1 ;; esac
        done
        mapfile -t RECALLS < <(awk 'BEGIN { for (i = 1; i <= 99; i += 1) printf "%g\n", i / 100 }')
        [ "$TEST" = 1 ] && RECALLS=(0.5)
        MAX_JOBS=20; USER_NAME=$(whoami); PREFIX=me31_f_r
        n_sub=0; n_skip=0
        for recall in "${RECALLS[@]}"; do
            out=/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260731-240k_middle_normal_samples_plus_early_allosomes_samples/features_recall/recall_${recall}/features.tsv.gz
            if [ -f "$out" ]; then echo "Skip features recall=${recall}"; n_skip=$((n_skip+1)); continue; fi
            while [ "$(squeue -u "$USER_NAME" -h -o '%j' 2>/dev/null | grep -c "^${PREFIX}" || true)" -ge "$MAX_JOBS" ]; do
                sleep 60
            done
            if [ "$DRY_RUN" = 1 ]; then
                echo "[DRY-RUN] features recall=${recall}"; n_sub=$((n_sub+1)); continue
            fi
            jobid=$(sbatch --parsable --job-name="${PREFIX}${recall}" run_features_recall.slurm "$recall")
            echo "Submitted features recall=${recall} job_id=${jobid}"
            n_sub=$((n_sub+1)); sleep 2
        done
        echo "features submitted=${n_sub} skipped=${n_skip}"
        ;;
    male_ref_episcore|male_ref_zscore|female_ref_episcore|female_ref_zscore)
        ./submit_recall_grid.sh "$cmd" "$@"
        ;;
    collect)
        "${PY[@]}" collect_chry_ff.py
        "${PY[@]}" collect_tables.py
        "${PY[@]}" plot_all.py
        "${PY[@]}" plot_ff_score_2x2.py
        ;;
    plot_ff_score)
        "${PY[@]}" plot_ff_score_2x2.py "$@"
        ;;
    -h|--help|help|"")
        sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "Unknown command: $cmd" >&2
        exit 2
        ;;
esac
