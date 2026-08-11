#!/usr/bin/bash
# Submit a recall-grid of episcore or zscore jobs.
#
#   ./submit_recall_grid.sh episcore|zscore|male_ref_episcore|male_ref_zscore|female_ref_episcore|female_ref_zscore [--test] [--dry-run]

set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
TEST=${TEST:-0}
MODE=${1:?mode required}
shift || true
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -t|--test) TEST=1 ;;
        -h|--help)
            sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown: $arg" >&2; exit 2 ;;
    esac
done

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$WORKDIR"
mkdir -p logs

INPUT=/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260731-240k_middle_normal_samples_plus_early_allosomes_samples
OUTPUT=/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260731-240k_middle_normal_samples_plus_early_allosomes_samples
CPG_DIR=/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260525-grid_search_240k_panel_240k_model/recall_list_240k
MAX_JOBS=25
SLEEP_BETWEEN=2
SLEEP_FULL=60
USER_NAME=$(whoami)

case "$MODE" in
    episcore)
        META="${INPUT}/episcore_samples_meta.csv"
        OUT_BASE="${OUTPUT}/episcore_recall_conventional"
        SLURM=run_episcore_recall.slurm
        PREFIX=me31_e_r
        ;;
    zscore)
        META="${INPUT}/zscore_samples_meta.csv"
        OUT_BASE="${OUTPUT}/zscore_recall_conventional"
        SLURM=run_zscore_recall.slurm
        PREFIX=me31_z_r
        ;;
    male_ref_episcore)
        META="${INPUT}/male_ref_episcore_meta.csv"
        OUT_BASE="${OUTPUT}/male_ref_episcore_recall"
        SLURM=run_episcore_recall.slurm
        PREFIX=me31_mre_r
        ;;
    male_ref_zscore)
        META="${INPUT}/male_ref_zscore_meta.csv"
        OUT_BASE="${OUTPUT}/male_ref_zscore_recall"
        SLURM=run_zscore_recall.slurm
        PREFIX=me31_mrz_r
        ;;
    female_ref_episcore)
        META="${INPUT}/female_ref_episcore_meta.csv"
        OUT_BASE="${OUTPUT}/female_ref_episcore_recall"
        SLURM=run_episcore_recall.slurm
        PREFIX=me31_fre_r
        ;;
    female_ref_zscore)
        META="${INPUT}/female_ref_zscore_meta.csv"
        OUT_BASE="${OUTPUT}/female_ref_zscore_recall"
        SLURM=run_zscore_recall.slurm
        PREFIX=me31_frz_r
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

if [ ! -f "$META" ]; then
    echo "ERROR: missing $META" >&2
    exit 1
fi

mapfile -t RECALLS < <(awk 'BEGIN { for (i = 1; i <= 99; i += 1) printf "%g\n", i / 100 }')
if [ "$TEST" = 1 ]; then
    RECALLS=(0.5)
fi

count_my_jobs() {
    squeue -u "$USER_NAME" -h -o '%j' 2>/dev/null | grep -c "^${PREFIX}" || true
}
wait_for_slot() {
    while :; do
        n=$(count_my_jobs); n=${n:-0}
        [ "$n" -lt "$MAX_JOBS" ] && return
        echo "  [$(date +%H:%M:%S)] ${n} ${PREFIX}* queued; sleep ${SLEEP_FULL}s"
        sleep "$SLEEP_FULL"
    done
}

n_submitted=0
n_skipped=0
for recall in "${RECALLS[@]}"; do
    cpg="${CPG_DIR}/240k_cpg_recall_${recall}.txt"
    [ -f "$cpg" ] || { echo "WARN missing $cpg"; n_skipped=$((n_skipped+1)); continue; }
    out="${OUT_BASE}/recall_${recall}/_analyze_zscore.tsv.gz"
    if [ -f "$out" ]; then
        # For zscore modes, also require non-zero chrX_percentage
        if [[ "$MODE" == *zscore* ]]; then
            ok=$(singularity exec -B /lustre1,/lustre2,/appsnew \
                /lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif \
                python3 -c "import pandas as pd; df=pd.read_csv('${out}',sep='\t'); print(int('chrX_percentage' in df.columns and float(df['chrX_percentage'].max())>0))" 2>/dev/null || echo 0)
            if [ "$ok" = 1 ]; then
                echo "Skip recall=${recall} (valid)"
                n_skipped=$((n_skipped+1))
                continue
            fi
            rm -f "$out" "${OUT_BASE}/recall_${recall}/_reference_percentage.tsv.gz"
        else
            echo "Skip recall=${recall} (exists)"
            n_skipped=$((n_skipped+1))
            continue
        fi
    fi
    job_name="${PREFIX}${recall}"
    if [ "$DRY_RUN" = 1 ]; then
        echo "[DRY-RUN] sbatch --job-name=${job_name} ${SLURM} ${recall} ${META} ${OUT_BASE}"
        n_submitted=$((n_submitted+1))
        continue
    fi
    wait_for_slot
    jobid=$(sbatch --parsable --job-name="$job_name" "$SLURM" "$recall" "$META" "$OUT_BASE")
    echo "Submitted ${job_name} job_id=${jobid}"
    n_submitted=$((n_submitted+1))
    sleep "$SLEEP_BETWEEN"
done
echo "mode=${MODE} submitted=${n_submitted} skipped=${n_skipped}"
