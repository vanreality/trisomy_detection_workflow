#!/usr/bin/bash
# Launch Nextflow EXTRACT_BETA (grid_search step) for batch-QC units.
#
# Mode A missing beta  → extract at threshold 0.5 (early profile)
# Mode B all units     → extract at threshold 0.1 then harvest episcore recall 0.61
#
# Usage:
#   TODAY_BASE=... ./submit_batch_qc_nf_extract.sh [--mode A|B|both] [-n]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/lustre1/cqyi/AIPT_2.0/workflow/episcore
TODAY_BASE=${TODAY_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
UNITS_CSV=${UNITS_CSV:-${TODAY_BASE}/units/nf_split_bam_samplesheet.csv}
MODE=${MODE:-both}
DRY_RUN=${DRY_RUN:-0}
# early supplies fasta/cpg; grid_search supplies EXTRACT_BETA resources; CLI --threshold overrides
PROFILE=${PROFILE:-early,grid_search,alioth_slurm,singularity}
NF=${NF:-/appsnew/home/syfan/softwares/nextflow/nextflow}

for arg in "$@"; do
    case "$arg" in
        A|B|both) MODE=$arg ;;
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

if [[ ! -f "$UNITS_CSV" ]]; then
    echo "Missing units samplesheet: $UNITS_CSV" >&2
    echo "Run submit_batch_qc_dual_mode.sh first (or build_batch_qc_units.py)." >&2
    exit 1
fi
if [[ ! -x "$NF" ]]; then
    echo "nextflow not executable: $NF" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs" "${TODAY_BASE}/nf_logs"

run_one () {
    local thr=$1
    local out=$2
    local name=$3
    echo "=== $name threshold=$thr out=$out ==="
    if [ "$DRY_RUN" = 1 ]; then
        echo "[DRY-RUN] sbatch ${NF} run ... --threshold $thr --outdir $out"
        return 0
    fi
    mkdir -p "$out"
    local jid
    jid=$(sbatch --parsable --job-name="${name}" \
        --partition=cn-long --cpus-per-task=4 --mem=8G --time=48:00:00 \
        --output="${SCRIPT_DIR}/logs/${name}_%j.log" \
        --wrap="mkdir -p '${out}/work' && \
            cd '${out}' && \
            '${NF}' run '${REPO}/main.nf' \
              -profile '${PROFILE}' \
              -name '${name}' \
              -w '${out}/work' \
              --step grid_search \
              --input '${UNITS_CSV}' \
              --outdir '${out}' \
              --threshold ${thr}")
    echo "$jid" > "${TODAY_BASE}/nf_logs/${name}.jobid"
    echo "Submitted nextflow launcher job_id=${jid}"
}

case "$MODE" in
    A) run_one 0.5 "${TODAY_BASE}/nf_extract_thr0.5" nf_bqc_thr05 ;;
    B) run_one 0.1 "${TODAY_BASE}/nf_extract_thr0.1" nf_bqc_thr01 ;;
    both)
        run_one 0.5 "${TODAY_BASE}/nf_extract_thr0.5" nf_bqc_thr05
        run_one 0.1 "${TODAY_BASE}/nf_extract_thr0.1" nf_bqc_thr01
        ;;
    *)
        echo "MODE must be A|B|both" >&2
        exit 2
        ;;
esac

echo "After NF extract finishes:"
echo "  singularity exec -B /lustre1,/lustre2,/appsnew containers/common_tools.sif \\"
echo "    python3 harvest_batch_qc_episcore.py \\"
echo "      --units ${TODAY_BASE}/units/unit_samplesheet.csv \\"
echo "      --nf-outdir ${TODAY_BASE}/nf_extract_thr0.1 \\"
echo "      --ep-dir ${TODAY_BASE}/mode_B_ep0.1_0.61_z0.9_0.92/scores/episcore \\"
echo "      --threshold 0.1 --recall 0.61"
echo "  then: bash ${TODAY_BASE}/_run_after_scores.sh"
