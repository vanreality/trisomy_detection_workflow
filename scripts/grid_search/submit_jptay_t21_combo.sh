#!/usr/bin/bash
# Recompute JPTAY T21 ezscore under better chr21 epi/z combos.
#
# Keeps production references unchanged:
#   17 early_ref  -> episcore / zscore mu,sigma
#   25 ez-ref     -> ezscore mu,sigma
# Reference raw scores are reused from fixed_ez25/input_with_missing4 parquets
# (ref_40 grid + 4 ez-ref Normals at production combo only).
#
# Query batches (from mqres, plus optional re-run sheets) need new scores:
#   zscore  : deconv -> percentage grid (one SLURM job per unit)
#   episcore: Nextflow EXTRACT_BETA at each epi threshold (full CpG panel),
#             then recall grid (one SLURM job per unit x threshold)
#
# Usage:
#   ./submit_jptay_t21_combo.sh [-n|--dry-run] [--skip-nf]
#
# Env overrides:
#   OUT_BASE  META  MQRES  SAMPLES  GRID_INPUT  SIF  PROFILE  NF

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

REPO=/lustre1/cqyi/AIPT_2.0/workflow/episcore
OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260824-grid_search_res_for_jptay}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
MQRES=${MQRES:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/mqres_samplesheet.csv}
SAMPLES=${SAMPLES:-JPTAY1835P7H1,JPTAY1927P8H1,JPTAY1964P9H1}
GRID_INPUT=${GRID_INPUT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size/fixed_ez25/input_with_missing4}
SIF=${SIF:-${REPO}/containers/common_tools.sif}
PROFILE=${PROFILE:-early,grid_search,alioth_slurm,singularity}
NF=${NF:-/appsnew/home/syfan/softwares/nextflow/nextflow}
CPG_DIR=${CPG_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260525-grid_search_240k_panel_240k_model/recall_list_220k}

DRY_RUN=${DRY_RUN:-0}
SKIP_NF=${SKIP_NF:-0}

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        --skip-nf) SKIP_NF=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

UNITS_DIR="${OUT_BASE}/units"
QUERY_Z="${OUT_BASE}/query_zscore"
QUERY_EP="${OUT_BASE}/query_episcore"
SEARCH_DIR="${OUT_BASE}/search"
NF_ROOT="${OUT_BASE}/nf_extract"

echo "OUT_BASE : $OUT_BASE"
echo "samples  : $SAMPLES"
echo "mqres    : $MQRES"
echo "grid     : $GRID_INPUT"
echo "skip-nf  : $SKIP_NF"
if [ "$DRY_RUN" = 1 ]; then
    echo "Mode     : DRY-RUN"
fi
echo

run_py() {
    if [ "$DRY_RUN" = 1 ]; then
        echo "[DRY-RUN] singularity exec ... python3 $*"
        return 0
    fi
    singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" python3 "$@"
}

submit() {
    local extra="$1"
    shift
    if [ "$DRY_RUN" = 1 ]; then
        echo "[DRY-RUN] sbatch --parsable $extra $*"
        echo "dryrun"
        return 0
    fi
    # shellcheck disable=SC2086
    sbatch --parsable $extra "$@"
}

nf_name_tag() {
    # Nextflow -name rejects dots; keep digits only (0.1 -> thr01, 0.33 -> thr033).
    local t="$1"
    echo "jptay_nf_thr${t//./}"
}

# --- 1. units + combo lists + resolved ref names ---------------------------
echo "=== prepare units ==="
mkdir -p "$UNITS_DIR"
run_py prepare_jptay_t21_units.py \
    --meta "$META" \
    --mqres "$MQRES" \
    --output-dir "$UNITS_DIR" \
    --samples "$SAMPLES" \
    --grid-input "$GRID_INPUT" \
    --cpg-dir "$CPG_DIR"

UNITS_CSV="${UNITS_DIR}/unit_samplesheet.csv"
NF_CSV="${UNITS_DIR}/nf_split_bam_samplesheet.csv"
EPI_COMBOS="${UNITS_DIR}/epi_combos.csv"
Z_COMBOS="${UNITS_DIR}/z_combos.csv"

if [ "$DRY_RUN" = 1 ]; then
    N_UNITS=4
    mapfile -t EPI_THRESHOLDS < <(printf '%s\n' 0.1 0.33 0.5 0.67 0.9)
else
    N_UNITS=$(tr -d '[:space:]' < "${UNITS_DIR}/n_units.txt")
    mapfile -t EPI_THRESHOLDS < "${UNITS_DIR}/epi_thresholds.txt"
fi
ARRAY_LAST=$((N_UNITS - 1))
echo "units=${N_UNITS}  epi_thresholds=${EPI_THRESHOLDS[*]}"
echo

# --- 2. query zscore grid (deconv, no BAM split) ---------------------------
echo "=== query zscore grid ==="
mkdir -p "$QUERY_Z"
Z_JID=$(submit "--array=0-${ARRAY_LAST} --job-name=jptay_zgrid" \
    run_query_zscore_grid.slurm \
    "$UNITS_CSV" "$Z_COMBOS" "$QUERY_Z" "$CPG_DIR")
echo "zscore job_id=${Z_JID}"

# --- 3. Nextflow EXTRACT_BETA per epi threshold, then episcore recall grid
EP_JIDS=()
if [ "$SKIP_NF" = 1 ]; then
    echo "=== skip Nextflow EXTRACT_BETA (--skip-nf); episcore jobs assume betas exist ==="
fi

for thr in "${EPI_THRESHOLDS[@]}"; do
    thr_g=$(awk -v t="$thr" 'BEGIN { printf "%g", t }')
    nf_out="${NF_ROOT}/thr_${thr_g}"
    ep_out="${QUERY_EP}/thr_${thr_g}"
    mkdir -p "$nf_out" "$ep_out" "${OUT_BASE}/nf_logs"

    nf_dep=""
    if [ "$SKIP_NF" != 1 ]; then
        echo "=== nf extract threshold=${thr_g} ==="
        if [ "$DRY_RUN" = 1 ]; then
            echo "[DRY-RUN] sbatch nextflow EXTRACT_BETA --threshold ${thr_g} --outdir ${nf_out}"
            nf_jid="dryrun"
        else
            nf_name=$(nf_name_tag "$thr_g")
            nf_jid=$(sbatch --parsable --job-name="jptay_nf_t${thr_g}" \
                --partition=cn-long --cpus-per-task=4 --mem=8G --time=48:00:00 \
                --output="${SCRIPT_DIR}/logs/jptay_nf_t${thr_g}_%j.log" \
                --error="${SCRIPT_DIR}/logs/jptay_nf_t${thr_g}_%j.log" \
                --wrap="mkdir -p '${nf_out}/work' && \
                    cd '${nf_out}' && \
                    '${NF}' run '${REPO}/main.nf' \
                      -profile '${PROFILE}' \
                      -name '${nf_name}' \
                      -w '${nf_out}/work' \
                      --step grid_search \
                      --input '${NF_CSV}' \
                      --outdir '${nf_out}' \
                      --threshold ${thr_g}")
            echo "$nf_jid" > "${OUT_BASE}/nf_logs/thr_${thr_g}.jobid"
        fi
        echo "nf extract job_id=${nf_jid}"
        nf_dep="--dependency=afterok:${nf_jid}"
    fi

    echo "=== query episcore grid threshold=${thr_g} ==="
    extra="--array=0-${ARRAY_LAST} --job-name=jptay_ep_t${thr_g}"
    if [ -n "$nf_dep" ]; then
        extra="${extra} ${nf_dep}"
    fi
    ep_jid=$(submit "$extra" \
        run_query_episcore_grid.slurm \
        "$UNITS_CSV" "$EPI_COMBOS" "$thr_g" "$ep_out" "$nf_out" "$CPG_DIR")
    echo "episcore job_id=${ep_jid} thr=${thr_g}"
    EP_JIDS+=("$ep_jid")
done

# --- 4. search (after zscore + all episcore arrays) ------------------------
echo "=== search chr21 combos ==="
deps="${Z_JID}"
for jid in "${EP_JIDS[@]}"; do
    deps="${deps}:${jid}"
done
SEARCH_JID=$(submit "--dependency=afterok:${deps} --job-name=jptay_t21search" \
    run_search_t21_combo.slurm \
    "$UNITS_CSV" "$UNITS_DIR" "$QUERY_EP" "$QUERY_Z" "$SEARCH_DIR")
echo "search job_id=${SEARCH_JID}"

echo
echo "Submitted. Per-batch ezscore tables land in:"
echo "  ${SEARCH_DIR}/per_batch_ezscore.tsv"
echo "  ${SEARCH_DIR}/search_summary.txt"
echo "  ${SEARCH_DIR}/best_combo_episcore.csv"
echo "  ${SEARCH_DIR}/best_combo_zscore.csv"
