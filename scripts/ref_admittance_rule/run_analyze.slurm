#!/usr/bin/bash
#SBATCH -J admit_analyze
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail

SCORE_DIR=${1:?score dir (…/baseline96)}
INPUT_DIR=${2:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[$(date)] analyze $SCORE_DIR"
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 analyze_perfect_vs_bad.py \
        --score-dir "$SCORE_DIR" \
        --input-dir "$INPUT_DIR"

echo "[$(date)] prove"
PROVE_EXTRA=()
if [ -n "${FORCE_RULE:-}" ]; then
    PROVE_EXTRA+=(--force-rule "$FORCE_RULE")
fi
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 derive_and_prove_rules.py \
        --score-dir "$SCORE_DIR" \
        "${PROVE_EXTRA[@]}"

echo "[$(date)] plot"
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 plot_admittance.py \
        --analysis-dir "${SCORE_DIR}/analysis" \
        --baseline-score-dir "$SCORE_DIR"

AUTO_REDRAW=${AUTO_REDRAW:-1}
PROOF="${SCORE_DIR}/analysis/proof"
if [ "$AUTO_REDRAW" = 1 ] && [ -s "${PROOF}/admitted_samples.txt" ]; then
    n_admitted=$(grep -cve '^$' "${PROOF}/admitted_samples.txt" || true)
    echo "[$(date)] admitted n=${n_admitted}"
    if [ "$n_admitted" -ge 80 ]; then
        OUT_BASE=$(dirname "$SCORE_DIR")
        echo "[$(date)] submitting prospective redraw"
        TOTAL_REPEATS=${REDRAW_REPEATS:-20000} SEED=7 OUT_BASE="$OUT_BASE" \
            bash submit_redraw.sh "$PROOF"
    else
        echo "[$(date)] skip redraw: admitted pool < 80"
    fi
fi

echo "[$(date)] done"
