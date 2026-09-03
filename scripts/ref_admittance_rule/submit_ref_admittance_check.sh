#!/usr/bin/bash
# Independent test of admittance rules on 96 randomly selected test Normals.
#
#   cd scripts/ref_admittance_rule
#   bash submit_ref_admittance_check.sh
#
# Also submits pool-size ez mean/SD Monte-Carlo plots (20…160, 10k repeats).

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/ref_admittance_check}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
CAND_SEED=${CAND_SEED:-13}
SCORE_REPEATS=${SCORE_REPEATS:-50000}
REDRAW_REPEATS=${REDRAW_REPEATS:-20000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-25000}
SCORE_SEED=${SCORE_SEED:-42}
REDRAW_SEED=${REDRAW_SEED:-7}
SKIP_POOL_EZ=${SKIP_POOL_EZ:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$OUT_BASE"
echo "OUT_BASE=$OUT_BASE"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 select_test_ref_candidates.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUT_BASE" \
        --n 96 \
        --seed "$CAND_SEED"

CAND="$OUT_BASE/test_ref_candidates.txt"
if [ ! -s "$CAND" ]; then
    echo "ERROR: missing $CAND" >&2
    exit 1
fi

if [ "$DRY_RUN" = 1 ]; then
    echo "[DRY-RUN] candidates ready at $CAND"
    exit 0
fi

parse_jid() { awk -F= '/Submitted job_id/{print $2; exit}'; }

if [ "$SKIP_POOL_EZ" != 1 ]; then
    POOL_SIZES=20,160,10 TOTAL_REPEATS=10000 SEED=42 FILL_SEED=7 \
      OUT_BASE="$OUT_BASE" bash submit_pool_size_ez_stats.sh
fi

tmp=$(mktemp)
export INPUT_DIR
export OUT_BASE
export POOL_SOURCE=listed
export EXCLUDE_EVAL_SAMPLES="$CAND"
export TOTAL_REPEATS=$SCORE_REPEATS
export REPEATS_PER_JOB
export SEED=$SCORE_SEED
export EZ_CUTOFF=4.5
export TAG=all_96_test
export POOL_SAMPLES="$CAND"
bash submit_score_repeats.sh | tee "$tmp"
jid_score=$(parse_jid < "$tmp")
rm -f "$tmp"
if [ -z "$jid_score" ]; then
    echo "ERROR: failed to submit all_96_test scoring" >&2
    exit 2
fi

export FORCE_RULE=toxic_keep80 AUTO_REDRAW=0
analyze_job=$(sbatch --parsable --job-name=admit_test_an \
    --dependency="afterok:${jid_score}" \
    --export=ALL \
    run_analyze.slurm "${OUT_BASE}/all_96_test" "$INPUT_DIR")
echo "Submitted analyze job_id=${analyze_job}"

export REDRAW_REPEATS REPEATS_PER_JOB REDRAW_SEED
redraw_job=$(sbatch --parsable --job-name=admit_test_rd \
    --dependency="afterok:${analyze_job}" \
    --export=ALL \
    run_launch_test_redraw.slurm "$OUT_BASE" "$INPUT_DIR" "$CAND")
echo "Submitted redraw-launcher job_id=${redraw_job}"
echo "all_96_test score job_id=${jid_score}"
