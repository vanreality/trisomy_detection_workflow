#!/usr/bin/bash
# MAD-rank drop-16, shared eval, 10k repeats for all six pools.
#
#   EZ_CUTOFF=4.5 TOTAL_REPEATS=10000 bash submit_mad_rank16_shared.sh

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

ROOT=${ROOT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule}
INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
MAD=${ROOT}/mad_rank16
OUT_BASE=${MAD}/scores
EVAL_SAMPLES=${MAD}/eval_samples.txt
EXTRA_INPUT_DIR=${MAD}/extra_eval

export EZ_CUTOFF=${EZ_CUTOFF:-4.5}
export TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
export REPEATS_PER_JOB=${REPEATS_PER_JOB:-10000}
export INPUT_DIR OUT_BASE EVAL_SAMPLES EXTRA_INPUT_DIR
export POOL_SOURCE=listed
export EXCLUDE_EVAL_SAMPLES=""

parse_jid() { awk -F= '/Submitted job_id/{print $2; exit}'; }
submit_one() {
    local tag=$1 pool=$2 seed=$3
    local tmp jid
    tmp=$(mktemp)
    POOL_SAMPLES="$pool" TAG="$tag" SEED="$seed" bash submit_score_repeats.sh | tee /dev/stderr | tee "$tmp" >/dev/null
    jid=$(parse_jid < "$tmp")
    rm -f "$tmp"
    [ -n "$jid" ] || { echo "ERROR: failed $tag" >&2; exit 3; }
    printf '%s\n' "$jid"
}

jids=()
jids+=("$(submit_one dev_all "$ROOT/baseline96/pool_samples.tsv" 42)")
jids+=("$(submit_one dev_mad16 "$MAD/dev/admitted_samples.txt" 7)")
jids+=("$(submit_one dev_random16 "$MAD/dev/random_control_samples.txt" 7)")
jids+=("$(submit_one test_all "$MAD/test/candidates.txt" 42)")
jids+=("$(submit_one test_mad16 "$MAD/test/admitted_samples.txt" 7)")
jids+=("$(submit_one test_random16 "$MAD/test/random_control_samples.txt" 7)")

dep=$(IFS=:; echo "${jids[*]}")
export DEV_ALL="$OUT_BASE/dev_all"
export DEV_MAD16="$OUT_BASE/dev_mad16"
export DEV_RANDOM16="$OUT_BASE/dev_random16"
export TEST_ALL="$OUT_BASE/test_all"
export TEST_MAD16="$OUT_BASE/test_mad16"
export TEST_RANDOM16="$OUT_BASE/test_random16"
export PLOT_OUT="$MAD/plots"
plot=$(sbatch --parsable --job-name=admit_mad16_pl \
    --dependency="afterok:${dep}" \
    --export=ALL \
    run_plot_mad_rank.slurm)
echo "Submitted ${dep} plot=${plot}"
