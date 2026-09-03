#!/usr/bin/bash
# Prospective redraw from admitted + matched-N random pool (after prove wrote the lists).
#
#   bash submit_redraw.sh /lustre1/.../20260813-ref_admittance_rule/baseline96/analysis/proof

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

PROOF_DIR=${1:?proof dir with admitted_samples.txt}
OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule}
TOTAL_REPEATS=${TOTAL_REPEATS:-20000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-20000}
SEED=${SEED:-7}

if [ ! -s "${PROOF_DIR}/admitted_samples.txt" ]; then
    echo "ERROR: missing ${PROOF_DIR}/admitted_samples.txt" >&2
    exit 1
fi
n_admitted=$(grep -cve '^$' "${PROOF_DIR}/admitted_samples.txt" || true)
echo "admitted n=$n_admitted"
if [ "$n_admitted" -lt 80 ]; then
    echo "ERROR: admitted pool < 80; cannot redraw 40+40" >&2
    exit 2
fi

export TOTAL_REPEATS REPEATS_PER_JOB SEED OUT_BASE
parse_jid() { awk -F= '/Submitted job_id/{print $2; exit}'; }

tmp_a=$(mktemp)
tmp_r=$(mktemp)
POOL_SAMPLES="${PROOF_DIR}/admitted_samples.txt" TAG=admitted \
    bash submit_score_repeats.sh | tee "$tmp_a"
POOL_SAMPLES="${PROOF_DIR}/random_control_samples.txt" TAG=random_n \
    bash submit_score_repeats.sh | tee "$tmp_r"
jid_a=$(parse_jid < "$tmp_a")
jid_r=$(parse_jid < "$tmp_r")
rm -f "$tmp_a" "$tmp_r"
if [ -z "$jid_a" ] || [ -z "$jid_r" ]; then
    echo "ERROR: failed to parse redraw job ids" >&2
    exit 3
fi

ANALYSIS_DIR=$(dirname "$PROOF_DIR")
plot_job=$(sbatch --parsable --job-name=admit_plot_rd \
    --dependency="afterok:${jid_a}:${jid_r}" \
    run_plot_redraw.slurm \
        "$ANALYSIS_DIR" \
        "${OUT_BASE}/baseline96" \
        "${OUT_BASE}/admitted" \
        "${OUT_BASE}/random_n")
echo "Submitted redraw plot job_id=${plot_job} after ${jid_a} ${jid_r}"
