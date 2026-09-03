#!/usr/bin/bash
#SBATCH -J admit_test_rd
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

# After all_96_test prove: submit 40+40 redraws for admitted vs random 80.
set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"

CHECK_DIR=${1:?check dir}
INPUT_DIR=${2:?input dir}
CAND=${3:?test_ref_candidates.txt}
REDRAW_REPEATS=${REDRAW_REPEATS:-20000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-20000}
REDRAW_SEED=${REDRAW_SEED:-7}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

PROOF="${CHECK_DIR}/all_96_test/analysis/proof"
if [ ! -s "${PROOF}/admitted_samples.txt" ] || [ ! -s "${PROOF}/random_control_samples.txt" ]; then
    echo "ERROR: missing admitted/random lists in $PROOF" >&2
    ls -l "$PROOF" || true
    exit 1
fi
n_adm=$(grep -cve '^$' "${PROOF}/admitted_samples.txt")
n_rnd=$(grep -cve '^$' "${PROOF}/random_control_samples.txt")
echo "admitted=$n_adm random=$n_rnd"
if [ "$n_adm" -lt 80 ] || [ "$n_rnd" -lt 80 ]; then
    echo "ERROR: need n>=80 for 40+40" >&2
    exit 2
fi

parse_jid() { awk -F= '/Submitted job_id/{print $2; exit}'; }

export INPUT_DIR OUT_BASE="$CHECK_DIR"
export POOL_SOURCE=listed EXCLUDE_EVAL_SAMPLES="$CAND"
export TOTAL_REPEATS=$REDRAW_REPEATS REPEATS_PER_JOB SEED=$REDRAW_SEED EZ_CUTOFF=4.5

tmp_a=$(mktemp)
tmp_r=$(mktemp)
POOL_SAMPLES="${PROOF}/admitted_samples.txt" TAG=toxic_16_excluded \
    bash submit_score_repeats.sh | tee "$tmp_a"
POOL_SAMPLES="${PROOF}/random_control_samples.txt" TAG=random_16_excluded \
    bash submit_score_repeats.sh | tee "$tmp_r"
jid_a=$(parse_jid < "$tmp_a")
jid_r=$(parse_jid < "$tmp_r")
rm -f "$tmp_a" "$tmp_r"
if [ -z "$jid_a" ] || [ -z "$jid_r" ]; then
    echo "ERROR: failed to submit redraw arrays" >&2
    exit 3
fi
plot=$(sbatch --parsable --job-name=admit_chk_pl \
    --dependency="afterok:${jid_a}:${jid_r}" \
    --export=ALL \
    run_plot_ref_check.slurm "$CHECK_DIR")
echo "Submitted redraw ${jid_a} ${jid_r} plot=${plot}"
