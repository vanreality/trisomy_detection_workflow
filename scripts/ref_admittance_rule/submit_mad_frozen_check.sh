#!/usr/bin/bash
# Score 40+40 from MAD/FF screens frozen on the original 96-dev pool.
#
#   cd scripts/ref_admittance_rule
#   EZ_CUTOFF=4.5 TOTAL_REPEATS=20000 bash submit_mad_frozen_check.sh
#
# Requires apply_frozen_mad_rule.py to have written mad_frozen/<rule>/admitted_samples.txt

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

CHECK_DIR=${CHECK_DIR:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/ref_admittance_check}
INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
CAND=${CAND:-${CHECK_DIR}/test_ref_candidates.txt}
MAD_DIR=${MAD_DIR:-${CHECK_DIR}/mad_frozen}
TOTAL_REPEATS=${TOTAL_REPEATS:-20000}
REPEATS_PER_JOB=${REPEATS_PER_JOB:-20000}
SEED=${SEED:-7}
EZ_CUTOFF=${EZ_CUTOFF:-4.5}
# Comma-separated rule names under mad_frozen/, or "viable" to read rule_summary.json
RULES=${RULES:-pct_mad_3_5,mad_keep80}

if [ ! -s "$CAND" ]; then
    echo "ERROR: missing $CAND" >&2
    exit 1
fi
if [ ! -d "$MAD_DIR" ]; then
    echo "ERROR: missing $MAD_DIR (run apply_frozen_mad_rule.py first)" >&2
    exit 1
fi

parse_jid() { awk -F= '/Submitted job_id/{print $2; exit}'; }

if [ "$RULES" = "viable" ]; then
    RULES=$(python3 - << PY
import json
from pathlib import Path
p = Path("${MAD_DIR}") / "rule_summary.json"
print(",".join(json.loads(p.read_text())["viable_rules"]))
PY
    )
fi

export INPUT_DIR OUT_BASE="$CHECK_DIR"
export POOL_SOURCE=listed EXCLUDE_EVAL_SAMPLES="$CAND"
export TOTAL_REPEATS REPEATS_PER_JOB SEED EZ_CUTOFF

jids=()
plot_tags="all_96_test,toxic_16_excluded"
for rule in ${RULES//,/ }; do
    admitted="${MAD_DIR}/${rule}/admitted_samples.txt"
    random="${MAD_DIR}/${rule}/random_control_samples.txt"
    if [ ! -s "$admitted" ]; then
        echo "skip ${rule}: no admitted list" >&2
        continue
    fi
    n=$(grep -cve '^$' "$admitted" || true)
    if [ "$n" -lt 80 ]; then
        echo "skip ${rule}: admitted=$n < 80" >&2
        continue
    fi
    case "$rule" in
        pct_mad_3_5) tag_a=mad_pct_excluded; tag_r=random_pct_excluded ;;
        mad_keep80) tag_a=mad_keep80_excluded; tag_r=random_mad_keep80_excluded ;;
        mad_or_ff) tag_a=mad_union_excluded; tag_r=random_union_excluded ;;
        intra_mad_3_5) tag_a=mad_intra_excluded; tag_r=random_intra_excluded ;;
        ff_tail_5_95) tag_a=mad_ff_excluded; tag_r=random_ff_excluded ;;
        *) tag_a="mad_${rule}_excluded"; tag_r="random_${rule}_excluded" ;;
    esac

    tmp=$(mktemp)
    POOL_SAMPLES="$admitted" TAG="$tag_a" bash submit_score_repeats.sh | tee "$tmp"
    jid=$(parse_jid < "$tmp")
    rm -f "$tmp"
    [ -n "$jid" ] || { echo "ERROR: failed to submit $tag_a" >&2; exit 3; }
    jids+=("$jid")
    plot_tags="${plot_tags},${tag_a}"

    if [ ! -s "$random" ]; then
        echo "skip random for ${rule}: no random_control_samples.txt" >&2
        continue
    fi
    tmp=$(mktemp)
    POOL_SAMPLES="$random" TAG="$tag_r" bash submit_score_repeats.sh | tee "$tmp"
    jid=$(parse_jid < "$tmp")
    rm -f "$tmp"
    [ -n "$jid" ] || { echo "ERROR: failed to submit $tag_r" >&2; exit 3; }
    jids+=("$jid")
    plot_tags="${plot_tags},${tag_r}"
done

if [ "${#jids[@]}" -eq 0 ]; then
    echo "ERROR: no MAD redraw jobs submitted" >&2
    exit 4
fi

dep=$(IFS=:; echo "${jids[*]}")
export PLOT_TAGS="$plot_tags"
export PLOT_OUT="${CHECK_DIR}/plots_mad_frozen"
plot=$(sbatch --parsable --job-name=admit_mad_pl \
    --dependency="afterok:${dep}" \
    --export=ALL \
    run_plot_ref_check.slurm "$CHECK_DIR")
echo "Submitted MAD redraw ${dep} plot=${plot} tags=${plot_tags}"
