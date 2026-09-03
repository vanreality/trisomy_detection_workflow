#!/usr/bin/bash
# After score shards exist:
#   bash submit_analyze.sh /lustre1/.../20260813-ref_admittance_rule/baseline96
# Or chain after scoring:
#   bash submit_analyze.sh ... --dependency=afterok:<score_job_id>

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

SCORE_DIR=${1:?score dir}
shift || true
INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}

job=$(sbatch --parsable "$@" --job-name=admit_analyze \
    run_analyze.slurm "$SCORE_DIR" "$INPUT_DIR")
echo "Submitted analyze+prove+plot job_id=${job}"
