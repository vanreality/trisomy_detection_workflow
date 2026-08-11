#!/usr/bin/bash
# Submit Nextflow split_bam for early (n=4) and/or middle (n=64).
#
#   ./submit_nextflow.sh early|middle|both [--dry-run]

set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
TARGET=${1:-both}
shift || true
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$WORKDIR"
mkdir -p logs

submit_one() {
    local cohort=$1
    local job_name="me31_nf_${cohort}"
    if [ "$DRY_RUN" = 1 ]; then
        echo "[DRY-RUN] sbatch --job-name=${job_name} run_nextflow.slurm ${cohort}"
        return
    fi
    jobid=$(sbatch --parsable --job-name="$job_name" run_nextflow.slurm "$cohort")
    echo "Submitted ${job_name}  job_id=${jobid}  log=logs/${job_name}.log"
}

case "$TARGET" in
    early) submit_one early ;;
    middle) submit_one middle ;;
    both)
        submit_one early
        submit_one middle
        ;;
    *)
        echo "Usage: $0 early|middle|both [--dry-run]" >&2
        exit 2
        ;;
esac
