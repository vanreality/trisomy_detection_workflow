#!/usr/bin/env bash
# Rebuild meta + mqres samplesheets with batch mapping and QA reports.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIF="${ROOT}/containers/common_tools.sif"
OUTDIR="${OUTDIR:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary}"
SCRIPT="${ROOT}/scripts/samplesheet_summary/reorganize_samplesheet.py"

if [[ ! -f "$SIF" ]]; then
  echo "Missing singularity image: $SIF" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

singularity exec \
  -B /lustre1:/lustre1 \
  -B /appsnew:/appsnew \
  "$SIF" \
  python "$SCRIPT" \
    --outdir "$OUTDIR" \
    "$@"

echo "Done. Outputs under $OUTDIR"
