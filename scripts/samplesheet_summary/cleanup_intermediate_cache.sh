#!/usr/bin/bash
# After intermediate parquets are good: drop NF work dirs, temp units, orphan logs.
# Keeps: score TSVs, merged_deconv, percentage caches, job manifests, parquets.
#
# Usage: ./cleanup_intermediate_cache.sh [--dry-run]

set -euo pipefail
DRY=0
[[ "${1:-}" = "--dry-run" || "${1:-}" = "-n" ]] && DRY=1

OUTDIR=${OUTDIR:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary}
CACHE=${OUTDIR}/intermediate_cache
NF_BASE=${NF_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260814-intermediate_nf}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

run () {
  if [[ "$DRY" = 1 ]]; then
    echo "[DRY] $*"
  else
    eval "$@"
  fi
}

echo "Cleaning NF work dirs under $NF_BASE"
for d in "$NF_BASE"/nf_extract_thr0.5/work "$NF_BASE"/nf_extract_thr0.1/work \
         "$NF_BASE"/nf_retry_thr0.5/work "$NF_BASE"/nf_retry_thr0.1/work; do
  if [[ -d "$d" ]]; then
    run "rm -rf '$d'"
  fi
done

echo "Cleaning temp unit csvs / mktemp leftovers in cache"
run "find '$CACHE' -maxdepth 1 -type f -name '_merge_unit_*' -delete"
run "find '$CACHE' -maxdepth 1 -type f -name '_unit_*' -delete"

echo "Cleaning empty/corrupt zero-byte merge parquet if any"
run "find '$CACHE/merged_deconv' -type f -name '*.parquet' -size 0 -delete"

# Drop orphan percentage files whose unit_id is not in current units.csv
# (e.g. BAM-path date mismatches like __20250806-XML)
if [[ -f "$CACHE/units.csv" ]]; then
  echo "Pruning orphan BQC percentage/episcore files not in units.csv (optional)"
  # leave BQC scores alone by default — only document; aggressive prune needs --prune-orphans
  if [[ "${PRUNE_ORPHANS:-0}" = 1 ]]; then
    export PATH="${PY_BIN:-$HOME/softwares/miniconda3/envs/custom_bert/bin}:$PATH"
    python - <<'PY'
from pathlib import Path
import pandas as pd
units=set(pd.read_csv("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_cache/units.csv")["unit_id"].astype(str))
BQC=Path("/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc")
n=0
for sub in ["mode_A_ep0.5_0.65_z0.85_0.95","mode_B_ep0.1_0.61_z0.9_0.92"]:
  for kind,suf in [("percentage",".percentage.tsv"),("episcore",".episcore.tsv")]:
    d=BQC/sub/"scores"/kind
    if not d.is_dir():
      continue
    for p in d.glob(f"*{suf}"):
      uid=p.name[: -len(suf)]
      if uid not in units and "__MERGED" not in uid:
        p.unlink()
        n+=1
print("removed", n, "orphan score files")
PY
  fi
fi

echo "Done. Kept percentage caches, merged_deconv, jobs/, and result parquets."
