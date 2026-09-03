#!/usr/bin/bash
# Print progress for intermediate fill jobs.
set -euo pipefail
OUTDIR=${OUTDIR:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary}
CACHE=${OUTDIR}/intermediate_cache
JOBS=${CACHE}/jobs
BQC=${BQC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc}
NF_BASE=${NF_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260814-intermediate_nf}

count_glob () {
  local pat=$1
  # intentional unquoted glob
  set +e
  local n
  n=$(ls -1 $pat 2>/dev/null | wc -l)
  set -e
  echo "$n"
}

echo "=== queue (im_/nf-) ==="
squeue -u "$USER" -o '%.18i %.12j %.2t %.10M %R' 2>/dev/null | grep -E 'im_|nf-|JOBID' | head -40 || true

echo
echo "=== outputs ==="
printf 'before_pct modeA: %s / 935\n' "$(count_glob "${CACHE}/percentage_thr0_modeA/*.percentage.tsv")"
printf 'before_pct modeB: %s / 935\n' "$(count_glob "${CACHE}/percentage_thr0_modeB/*.percentage.tsv")"
printf 'after_pct  modeA: %s / 935\n' "$(count_glob "${BQC}/mode_A_ep0.5_0.65_z0.85_0.95/scores/percentage/*.percentage.tsv")"
printf 'after_pct  modeB: %s / 935\n' "$(count_glob "${BQC}/mode_B_ep0.1_0.61_z0.9_0.92/scores/percentage/*.percentage.tsv")"
printf 'after_ep   modeA: %s / 935\n' "$(count_glob "${BQC}/mode_A_ep0.5_0.65_z0.85_0.95/scores/episcore/*.episcore.tsv")"
printf 'after_ep   modeB: %s / 935\n' "$(count_glob "${BQC}/mode_B_ep0.1_0.61_z0.9_0.92/scores/episcore/*.episcore.tsv")"
printf 'merged deconv:    %s / 72\n' "$(count_glob "${CACHE}/merged_deconv/*.parquet")"
printf 'NF beta thr0.5:   %s\n' "$(find "${NF_BASE}/nf_extract_thr0.5" -name '*_beta_value.tsv.gz' 2>/dev/null | wc -l)"
printf 'NF beta thr0.1:   %s\n' "$(find "${NF_BASE}/nf_extract_thr0.1" -name '*_beta_value.tsv.gz' 2>/dev/null | wc -l)"

echo
echo "=== array exit summary (recent jobids) ==="
for f in "${JOBS}"/*.jobid; do
  [[ -f "$f" ]] || continue
  j=$(cat "$f")
  name=$(basename "$f" .jobid)
  states=$(sacct -j "$j" --format=State -n -P 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "$name ($j): $states"
done
