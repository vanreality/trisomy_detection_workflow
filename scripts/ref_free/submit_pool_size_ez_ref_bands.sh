#!/usr/bin/bash
# Submit even-split vs LOO ez-ref mean/SD bands (10k repeats, pool 20…220 step 2).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev}
POOL_SIZES=${POOL_SIZES:-20,220,2}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}

export POOL_SIZES
SIZE_CSV=$(POOL_SIZES="$POOL_SIZES" python3 - <<'PY'
import os
text = os.environ["POOL_SIZES"]
parts = [p.strip() for p in text.split(",") if p.strip()]
if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
    lo, hi, step = map(int, parts)
    print(",".join(str(p) for p in range(lo, hi + 1, step)))
else:
    print(",".join(parts))
PY
)
IFS=',' read -r -a SIZE_ARR <<< "$SIZE_CSV"
N=${#SIZE_ARR[@]}
ARRAY_LAST=$((N - 1))
echo "OUT_BASE=$OUT_BASE pools=$SIZE_CSV n=$N repeats=$TOTAL_REPEATS"

export TOTAL_REPEATS SEED POOL_SIZES POOL_SIZE_LIST="$SIZE_CSV" PARQUET META TOXIC
job=$(sbatch --parsable --job-name=ez_ref_bands \
    --array="0-${ARRAY_LAST}" \
    --export=ALL \
    run_pool_size_ez_ref_bands.slurm "$OUT_BASE")
echo "Submitted array job_id=${job}"
plot=$(sbatch --parsable --job-name=ez_ref_plot \
    --dependency="afterok:${job}" \
    --export=ALL \
    run_plot_pool_size_ez_ref_bands.slurm "$OUT_BASE")
echo "Submitted plot job_id=${plot}"
echo "Outputs: $OUT_BASE/figures/comp{1,2}_ez_{mu,sd}.png"
