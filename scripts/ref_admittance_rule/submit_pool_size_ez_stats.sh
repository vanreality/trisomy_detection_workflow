#!/usr/bin/bash
# Submit pool-size ez-ref mean/SD Monte-Carlo (10k repeats, pool 20…160 step 10).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/ref_admittance_check}
POOL_SIZES=${POOL_SIZES:-20,160,10}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
FILL_SEED=${FILL_SEED:-7}

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

export TOTAL_REPEATS SEED FILL_SEED POOL_SIZES POOL_SIZE_LIST="$SIZE_CSV"
job=$(sbatch --parsable --job-name=pool_ezstat \
    --array="0-${ARRAY_LAST}" \
    --export=ALL \
    run_pool_size_ez_stats.slurm "$INPUT_DIR" "$OUT_BASE")
echo "Submitted array job_id=${job}"
plot=$(sbatch --parsable --job-name=ezstat_plot \
    --dependency="afterok:${job}" \
    --export=ALL \
    run_plot_pool_ez_stats.slurm "$OUT_BASE")
echo "Submitted plot job_id=${plot}"
