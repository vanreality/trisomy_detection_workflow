#!/usr/bin/bash
# Submit loo_clean arm only (merge into existing summary.npz), then plot + detection.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev}
POOL_SIZES=${POOL_SIZES:-20,220,2}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
ARMS=${ARMS:-loo_clean}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
RUN_DETECT=${RUN_DETECT:-1}

export POOL_SIZES ARMS
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
echo "OUT_BASE=$OUT_BASE arms=$ARMS pools n=$N repeats=$TOTAL_REPEATS"

export TOTAL_REPEATS SEED POOL_SIZES POOL_SIZE_LIST="$SIZE_CSV" PARQUET META TOXIC ARMS SIF
job=$(sbatch --parsable --job-name=ez_loo_clean \
    --array="0-${ARRAY_LAST}" \
    --export=ALL \
    run_pool_size_ez_ref_bands.slurm "$OUT_BASE")
echo "Submitted array job_id=${job}"

plot=$(sbatch --parsable --job-name=ez_ref_plot \
    --dependency="afterok:${job}" \
    --export=ALL \
    run_plot_pool_size_ez_ref_bands.slurm "$OUT_BASE")
echo "Submitted plot job_id=${plot}"

if [ "$RUN_DETECT" = 1 ]; then
    det=$(sbatch --parsable --job-name=ez_fixed_det \
        --dependency="afterok:${job}" \
        --cpus-per-task=16 --mem=48G --time=04:00:00 \
        --partition=cn-long \
        -o "logs/ez_fixed_det_%j.log" -e "logs/ez_fixed_det_%j.log" \
        --wrap="cd '$SCRIPT_DIR' && singularity exec -B /lustre1,/lustre2,/appsnew '$SIF' \
            python3 compare_fixed_ez_ref.py \
                --output-dir '$OUT_BASE' \
                --parquet '$PARQUET' --meta '$META' --toxic '$TOXIC' \
                --e-arm loo_clean --e-pool-size 220 \
                --e-repeats '$TOTAL_REPEATS' --detect-repeats '$TOTAL_REPEATS' \
                --seed '$SEED' --n-jobs 16")
    echo "Submitted detection job_id=${det}"
fi
