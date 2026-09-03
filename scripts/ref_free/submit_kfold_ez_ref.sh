#!/usr/bin/bash
# Prepare the fixed 220-Normal pool, score even split + k-fold (2…220=LOO), then plot.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

OUT=${OUT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev/0824}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
POOL_N=${POOL_N:-220}
K_VALUES=${K_VALUES:-2-220}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

echo "Preparing fixed pool → $OUT  (seed=${SEED} pool_n=${POOL_N})"
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 kfold_ez_ref.py prepare \
        --parquet "$PARQUET" \
        --meta "$META" \
        --toxic "$TOXIC" \
        --output-dir "$OUT" \
        --pool-n "$POOL_N" \
        --total-repeats "$TOTAL_REPEATS" \
        --seed "$SEED"

export TOTAL_REPEATS SEED POOL_N K_VALUES PARQUET META TOXIC SIF
job=$(sbatch --parsable --job-name=kfold_ez_ref \
    --export=ALL \
    run_kfold_ez_ref.slurm "$OUT")
echo "Submitted score job_id=${job}"

plot=$(sbatch --parsable --job-name=kfold_ez_plot \
    --dependency="afterok:${job}" \
    --export=ALL \
    run_plot_kfold_ez_ref.slurm "$OUT")
echo "Submitted plot job_id=${plot}"
echo "Outputs: $OUT/figures/ez_{mu,sd}_by_strategy.png"
