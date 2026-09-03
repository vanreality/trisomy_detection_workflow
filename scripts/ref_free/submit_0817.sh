#!/usr/bin/bash
# Prepare 0817 pools, submit 12 scoring jobs (raw/clean/ref60 × 4 modes), then rebuild INDEX.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

OUT=${OUT:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev/0817}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
CUTOFF=${CUTOFF:-3.0}
PARQUET=${PARQUET:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet}
META=${META:-/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv}
TOXIC=${TOXIC:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/toxic_samplesheet.tsv}
MAD=${MAD:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/expanded_pool_mad/candidate_mad_scores.tsv}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

echo "Preparing pools → $OUT  (seed=${SEED})"
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 run_0817.py prepare \
        --parquet "$PARQUET" \
        --meta "$META" \
        --toxic "$TOXIC" \
        --mad "$MAD" \
        --seed "$SEED" \
        --output-dir "$OUT"

export TOTAL_REPEATS SEED CUTOFF PARQUET META TOXIC SIF
job=$(sbatch --parsable --job-name=0817_ref \
    --array=0-11 \
    --export=ALL \
    run_0817.slurm "$OUT")
echo "Submitted array job_id=${job}"

idx=$(sbatch --parsable --job-name=0817_idx \
    --dependency="afterok:${job}" \
    --export=ALL \
    run_0817_index.slurm "$OUT")
echo "Submitted index job_id=${idx}"
echo "Index: $OUT/INDEX.html"
