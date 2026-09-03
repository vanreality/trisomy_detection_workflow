#!/usr/bin/bash
#SBATCH -J pool_ezstat
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%a.log
#SBATCH -e logs/%x_%a.log

set -euo pipefail
INPUT_DIR=${1:?input dir}
OUTPUT_DIR=${2:?output dir}
POOL_SIZE_LIST=${POOL_SIZE_LIST:-20,30,40,50,60,70,80,90,100,110,120,130,140,150,160}
POOL_SIZES=${POOL_SIZES:-20,160,10}
TOTAL_REPEATS=${TOTAL_REPEATS:-10000}
SEED=${SEED:-42}
FILL_SEED=${FILL_SEED:-7}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

IFS=',' read -r -a SIZE_ARR <<< "$POOL_SIZE_LIST"
POOL_SIZE=${SIZE_ARR[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 POOL_SIZES

echo "[$(date)] pool_size_ez_stats pool=${POOL_SIZE} repeats=${TOTAL_REPEATS}"

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 pool_size_ez_stats.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --pool-sizes "$POOL_SIZES" \
        --pool-size "$POOL_SIZE" \
        --total-repeats "$TOTAL_REPEATS" \
        --seed "$SEED" \
        --fill-seed "$FILL_SEED" \
        --n-jobs "$N_JOBS"

echo "[$(date)] finished pool=${POOL_SIZE}"
