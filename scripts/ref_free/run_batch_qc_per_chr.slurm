#!/usr/bin/bash
#SBATCH -J bqc_per_chr
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
INPUT_DIR=${1:?}
RESULT_DIR=${2:?}
OUT_TSV=${3:?}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 compute_batch_qc_per_chr_stats.py \
        --input-dir "$INPUT_DIR" \
        --result-dir "$RESULT_DIR" \
        --output-tsv "$OUT_TSV"
