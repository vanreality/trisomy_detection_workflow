#!/usr/bin/bash
#SBATCH -J 0817_idx
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o logs/%x.log
#SBATCH -e logs/%x.log

set -euo pipefail
OUTPUT_DIR=${1:?output dir}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 run_0817.py index --output-dir "$OUTPUT_DIR"
