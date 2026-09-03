#!/usr/bin/bash
#SBATCH -J bqc_harvest
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"

MODE=${1:?A|B}
BASE=${2:?TODAY_BASE}
UNITS=${3:?units csv}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}

if [[ "$MODE" == A ]]; then
  singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 harvest_batch_qc_episcore.py \
      --units "$UNITS" \
      --nf-outdir "$BASE/nf_extract_thr0.5" \
      --ep-dir "$BASE/mode_A_ep0.5_0.65_z0.85_0.95/scores/episcore" \
      --threshold 0.5 --recall 0.65
elif [[ "$MODE" == B ]]; then
  singularity exec -B /lustre1,/lustre2,/appsnew "$SIF" \
    python3 harvest_batch_qc_episcore.py \
      --units "$UNITS" \
      --nf-outdir "$BASE/nf_extract_thr0.1" \
      --ep-dir "$BASE/mode_B_ep0.1_0.61_z0.9_0.92/scores/episcore" \
      --threshold 0.1 --recall 0.61
else
  echo "MODE must be A or B" >&2
  exit 2
fi
