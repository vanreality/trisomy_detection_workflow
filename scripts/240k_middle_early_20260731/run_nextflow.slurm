#!/usr/bin/bash
#SBATCH -J me31_nf
#SBATCH --partition=cn-long
#SBATCH --cpus-per-task=4
#SBATCH -o logs/%x.log
#SBATCH -e logs/%x.log
#SBATCH --time=72:00:00

# Usage: sbatch --job-name=me31_nf_early run_nextflow.slurm early
#         sbatch --job-name=me31_nf_middle run_nextflow.slurm middle

set -euo pipefail

cohort=${1:?Usage: run_nextflow.slurm early|middle}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT_DIR=/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260731-240k_middle_normal_samples_plus_early_allosomes_samples
OUTPUT_BASE=/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260731-240k_middle_normal_samples_plus_early_allosomes_samples
PROJECT=/lustre1/cqyi/AIPT_2.0/workflow/episcore
CPG_LIST="${PROJECT}/assets/cpgs_in_240k_probes.txt"

case "$cohort" in
    early)
        SAMPLESHEET="${INPUT_DIR}/samplesheet_nf_early.csv"
        OUTPUT_DIR="${OUTPUT_BASE}/early_nf"
        ;;
    middle)
        SAMPLESHEET="${INPUT_DIR}/samplesheet_nf_middle.csv"
        OUTPUT_DIR="${OUTPUT_BASE}/middle_nf"
        ;;
    *)
        echo "ERROR: cohort must be early|middle" >&2
        exit 2
        ;;
esac

if [ ! -f "$SAMPLESHEET" ]; then
    echo "ERROR: missing $SAMPLESHEET — run prepare_inputs.py first" >&2
    exit 1
fi
if [ ! -f "$CPG_LIST" ]; then
    echo "ERROR: missing $CPG_LIST" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
NEXTFLOW=${NEXTFLOW:-/appsnew/home/syfan/softwares/nextflow/nextflow}

# Separate workDir + project cache (launch from OUTPUT_DIR) so early|middle
# can run in parallel without sharing the repo .nextflow lock.
WORK_DIR="/lustre1/cqyi/AIPT_2.0/tmp/episcore_workflow_tmp_dir_20260731_${cohort}"
mkdir -p "$WORK_DIR"

echo "[$(date)] Nextflow cohort=${cohort}"
echo "  samplesheet : $SAMPLESHEET"
echo "  outdir      : $OUTPUT_DIR"
echo "  workDir     : $WORK_DIR"
echo "  threshold   : 0.5"
echo "  cpg_list    : $CPG_LIST"

cd "$OUTPUT_DIR"
"$NEXTFLOW" run "$PROJECT/main.nf" \
    -profile early_240k,alioth_slurm,singularity \
    --step split_bam \
    --input "$SAMPLESHEET" \
    --outdir "$OUTPUT_DIR" \
    --threshold 0.5 \
    --cpg_list "$CPG_LIST" \
    -w "$WORK_DIR" \
    -resume \
    2>&1 | tee "${OUTPUT_DIR}/nextflow_run.log"

echo "[$(date)] Nextflow finished cohort=${cohort}"
