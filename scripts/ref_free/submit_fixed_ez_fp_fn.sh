#!/usr/bin/bash
# Fixed ezscore refs (ezscore_ref_samples.txt) + random 40 epi/z from remaining
# dev Normal → 10k flags → FP+FN density.
#
# Layout:
#   ${OUT_BASE}/fixed_ez_flags/flags_*.npz
#   ${OUT_BASE}/fp_fn_density_fixed_ez/fp_fn_density.html

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
OUT_BASE=${OUT_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size/fixed_ez25}
# Pin defaults so a polluted shell env cannot override (use FORCE_*=1 to allow override).
: "${TOTAL_REPEATS:=10000}"
: "${REPEATS_PER_JOB:=10000}"
: "${REF_N:=40}"
: "${SEED:=42}"
: "${FF_MIN:=0.01}"
: "${EZ_CUTOFF:=4.5}"
if [ "${FORCE_DEFAULTS:-1}" = 1 ]; then
    TOTAL_REPEATS=10000
    REPEATS_PER_JOB=10000
    EZ_CUTOFF=4.5
    REF_N=40
fi
BLACKLIST=${BLACKLIST:-PTAY0577P9S1,PTAY0599P8S1,PTAY0666P7S1,PTAY0682P7S1,PTAY0689P8H1}
EZSCORE_REF_FILE=${EZSCORE_REF_FILE:-${INPUT_DIR}/ezscore_ref_samples.txt}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
CLEAN_OLD=${CLEAN_OLD:-1}
DRY_RUN=${DRY_RUN:-0}

N_JOBS=$(( (TOTAL_REPEATS + REPEATS_PER_JOB - 1) / REPEATS_PER_JOB ))
ARRAY_LAST=$((N_JOBS - 1))
FLAG_DIR="${OUT_BASE}/fixed_ez_flags"
FPFN_DIR="${OUT_BASE}/fp_fn_density_fixed_ez"

mkdir -p "$OUT_BASE"
echo "OUT_BASE     : $OUT_BASE"
echo "ez refs      : $EZSCORE_REF_FILE"
echo "epi/z draw   : ${REF_N} from remaining dev Normal"
echo "repeats      : $TOTAL_REPEATS (array 0-${ARRAY_LAST})"
echo "ez cutoff    : $EZ_CUTOFF"
echo "blacklist    : $BLACKLIST"

if [ "$DRY_RUN" = 1 ]; then
    exit 0
fi

if [ "$CLEAN_OLD" = 1 ]; then
    rm -rf "$FLAG_DIR" "$FPFN_DIR"
fi

export TOTAL_REPEATS REPEATS_PER_JOB REF_N SEED SIF EZ_CUTOFF BLACKLIST
export EZSCORE_REF_FILE CUTOFF=3.0
export EP_THRESHOLD=0.5 EP_RECALL=0.65 Z_THRESHOLD=0.85 Z_RECALL=0.95

flag_job=$(sbatch --parsable --job-name=fixed_ez_flags \
    --array="0-${ARRAY_LAST}" \
    run_fixed_ez_flags.slurm "$INPUT_DIR" "$OUT_BASE")
echo "Submitted flags job_id=${flag_job}"

fp_job=$(sbatch --parsable --job-name=fp_fn_fixed_ez \
    --dependency="afterok:${flag_job}" \
    --wrap="cd '$SCRIPT_DIR' && singularity exec -B /lustre1,/lustre2,/appsnew '$SIF' bash -lc \
        'python3 summarize_fp_fn_density.py \
            --flag-dir \"$FLAG_DIR\" \
            --output-dir \"$FPFN_DIR\" \
            --ff-min \"$FF_MIN\" \
            --score all \
            --blacklist \"$BLACKLIST\" && \
         python3 report_fp_fn_samples.py \
            --flag-dir \"$FLAG_DIR\" \
            --output-dir \"$FPFN_DIR\" \
            --ff-min \"$FF_MIN\" \
            --score ezscore \
            --blacklist \"$BLACKLIST\"'")
echo "Submitted FP+FN dens+detail job_id=${fp_job}"
echo "HTML: $FPFN_DIR/fp_fn_density.html"
echo "Detail: $FPFN_DIR/fp_fn_sample_detail_ezscore.tsv"
