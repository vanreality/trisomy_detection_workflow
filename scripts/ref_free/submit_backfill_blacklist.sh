#!/usr/bin/bash
# Submit SLURM array to backfill blacklist rows into fixed/pool_*/abnormality_signal_ratio.tsv
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs

INPUT_DIR=${INPUT_DIR:-/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng}
SWEEP_BASE=${SWEEP_BASE:-/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size}
TOTAL_REPEATS=${TOTAL_REPEATS:-20000}
SEED=${SEED:-42}
FILL_SEED=${FILL_SEED:-7}
BLACKLIST=${BLACKLIST:-PTAY0577P9S1,PTAY0599P8S1,PTAY0666P7S1,PTAY0682P7S1,PTAY0689P8H1}
SIF=${SIF:-/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif}
FORCE=${FORCE:-0}
ONLY_MISSING=${ONLY_MISSING:-1}

if [ -z "${POOL_SIZES:-}" ]; then
    if [ "$ONLY_MISSING" = "1" ]; then
        export SWEEP_BASE BLACKLIST
        POOL_SIZES=$(singularity exec -B /lustre1:/lustre1 "$SIF" python3 -c '
from pathlib import Path
import os
import pandas as pd
bl = {s.strip() for s in os.environ["BLACKLIST"].split(",") if s.strip()}
base = Path(os.environ["SWEEP_BASE"]) / "fixed"
miss = []
for p in sorted(base.glob("pool_*")):
    parts = p.name.split("_")
    if len(parts) != 2 or not parts[1].isdigit():
        continue
    n = int(parts[1])
    if n < 20 or n > 160 or n % 2:
        continue
    tsv = p / "abnormality_signal_ratio.tsv"
    if not tsv.is_file():
        miss.append(n)
        continue
    samples = set(pd.read_csv(tsv, sep="\t", usecols=["sample"])["sample"].astype(str))
    if not bl <= samples:
        miss.append(n)
print(",".join(str(x) for x in sorted(miss)))
')
    else
        POOL_SIZES=$(python3 -c 'print(",".join(str(p) for p in range(20, 161, 2)))')
    fi
fi

if [ -z "$POOL_SIZES" ]; then
    echo "Nothing to do: all pools already contain blacklist rows."
    exit 0
fi

IFS=',' read -r -a SIZE_ARR <<< "$POOL_SIZES"
N=${#SIZE_ARR[@]}
ARRAY_MAX=$((N - 1))

echo "INPUT_DIR     : $INPUT_DIR"
echo "SWEEP_BASE    : $SWEEP_BASE"
echo "POOL_SIZES    : $POOL_SIZES"
echo "n pools       : $N (array 0-${ARRAY_MAX})"
echo "TOTAL_REPEATS : $TOTAL_REPEATS"
echo "FORCE         : $FORCE"

export POOL_SIZES TOTAL_REPEATS SEED FILL_SEED BLACKLIST SIF FORCE
job=$(sbatch --parsable --array="0-${ARRAY_MAX}" \
    --export=ALL \
    run_backfill_blacklist.slurm "$INPUT_DIR" "$SWEEP_BASE")
echo "Submitted job $job"
echo "Logs: $SCRIPT_DIR/logs/bl_backfill_<task>.log"
