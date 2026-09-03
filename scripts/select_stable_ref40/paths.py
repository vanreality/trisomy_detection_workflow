#!/usr/bin/env python3
"""Shared paths for select_stable_ref40 notebooks / scripts."""

from pathlib import Path

# Prior search (matched recomputed early_ref baseline; PTAY1472P9S1 flipped Gray→T)
OUT_DIR_PREV = Path("/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260730-stable_ref40")
# Current search (matches meta/ref17 final_zscores; protects Gray borderlines)
OUT_DIR = Path("/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260812-stable_ref40")
META_ORIG = Path("/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet.csv")
META_REF40 = Path("/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet_ref40.csv")
EPISCORE_SS = Path("/lustre1/cqyi/syfan/nipt_article_plot/episcore_result_samplesheet.csv")
PCT_ORIG = Path("/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260607-ref_40/percentage.csv")
EZ_REF = Path("/lustre1/cqyi/myli/bert/analysis_nipt/multiomics/chr_stats_reference_samples.txt")
SCRIPT_DIR = Path("/lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/select_stable_ref40")
