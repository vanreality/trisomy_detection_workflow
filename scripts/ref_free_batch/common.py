"""Shared constants for Set A pool-size exploration (modeA / modeB)."""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REF_FREE_DIR = SCRIPT_DIR.parent / "ref_free"
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"

BQC_ROOT = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc"
)
MAIN_INPUT = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng"
)
DEFAULT_OUT = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_free_pool_plus_batch"
)
SIF = Path(
    "/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif"
)

META_SHEET = Path(
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv"
)
VIZ_UNITS = BQC_ROOT / "cohort" / "viz_units.csv"
EXCLUDE_LABELS = ("Unknown", "XO", "Twin", "M21")
DROP_SAMPLES = ("PTAY1351P8S1",)
PURITY_MIN = 0.8
FF_MIN = 0.01
EZ_CUTOFFS = (3.0, 4.5)
POOL_MIN = 20
POOL_MAX = 160
POOL_STEP = 2
DEFAULT_POOL = 80
DEFAULT_REPEATS = 10000
SEED = 42
FILL_SEED = 7
EP_CUTOFF = 3.0

MODES = {
    "modeA": {
        "label": "modeA",
        "ep_threshold": 0.5,
        "ep_recall": 0.65,
        "z_threshold": 0.85,
        "z_recall": 0.95,
        "bqc_dir": BQC_ROOT / "mode_A_ep0.5_0.65_z0.85_0.95",
    },
    "modeB": {
        "label": "modeB",
        "ep_threshold": 0.1,
        "ep_recall": 0.61,
        "z_threshold": 0.9,
        "z_recall": 0.92,
        "bqc_dir": BQC_ROOT / "mode_B_ep0.1_0.61_z0.9_0.92",
    },
}

SPECIAL_COLOR = "#1f77b4"
NORMAL_COLOR = "#9e9e9e"
TRISOMY_COLOR = "#d62728"


def ez_ratio_col(cutoff: float) -> str:
    return f"ezscore_signal_ratio_{cutoff:g}"


def pool_sizes(lo: int = POOL_MIN, hi: int = POOL_MAX, step: int = POOL_STEP) -> list[int]:
    return list(range(lo, hi + 1, step))
