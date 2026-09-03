#!/usr/bin/env python3
"""Shared constants and helpers for the JPTAY T21 combo-recompute job."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

CHR_LIST = [f"chr{i}" for i in range(1, 23)]
CHR_INDEX = {c: i for i, c in enumerate(CHR_LIST)}

PRODUCTION_EP = (0.5, 0.65)
PRODUCTION_Z = (0.85, 0.95)
EZ_CUTOFF = 4.5
GRAY_CUTOFF = 3.0
BETA_DEPTH = 30

ROOT = Path("/lustre1/cqyi/AIPT_2.0/workflow/episcore")
DEFAULT_META = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv")
DEFAULT_MQRES = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/mqres_samplesheet.csv")
# Main ref_40 grid parquets plus 4 ez-ref Normals merged at production combo only
# (see scripts/ref_free/prepare_fixed_ez25_assets.py).
DEFAULT_GRID_INPUT = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size/"
    "fixed_ez25/input_with_missing4"
)
DEFAULT_GRID_INPUT_BASE = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng"
)
DEFAULT_EARLY_REF_MATRIX = ROOT / "assets/early_reference_beta_zscore.tsv"
DEFAULT_EZ_REF = Path(
    "/lustre1/cqyi/myli/bert/analysis_nipt/multiomics/chr_stats_reference_samples.txt"
)
DEFAULT_CPG_DIR = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260525-grid_search_240k_panel_240k_model/recall_list_220k"
)
DEFAULT_SIF = ROOT / "containers/common_tools.sif"
DEFAULT_SAMPLES = ("JPTAY1835P7H1", "JPTAY1927P8H1", "JPTAY1964P9H1")
EXTRA_MQRES_DEFAULT = (
    Path("/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260807-jptay_multibatch_check/mqres.csv"),
    Path("/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260811-jptay1927_1964_check/mqres.csv"),
)

_DATE_RE = re.compile(r"(\d{8})")
_SAMPLE_BATCH_RE = re.compile(r"^(?P<sample>.+)_(?P<batch>\d{8})$")


def fmt_combo(value: float) -> str:
    return f"{float(value):g}"


def yyyymmdd(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    m = _DATE_RE.search(s)
    return m.group(1) if m else None


def batch_key(val) -> str:
    d = yyyymmdd(val) or str(val).strip()
    return f"{d}-XML" if not str(d).endswith("-XML") else d


def norm_hcpt(sample: str) -> str:
    s = str(sample)
    if s.startswith("HCPT") and len(s) > 8:
        return s[:8]
    return s


def load_sample_list(path: Path) -> list[str]:
    samples: list[str] = []
    with path.open() as handle:
        for line in handle:
            s = line.strip()
            if s and not s.startswith("#"):
                samples.append(s)
    if not samples:
        raise ValueError(f"No samples in {path}")
    return samples


def early_ref_from_matrix(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t", usecols=["sample"])
    return df["sample"].astype(str).tolist()


def resolve_names(wanted: Sequence[str], available: Iterable[str]) -> tuple[list[str], list[str]]:
    """Map requested IDs onto names present in ``available`` (HCPT truncation)."""
    avail = [str(s) for s in available]
    by_exact = {s: s for s in avail}
    by_norm: dict[str, str] = {}
    for s in avail:
        by_norm.setdefault(norm_hcpt(s), s)
        if s.startswith("HCPT") and s.endswith("P") and len(s) > 8:
            by_norm.setdefault(s[:8], s)
    resolved: list[str] = []
    missing: list[str] = []
    for name in wanted:
        key = str(name)
        hit = by_exact.get(key) or by_norm.get(norm_hcpt(key))
        if hit is None:
            missing.append(key)
        else:
            resolved.append(hit)
    return resolved, missing


def parse_sample_batch(sample: str, mqres_batch=None) -> tuple[str, str]:
    """Return (sample, yyyymmdd batch) from either encoded sample or mqres_batch."""
    raw = str(sample)
    m = _SAMPLE_BATCH_RE.match(raw)
    if m is not None and m.group("sample").startswith(("JPTAY", "PTAY", "HCPT")):
        return m.group("sample"), m.group("batch")
    d = yyyymmdd(mqres_batch)
    if d is None:
        raise ValueError(f"Cannot parse batch for sample={sample!r} mqres_batch={mqres_batch!r}")
    return raw, d


def pick_deconv_row(group: pd.DataFrame) -> pd.Series:
    g = group.copy()
    g["_se"] = g["deconv_res"].astype(str).str.contains("single_end", case=False, na=False)
    if (~g["_se"]).any():
        g = g[~g["_se"]]
    g["_prio"] = g["deconv_res"].map(lambda p: 0 if str(p).endswith(".parquet") else 1)
    return g.sort_values("_prio").iloc[0]


def bam_root(clean_bam: str) -> Optional[Path]:
    p = Path(str(clean_bam))
    if "bwameth_results" not in p.parts:
        return None
    i = p.parts.index("bwameth_results")
    return Path(*p.parts[: i + 1])


def find_production_beta(sample: str, clean_bam: str) -> Optional[Path]:
    root = bam_root(clean_bam)
    if root is None:
        return None
    p = (
        root
        / "zscore_downstream"
        / "beta_zscore"
        / sample
        / "extract_beta_value"
        / f"{sample}_beta_value.tsv.gz"
    )
    return p if p.is_file() else None


def find_nf_beta(nf_outdir: Path, unit_id: str) -> Optional[Path]:
    direct = nf_outdir / "extract_beta_value" / f"{unit_id}_beta_value.tsv.gz"
    if direct.is_file():
        return direct
    hits = sorted(nf_outdir.glob(f"**/{unit_id}_beta_value.tsv.gz"))
    return hits[0] if hits else None


def pred_label_from_ez(ez: np.ndarray, chr_list: Sequence[str] = CHR_LIST) -> str:
    t_labels, gray_labels = [], []
    for chrom, z in zip(chr_list, ez):
        if not np.isfinite(z):
            continue
        num = str(chrom).removeprefix("chr")
        if z > EZ_CUTOFF:
            t_labels.append(f"T{num}")
        elif GRAY_CUTOFF <= z <= EZ_CUTOFF:
            gray_labels.append(f"Gray_T{num}")
    parts = t_labels + gray_labels
    return ",".join(parts) if parts else "Normal"


def maybe_fraction(values: np.ndarray) -> np.ndarray:
    """Convert percent-scale chromosome shares to fractions when needed."""
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmedian(np.abs(finite)) > 1.5:
        return values / 100.0
    return values


def write_combo_csv(path: Path, combos: Sequence[tuple[float, float]]) -> None:
    pd.DataFrame(combos, columns=["threshold", "recall"]).to_csv(path, index=False)


def read_combo_csv(path: Path) -> list[tuple[float, float]]:
    df = pd.read_csv(path)
    return [(float(t), float(r)) for t, r in zip(df["threshold"], df["recall"])]
