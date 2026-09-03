#!/usr/bin/env python3
"""Build intermediate wide matrices for fixed_ref / ref_free (modeA + modeB).

Outputs under ``--outdir`` (default samplesheet_summary):
  - intermediate_each_batch_mode{A,B}.parquet
  - intermediate_merged_batches_mode{A,B}.parquet
  - intermediate_matrices_report.md

Sources (prefer reuse):
  - Batch-QC scores under 20260811-ref_free_batch_qc mode_A / mode_B
  - Production ``*_ff.tsv`` / wide ``*_zscore.tsv`` when present
  - thr=0 percentage on demand (before_mq), cached
  - Multi-batch: merge_deconv + %; episcore from latest batch as placeholder
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ref_free"))
sys.path.insert(0, str(ROOT / "bin"))

DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")
DEFAULT_MQRES = DEFAULT_OUTDIR / "mqres_samplesheet.csv"
DEFAULT_META = DEFAULT_OUTDIR / "meta_samplesheet.csv"
DEFAULT_BQC = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc"
)
DEFAULT_CPG_DIR = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260525-grid_search_240k_panel_240k_model/recall_list_220k"
)
SIF = ROOT / "containers" / "common_tools.sif"
MERGE_BIN = ROOT / "bin" / "merge_deconv_res_full.py"
CHR_LIST = [f"chr{i}" for i in range(1, 23)]
BEFORE_THR = 0.0
_DATE_RE = re.compile(r"(\d{8})")
_BATCH_KEY_RE = re.compile(r"(\d{8}-XML)")

# Mode pins from ref_free_batch_qc.ipynb
MODES = {
    "A": {
        "label": "modeA",
        "scores": DEFAULT_BQC / "mode_A_ep0.5_0.65_z0.85_0.95" / "scores",
        "ep_thr": 0.5,
        "ep_recall": 0.65,
        "pct_thr": 0.85,
        "pct_recall": 0.95,
    },
    "B": {
        "label": "modeB",
        "scores": DEFAULT_BQC / "mode_B_ep0.1_0.61_z0.9_0.92" / "scores",
        "ep_thr": 0.1,
        "ep_recall": 0.61,
        "pct_thr": 0.9,
        "pct_recall": 0.92,
    },
}


def yyyymmdd(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    m = _DATE_RE.search(s)
    return m.group(1) if m else None


def bam_root(clean_bam: str) -> Optional[Path]:
    p = Path(str(clean_bam))
    if "bwameth_results" not in p.parts:
        return None
    i = p.parts.index("bwameth_results")
    return Path(*p.parts[: i + 1])


def batch_key_from_mqres(mqres_batch: str) -> str:
    """Always key units by mqres_batch (not BAM path date).

    BAM paths can embed a different YYYYMMDD (e.g. ``20250806-XML.20250808-XML``);
    BQC score files are named ``{sample}__{mqres_batch}-XML``.
    """
    d = yyyymmdd(mqres_batch) or str(mqres_batch)
    return f"{d}-XML"


def batch_key_from_bam(clean_bam: str, mqres_batch: str) -> str:
    """Legacy helper: prefer BAM path date, else mqres_batch."""
    m = _BATCH_KEY_RE.search(str(clean_bam))
    if m:
        return m.group(1)
    return batch_key_from_mqres(mqres_batch)


def build_units(mqres: pd.DataFrame) -> pd.DataFrame:
    """One unit per (sample, mqres_batch); prefer non-single_end deconv."""
    df = mqres.copy()
    df["mqres_batch"] = df["mqres_batch"].map(yyyymmdd)
    df["is_se"] = df["deconv_res"].astype(str).str.contains("single_end", case=False, na=False)
    rows = []
    for (sample, batch), g in df.groupby(["sample", "mqres_batch"], sort=False):
        g2 = g[~g["is_se"]] if (~g["is_se"]).any() else g
        # prefer parquet deconv
        g2 = g2.copy()
        g2["_prio"] = g2["deconv_res"].map(
            lambda p: 0 if str(p).endswith(".parquet") else 1
        )
        r = g2.sort_values("_prio").iloc[0]
        bk = batch_key_from_mqres(batch)
        root = bam_root(r["clean_bam"])
        rows.append(
            {
                "sample": sample,
                "batch": batch,
                "batch_key": bk,
                "unit_id": f"{sample}__{bk}",
                "clean_bam": r["clean_bam"],
                "deconv_res": r["deconv_res"],
                "bam_root": str(root) if root else None,
            }
        )
    return pd.DataFrame(rows)


def find_ff(sample: str, root: Optional[Path]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if root is None:
        return None, None, None
    for sub in ("estimate_ff_higher_precision", "snp_to_ff"):
        p = Path(root) / "zscore_downstream" / "beta_zscore" / sample / sub / f"{sample}_ff.tsv"
        if not p.is_file():
            continue
        try:
            ff = pd.read_csv(p, sep="\t")
            if "ff_before_mq" not in ff.columns:
                continue
            if "chr" in ff.columns and (ff["chr"].astype(str) == "all").any():
                row = ff.loc[ff["chr"].astype(str) == "all"].iloc[0]
            else:
                row = ff.iloc[0]
            return (
                float(row["ff_before_mq"]),
                float(row["ff_after_mq"]) if "ff_after_mq" in ff.columns else None,
                str(p),
            )
        except Exception:
            continue
    return None, None, None


def find_ep_wide(sample: str, root: Optional[Path]) -> Optional[Path]:
    if root is None:
        return None
    for sub in ("beta_to_episcore", "beta_to_zscore"):
        p = Path(root) / "zscore_downstream" / "beta_zscore" / sample / sub / f"{sample}_zscore.tsv"
        if p.is_file():
            return p
    return None


def empty_chr_block(prefix_suffix: str) -> dict:
    """Build NaN dict for all chr feature columns of one kind."""
    out = {}
    for chr_i in range(1, 23):
        out[f"chr{chr_i}_{prefix_suffix}"] = np.nan
    return out


def pct_tsv_to_wide(path: Path, suffix: str) -> dict:
    """``{unit}.percentage.tsv`` → ``chr#_percentage_{suffix}``."""
    out = empty_chr_block(f"percentage_{suffix}")
    if not path.is_file():
        return out
    df = pd.read_csv(path, sep="\t")
    for _, r in df.iterrows():
        chr_name = str(r["chr"])
        if not chr_name.startswith("chr"):
            chr_name = f"chr{chr_name}"
        key = f"{chr_name}_percentage_{suffix}"
        if key in out:
            out[key] = float(r["percentage"])
    return out


def ep_tsv_to_wide(path: Path, suffix: str) -> dict:
    """``{unit}.episcore.tsv`` → hypo/hyper z_intra + cpg_count columns."""
    out = {}
    for kind in ("hypo_z_intra", "hyper_z_intra", "hypo_cpg_count", "hyper_cpg_count"):
        out.update(empty_chr_block(f"{kind}_{suffix}"))
    if not path.is_file():
        return out
    df = pd.read_csv(path, sep="\t")
    for _, r in df.iterrows():
        chr_name = str(r["chr"])
        if not chr_name.startswith("chr"):
            chr_name = f"chr{chr_name}"
        num = chr_name.replace("chr", "")
        mapping = {
            f"chr{num}_hypo_z_intra_{suffix}": "hypo_z_intra",
            f"chr{num}_hyper_z_intra_{suffix}": "hyper_z_intra",
            f"chr{num}_hypo_cpg_count_{suffix}": "hypo_cpgs_count",
            f"chr{num}_hyper_cpg_count_{suffix}": "hyper_cpgs_count",
        }
        for dest, src in mapping.items():
            if dest in out and src in r:
                out[dest] = float(r[src])
    return out


def ep_wide_tsv_to_wide(path: Path, suffix: str) -> dict:
    """Production wide ``*_zscore.tsv`` → hypo/hyper columns."""
    out = {}
    for kind in ("hypo_z_intra", "hyper_z_intra", "hypo_cpg_count", "hyper_cpg_count"):
        out.update(empty_chr_block(f"{kind}_{suffix}"))
    if not path.is_file():
        return out
    row = pd.read_csv(path, sep="\t").iloc[0]
    for i in range(1, 23):
        for kind, src in (
            ("hypo_z_intra", f"chr{i}_hypo_z_intra"),
            ("hyper_z_intra", f"chr{i}_hyper_z_intra"),
            ("hypo_cpg_count", f"chr{i}_hypo_cpgs_count"),
            ("hyper_cpg_count", f"chr{i}_hyper_cpgs_count"),
        ):
            dest = f"chr{i}_{kind}_{suffix}"
            if src in row and pd.notna(row[src]):
                out[dest] = float(row[src])
    return out


def compute_percentage(
    unit_id: str,
    deconv_res: str,
    out_path: Path,
    threshold: float,
    recall: float,
    cpg_dir: Path,
    force: bool = False,
) -> bool:
    if out_path.is_file() and not force:
        return True
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ref_free" / "compute_unit_percentages.py"),
        "--units",
        "/dev/stdin",  # replaced below — use temp units csv instead
    ]
    # Write one-row units csv next to output
    units_csv = out_path.parent / f"_unit_{unit_id}.csv"
    pd.DataFrame(
        [{"unit_id": unit_id, "deconv_res": deconv_res}]
    ).to_csv(units_csv, index=False)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ref_free" / "compute_unit_percentages.py"),
        "--units",
        str(units_csv),
        "--output-dir",
        str(out_path.parent),
        "--threshold",
        str(threshold),
        "--recall",
        str(recall),
        "--cpg-recall-dir",
        str(cpg_dir),
        "--unit-id",
        unit_id,
    ]
    if force:
        cmd.append("--force")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[pct fail] {unit_id}: {e.stderr[-500:] if e.stderr else e}")
        return False
    finally:
        if units_csv.is_file():
            units_csv.unlink(missing_ok=True)
    return out_path.is_file()


def merge_deconvs(sample: str, deconv_paths: list[str], out_parquet: Path) -> bool:
    if out_parquet.is_file():
        return True
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(MERGE_BIN),
        "--inputs",
        " ".join(str(p) for p in deconv_paths),
        "--output",
        str(out_parquet),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_parquet.is_file()
    except subprocess.CalledProcessError as e:
        print(f"[merge fail] {sample}: {e.stderr[-500:] if e.stderr else e}")
        return False


def feature_column_order() -> list[str]:
    cols = ["sample", "ff_before_mq", "ff_after_mq", "batch"]
    for i in range(1, 23):
        cols.append(f"chr{i}_percentage_before_mq")
    for i in range(1, 23):
        cols.append(f"chr{i}_percentage_after_mq")
    for kind in ("hypo_z_intra", "hyper_z_intra"):
        for when in ("before_mq", "after_mq"):
            for i in range(1, 23):
                cols.append(f"chr{i}_{kind}_{when}")
    for kind in ("hypo_cpg_count", "hyper_cpg_count"):
        for when in ("before_mq", "after_mq"):
            for i in range(1, 23):
                cols.append(f"chr{i}_{kind}_{when}")
    return cols


def assemble_each_batch(
    units: pd.DataFrame,
    meta: pd.DataFrame,
    mode_cfg: dict,
    cache_dir: Path,
    cpg_dir: Path,
    compute_missing_before_pct: bool,
) -> tuple[pd.DataFrame, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode_label = mode_cfg["label"]
    scores = Path(mode_cfg["scores"])
    pct_recall = float(mode_cfg["pct_recall"])
    before_pct_dir = cache_dir / f"percentage_thr0_{mode_label}"
    before_pct_dir.mkdir(parents=True, exist_ok=True)
    meta_ff = meta.set_index("sample")[["ff_before_mq", "ff_after_mq"]].to_dict("index")

    stats = {
        "mode": mode_label,
        "n_units": len(units),
        "after_pct_reuse": 0,
        "after_ep_reuse": 0,
        "before_pct_reuse_or_compute": 0,
        "before_ep_from_production": 0,
        "ff_from_batch_tsv": 0,
        "ff_from_meta": 0,
        "missing_after_pct": 0,
        "missing_after_ep": 0,
        "missing_before_pct": 0,
        "missing_before_ep": 0,
    }
    rows = []
    for r in units.itertuples(index=False):
        uid = r.unit_id
        sample = r.sample
        root = Path(r.bam_root) if r.bam_root else None

        # FF
        ff_b, ff_a, ff_src = find_ff(sample, root)
        if ff_b is None and sample in meta_ff:
            ff_b = meta_ff[sample].get("ff_before_mq")
            ff_a = meta_ff[sample].get("ff_after_mq")
            stats["ff_from_meta"] += 1
        elif ff_b is not None:
            stats["ff_from_batch_tsv"] += 1

        # after_mq percentage / episcore from mode scores/
        after_pct_path = scores / "percentage" / f"{uid}.percentage.tsv"
        after_ep_path = scores / "episcore" / f"{uid}.episcore.tsv"
        after_pct = pct_tsv_to_wide(after_pct_path, "after_mq")
        after_ep = ep_tsv_to_wide(after_ep_path, "after_mq")
        if after_pct_path.is_file():
            stats["after_pct_reuse"] += 1
        else:
            stats["missing_after_pct"] += 1
        if after_ep_path.is_file():
            stats["after_ep_reuse"] += 1
        else:
            stats["missing_after_ep"] += 1

        # before_mq percentage at thr=0 (panel recall = mode pct_recall)
        before_pct_path = before_pct_dir / f"{uid}.percentage.tsv"
        if before_pct_path.is_file():
            before_pct = pct_tsv_to_wide(before_pct_path, "before_mq")
            stats["before_pct_reuse_or_compute"] += 1
        elif compute_missing_before_pct and Path(r.deconv_res).is_file():
            ok = compute_percentage(
                uid, r.deconv_res, before_pct_path, BEFORE_THR, pct_recall, cpg_dir
            )
            before_pct = pct_tsv_to_wide(before_pct_path, "before_mq")
            if ok:
                stats["before_pct_reuse_or_compute"] += 1
            else:
                stats["missing_before_pct"] += 1
        else:
            before_pct = empty_chr_block("percentage_before_mq")
            stats["missing_before_pct"] += 1

        # before_mq episcore: production wide (best available proxy on disk)
        ep_wide = find_ep_wide(sample, root)
        if ep_wide is not None:
            before_ep = ep_wide_tsv_to_wide(ep_wide, "before_mq")
            stats["before_ep_from_production"] += 1
        else:
            before_ep = {}
            for kind in ("hypo_z_intra", "hyper_z_intra", "hypo_cpg_count", "hyper_cpg_count"):
                before_ep.update(empty_chr_block(f"{kind}_before_mq"))
            stats["missing_before_ep"] += 1

        row = {
            "sample": sample,
            "ff_before_mq": ff_b,
            "ff_after_mq": ff_a,
            "batch": r.batch,
            "batch_key": r.batch_key,
            "unit_id": uid,
            **before_pct,
            **after_pct,
            **before_ep,
            **after_ep,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    # enforce column order (+ keep helpers at end for debugging then drop)
    ordered = feature_column_order()
    extras = [c for c in ("batch_key", "unit_id") if c in out.columns]
    out = out[ordered + extras]
    return out, stats


def assemble_merged(
    each: pd.DataFrame,
    units: pd.DataFrame,
    meta: pd.DataFrame,
    mode_cfg: dict,
    cache_dir: Path,
    cpg_dir: Path,
    compute_merged: bool,
) -> tuple[pd.DataFrame, dict]:
    """One row per sample; multi-batch merges deconv for percentages."""
    mode_label = mode_cfg["label"]
    pct_thr = float(mode_cfg["pct_thr"])
    pct_recall = float(mode_cfg["pct_recall"])
    merge_dir = cache_dir / "merged_deconv"
    merge_pct_before = cache_dir / f"merged_percentage_thr0_{mode_label}"
    merge_pct_after = cache_dir / f"merged_percentage_thr{pct_thr}_{mode_label}"
    for d in (merge_dir, merge_pct_before, merge_pct_after):
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        "mode": mode_label,
        "n_samples": 0,
        "n_single_batch_copy": 0,
        "n_multi_batch": 0,
        "n_multi_pct_ok": 0,
        "n_multi_ep_from_latest_batch": 0,
        "n_multi_needs_ep_recompute": 0,
    }
    meta_by = meta.drop_duplicates("sample").set_index("sample")
    each_by_unit = each.set_index("unit_id") if "unit_id" in each.columns else None

    rows = []
    for sample, ug in units.groupby("sample", sort=False):
        stats["n_samples"] += 1
        batches = sorted(ug["batch"].astype(str).unique())
        batch_str = ",".join(batches)

        if sample in meta_by.index:
            ff_b = meta_by.at[sample, "ff_before_mq"]
            ff_a = meta_by.at[sample, "ff_after_mq"]
        else:
            ff_b = ff_a = np.nan

        if len(batches) == 1:
            stats["n_single_batch_copy"] += 1
            uid = ug.iloc[0]["unit_id"]
            src = each[each["unit_id"] == uid].iloc[0].to_dict() if "unit_id" in each.columns else each[each["sample"] == sample].iloc[0].to_dict()
            row = {c: src.get(c) for c in feature_column_order()}
            row["sample"] = sample
            row["batch"] = batch_str
            row["ff_before_mq"] = ff_b if pd.notna(ff_b) else src.get("ff_before_mq")
            row["ff_after_mq"] = ff_a if pd.notna(ff_a) else src.get("ff_after_mq")
            rows.append(row)
            continue

        stats["n_multi_batch"] += 1
        row = {
            "sample": sample,
            "ff_before_mq": ff_b,
            "ff_after_mq": ff_a,
            "batch": batch_str,
        }
        row.update(empty_chr_block("percentage_before_mq"))
        row.update(empty_chr_block("percentage_after_mq"))
        for kind in ("hypo_z_intra", "hyper_z_intra", "hypo_cpg_count", "hyper_cpg_count"):
            for when in ("before_mq", "after_mq"):
                row.update(empty_chr_block(f"{kind}_{when}"))

        # percentages from merged deconv (reuse cache even when not computing)
        deconv_paths = []
        for _, ur in ug.iterrows():
            p = Path(ur["deconv_res"])
            if p.is_file():
                deconv_paths.append(str(p))
        merged_path = merge_dir / f"{sample}.merged.deconv.parquet"
        uid = f"{sample}__MERGED"
        before_p = merge_pct_before / f"{uid}.percentage.tsv"
        after_p = merge_pct_after / f"{uid}.percentage.tsv"

        if before_p.is_file():
            row.update(pct_tsv_to_wide(before_p, "before_mq"))
        if after_p.is_file():
            row.update(pct_tsv_to_wide(after_p, "after_mq"))

        if compute_merged and deconv_paths:
            if merge_deconvs(sample, deconv_paths, merged_path):
                ok1 = compute_percentage(
                    uid, str(merged_path), before_p, BEFORE_THR, pct_recall, cpg_dir
                )
                ok2 = compute_percentage(
                    uid, str(merged_path), after_p, pct_thr, pct_recall, cpg_dir
                )
                if ok1:
                    row.update(pct_tsv_to_wide(before_p, "before_mq"))
                if ok2:
                    row.update(pct_tsv_to_wide(after_p, "after_mq"))

        if before_p.is_file() and after_p.is_file():
            stats["n_multi_pct_ok"] += 1

        # episcore + % fallback: use latest batch's each_batch features when
        # true merged compute is unavailable / incomplete
        latest = ug.sort_values("batch").iloc[-1]["unit_id"]
        src_rows = each[each["unit_id"] == latest] if "unit_id" in each.columns else each.iloc[0:0]
        if len(src_rows):
            src = src_rows.iloc[0]
            for c in feature_column_order():
                if not c.startswith("chr"):
                    continue
                # keep merged % if already filled
                if "percentage" in c and pd.notna(row.get(c)):
                    continue
                row[c] = src.get(c)
            stats["n_multi_ep_from_latest_batch"] += 1
            stats["n_multi_needs_ep_recompute"] += 1

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[feature_column_order()]
    return out, stats



def write_report(
    path: Path,
    units: pd.DataFrame,
    results: dict,
) -> None:
    def cov(df: pd.DataFrame, col_suffix: str) -> float:
        cands = [c for c in df.columns if c.startswith("chr1_") and c.endswith(col_suffix)]
        if not cands:
            return float("nan")
        return float(df[cands[0]].notna().mean())

    lines = [
        "# Intermediate matrices (modeA + modeB)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Built from current `mqres_samplesheet.csv` (noisy batches already excluded).",
        "",
        "## Mode pins",
        "",
        "| Mode | episcore thr/recall | percentage thr/recall | scores dir |",
        "|------|---------------------|----------------------|------------|",
    ]
    for key, cfg in MODES.items():
        lines.append(
            f"| {cfg['label']} | {cfg['ep_thr']} / {cfg['ep_recall']} | "
            f"{cfg['pct_thr']} / {cfg['pct_recall']} | `{cfg['scores']}` |"
        )
    lines += [
        "",
        "before_mq percentage uses thr=`0.0` with the mode's percentage recall.",
        "before_mq z/cpg uses production wide `*_zscore.tsv` when present (proxy).",
        "",
        "## Outputs",
        "",
        "| File | Rows |",
        "|------|-----:|",
    ]
    for mode, pack in results.items():
        lines.append(
            f"| `intermediate_each_batch_{mode}.parquet` | {len(pack['each'])} |"
        )
        lines.append(
            f"| `intermediate_merged_batches_{mode}.parquet` | {len(pack['merged'])} |"
        )
    lines += ["", "## Per-mode assembly stats", ""]
    for mode, pack in results.items():
        es, ms = pack["each_stats"], pack["merge_stats"]
        each, merged = pack["each"], pack["merged"]
        lines += [
            f"### {mode}",
            "",
            f"- after_mq % / ep reused: **{es['after_pct_reuse']}** / **{es['after_ep_reuse']}** "
            f"(missing {es['missing_after_pct']} / {es['missing_after_ep']})",
            f"- before_mq % available: **{es['before_pct_reuse_or_compute']}** "
            f"(missing {es['missing_before_pct']})",
            f"- before_mq ep from production: **{es['before_ep_from_production']}** "
            f"(missing {es['missing_before_ep']})",
            f"- FF from batch tsv / meta: **{es['ff_from_batch_tsv']}** / **{es['ff_from_meta']}**",
            f"- merged: single-copy **{ms['n_single_batch_copy']}**, multi **{ms['n_multi_batch']}**, "
            f"multi % ok **{ms['n_multi_pct_ok']}**, ep from latest batch **{ms['n_multi_ep_from_latest_batch']}**",
            "",
            "| Matrix | % before | % after | hypo_z before | hypo_z after |",
            "|--------|---------:|--------:|--------------:|-------------:|",
            f"| each | {100*cov(each,'percentage_before_mq'):.1f}% | "
            f"{100*cov(each,'percentage_after_mq'):.1f}% | "
            f"{100*cov(each,'hypo_z_intra_before_mq'):.1f}% | "
            f"{100*cov(each,'hypo_z_intra_after_mq'):.1f}% |",
            f"| merged | {100*cov(merged,'percentage_before_mq'):.1f}% | "
            f"{100*cov(merged,'percentage_after_mq'):.1f}% | "
            f"{100*cov(merged,'hypo_z_intra_before_mq'):.1f}% | "
            f"{100*cov(merged,'hypo_z_intra_after_mq'):.1f}% |",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "1. Units rebuilt from post-review mqres (noisy multi-batch rows removed).",
        "2. Multi-batch merged rows: when true deconv-merge % is not computed, "
        "**percentage + episcore/cpg** are filled from the **latest remaining batch** "
        "(placeholder). Re-run with `--compute-merged` and/or NF merge→beta→episcore for gold.",
        "3. Gaps in after_mq scores: backfill with `compute_unit_percentages` / "
        "`compute_unit_episcore` / Mode B harvest under the BQC tree.",
        "4. before_mq % is empty until `--compute-before-pct` (thr=0) is run; "
        "cache lands in `intermediate_cache/percentage_thr0_{modeA,modeB}/`.",
        "5. Rebuild: `python scripts/samplesheet_summary/build_intermediate_matrices.py "
        "[--skip-compute|--compute-before-pct|--compute-merged]`.",
        "",
        "## Column schema",
        "",
        "- `sample`, `ff_before_mq`, `ff_after_mq`, `batch`",
        "- `chr{1-22}_percentage_{before,after}_mq`",
        "- `chr{1-22}_{hypo,hyper}_z_intra_{before,after}_mq`",
        "- `chr{1-22}_{hypo,hyper}_cpg_count_{before,after}_mq`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--mqres", type=Path, default=DEFAULT_MQRES)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--cpg-dir", type=Path, default=DEFAULT_CPG_DIR)
    p.add_argument("--modes", default="A,B", help="Comma-separated modes, e.g. A,B")
    p.add_argument(
        "--compute-before-pct",
        action="store_true",
        help="Compute missing thr=0 percentages for each_batch (slow).",
    )
    p.add_argument(
        "--compute-merged",
        action="store_true",
        help="Merge multi-batch deconvs and compute merged percentages.",
    )
    p.add_argument(
        "--skip-compute",
        action="store_true",
        help="Assemble only from existing on-disk scores/caches.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / "intermediate_cache"

    mqres = pd.read_csv(args.mqres)
    meta = pd.read_csv(args.meta)
    units = build_units(mqres)
    cache_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(cache_dir / "units.csv", index=False)
    print(f"units={len(units)}")

    compute_before = args.compute_before_pct and not args.skip_compute
    compute_merged = args.compute_merged and not args.skip_compute
    if not args.skip_compute and not args.compute_before_pct and not args.compute_merged:
        # default: assemble-only (reuse scores); heavy thr0/merge opt-in
        compute_before = False
        compute_merged = False

    mode_keys = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    results = {}
    all_stats = {}
    for key in mode_keys:
        if key not in MODES:
            raise SystemExit(f"Unknown mode {key}; choose from {list(MODES)}")
        cfg = MODES[key]
        label = cfg["label"]
        print(f"=== {label} scores={cfg['scores']} ===")
        each, each_stats = assemble_each_batch(
            units,
            meta,
            cfg,
            cache_dir,
            args.cpg_dir,
            compute_missing_before_pct=compute_before,
        )
        each_out = each[feature_column_order()].copy()
        each_path = outdir / f"intermediate_each_batch_{label}.parquet"
        each_out.to_parquet(each_path, index=False)

        merged, merge_stats = assemble_merged(
            each,
            units,
            meta,
            cfg,
            cache_dir,
            args.cpg_dir,
            compute_merged=compute_merged,
        )
        merged_path = outdir / f"intermediate_merged_batches_{label}.parquet"
        merged.to_parquet(merged_path, index=False)

        results[label] = {
            "each": each_out,
            "merged": merged,
            "each_stats": each_stats,
            "merge_stats": merge_stats,
        }
        all_stats[label] = {"each": each_stats, "merged": merge_stats}
        print(f"Wrote {each_path} ({len(each_out)} rows)")
        print(f"Wrote {merged_path} ({len(merged)} rows)")

    write_report(outdir / "intermediate_matrices_report.md", units, results)
    with open(outdir / "intermediate_matrices_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(json.dumps(all_stats, indent=2))
    print(f"Wrote {outdir / 'intermediate_matrices_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
