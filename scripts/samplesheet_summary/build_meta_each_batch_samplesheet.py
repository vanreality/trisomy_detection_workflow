#!/usr/bin/env python3
"""Build per-dataset meta samplesheet with dataset_status 0/1/2.

One row per (sample, dataset). Includes datasets dropped during the
previous reorganization (noisy multi-dataset rows + blacklist samples).
``dataset`` is the MQ folder id from the original path (``YYYYMMDD-XML``,
``YYYYMMDD-XML_igtc``, ``YYYYMMDD-XML.igtc``, …).

A sample with only one dataset and ``depth_qc=pass`` is status 1 even if the
pred_label is noisy (no alternative run exists). Depth-fail and blacklist
rows stay 0. Methylation conversion QC is not used for ``dataset_status``.

Writes under ``--outdir``:
  ``meta_each_batch_samplesheet.csv``
  ``mqres_samplesheet.csv`` (dataset column + status-0 rows restored)
  ``intermediate_each_batch_modeA.parquet`` (same)
  QC fields ``qc_dataset``, ``puc19``, ``lambda`` live on the meta sheet.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "samplesheet_summary"))
from finalize_samplesheet_review import (  # noqa: E402
    NOISY_BAD_BATCHES,
    NOISY_MIN_T,
    SAMPLE_BLACKLIST,
    yyyymmdd,
)
from reorganize_samplesheet import (  # noqa: E402
    DEFAULT_PIPELINE_OUTPUTS,
    canonical_sample_id,
    extract_batch_from_path,
    format_ff,
    load_ff_higher_precision,
    disk_sample_candidates,
    n_trisomy_signals,
    normalize_batch_key,
    repair_path,
    scan_batch_pred,
)

DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")
DEFAULT_ORIG_META = Path(
    "/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet.csv"
)
DEFAULT_ORIG_MQRES = Path("/lustre1/cqyi/syfan/nipt_article_plot/mqres_samplesheet.csv")
DEFAULT_FF_DIR = Path("/lustre1/cqyi/syfan/nipt_article_plot/ff_higher_precision")
DATA_RUN_OUTPUT = Path(
    "/lustre1/cqyi/yfan/workflow/NIPT/00.data/target_data/data_run/output"
)
META_SCORE_COLS = [
    "ff_before_mq",
    "ff_after_mq",
    "pred_label",
    "mean_target_coverage",
    "snp_mean_coverage",
    "depth_qc",
]

SNP_DEPTH_PASS = 20.0
OUT_COLS = [
    "sample",
    "dataset",
    "qc_dataset",
    "puc19",
    "lambda",
    "ff_before_mq",
    "ff_after_mq",
    "pred_label",
    "purity",
    "depth_qc",
    "dataset_status",
    "set",
    "mean_target_coverage",
    "snp_mean_coverage",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--mqres", type=Path, default=None)
    p.add_argument("--orig-meta", type=Path, default=DEFAULT_ORIG_META)
    p.add_argument("--orig-mqres", type=Path, default=DEFAULT_ORIG_MQRES)
    p.add_argument(
        "--pipeline-output",
        action="append",
        default=None,
        help="Pipeline output root(s) for per-batch score scan (repeatable).",
    )
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args(argv)


def dataset_id_from_paths(deconv: object, bam: object) -> Optional[str]:
    """MQ/sequencing folder id with suffix, e.g. ``20260321-XML`` / ``20251226-XML_igtc``."""
    return extract_batch_from_path(deconv) or extract_batch_from_path(bam)


def bam_root(clean_bam: object) -> Optional[Path]:
    p = Path(str(clean_bam))
    if "bwameth_results" not in p.parts:
        return None
    i = p.parts.index("bwameth_results")
    return Path(*p.parts[: i + 1])


def pipeline_root_from_bam(clean_bam: object) -> Optional[str]:
    root = bam_root(clean_bam)
    if root is None or root.parent is None:
        return None
    # .../<batch>/bwameth_results → pipeline output is parent of <batch>
    batch_dir = root.parent
    return str(batch_dir.parent)


def collect_pipeline_roots(
    units: pd.DataFrame, extra: Optional[Iterable[str]]
) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()

    def add(path: object) -> None:
        if not path:
            return
        s = str(path)
        if s in seen:
            return
        if Path(s).is_dir():
            seen.add(s)
            roots.append(s)

    for r in extra or []:
        add(r)
    for r in DEFAULT_PIPELINE_OUTPUTS:
        add(r)
    add(DATA_RUN_OUTPUT)
    for bam in units["clean_bam"].dropna().unique():
        add(pipeline_root_from_bam(bam))
    return roots


def _prepare_mqres(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample"] = out["sample"].map(canonical_sample_id)
    if "deconv_res" in out.columns:
        out["deconv_res"] = out["deconv_res"].map(repair_path)
    if "clean_bam" in out.columns:
        out["clean_bam"] = out["clean_bam"].map(repair_path)
    out["batch_raw"] = [
        dataset_id_from_paths(d, b)
        for d, b in zip(out["deconv_res"], out["clean_bam"])
    ]
    if "mqres_batch" in out.columns:
        out["batch"] = out["mqres_batch"].map(yyyymmdd)
    elif "dataset" in out.columns:
        out["batch"] = out["dataset"].map(yyyymmdd)
    else:
        out["batch"] = out["batch_raw"].map(yyyymmdd)
    miss_date = out["batch"].isna()
    out.loc[miss_date, "batch"] = out.loc[miss_date, "batch_raw"].map(yyyymmdd)
    out["batch_key"] = out["batch_raw"].map(normalize_batch_key)
    miss_key = out["batch_key"].isna() & out["batch"].notna()
    out.loc[miss_key, "batch_key"] = out.loc[miss_key, "batch"].map(
        lambda d: f"{d}-XML" if d else None
    )
    out["is_se"] = (
        out["deconv_res"].astype(str).str.contains("single_end", case=False, na=False)
    )
    if "puc19" not in out.columns:
        out["puc19"] = np.nan
    if "lambda" not in out.columns:
        out["lambda"] = np.nan
    if "qc_batch" not in out.columns:
        out["qc_batch"] = np.nan
    return out


def collapse_units(mqres: pd.DataFrame) -> pd.DataFrame:
    """One row per (sample, batch); prefer non-SE + parquet deconv."""
    rows = []
    for (sample, batch), g in mqres.groupby(["sample", "batch"], sort=False):
        if not sample or not batch:
            continue
        g2 = g[~g["is_se"]] if (~g["is_se"]).any() else g
        g2 = g2.copy()
        g2["_prio"] = g2["deconv_res"].map(
            lambda p: 0 if str(p).endswith(".parquet") else 1
        )
        r = g2.sort_values("_prio").iloc[0]
        puc = g["puc19"].dropna()
        lam = g["lambda"].dropna()
        qc = g["qc_batch"].dropna() if "qc_batch" in g.columns else pd.Series(dtype=object)
        if qc.empty and "qc_dataset" in g.columns:
            qc = g["qc_dataset"].dropna()
        rows.append(
            {
                "sample": sample,
                "batch": batch,
                "batch_key": r["batch_key"],
                "batch_raw": r["batch_raw"],
                "clean_bam": r["clean_bam"],
                "deconv_res": r["deconv_res"],
                "puc19": float(puc.iloc[0]) if len(puc) else np.nan,
                "lambda": float(lam.iloc[0]) if len(lam) else np.nan,
                "qc_batch": yyyymmdd(qc.iloc[0]) if len(qc) else None,
            }
        )
    return pd.DataFrame(rows)


def union_units(cur: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    """Current mqres wins on overlap (QC + repaired paths); keep orig-only rows."""
    cur_u = collapse_units(cur)
    orig_u = collapse_units(orig)
    cur_u["in_current_mqres"] = True
    orig_u["in_current_mqres"] = False
    key = ["sample", "batch"]
    orig_only = orig_u.merge(cur_u[key], on=key, how="left", indicator=True)
    orig_only = orig_only[orig_only["_merge"] == "left_only"].drop(columns="_merge")
    out = pd.concat([cur_u, orig_only], ignore_index=True)
    return out


def candidate_batch_keys(row: pd.Series) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()

    def add(val: object) -> None:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return
        s = str(val).strip()
        if not s or s in seen:
            return
        seen.add(s)
        keys.append(s)
        nk = normalize_batch_key(s)
        if nk and nk not in seen:
            seen.add(nk)
            keys.append(nk)

    add(row.get("batch_key"))
    add(row.get("batch_raw"))
    add(extract_batch_from_path(row.get("deconv_res")))
    add(extract_batch_from_path(row.get("clean_bam")))
    batch = row.get("batch")
    if batch:
        add(f"{batch}-XML")
        add(f"{batch}-XML_igtc")
    return keys


def _better_scan(a: dict, b: dict) -> dict:
    rank = {"ok": 3, "partial": 2, "no_report": 1, "no_zscore_dir": 0}
    if not a:
        return b or {}
    if not b:
        return a
    if rank.get(b.get("status") or "", 0) > rank.get(a.get("status") or "", 0):
        merged = dict(b)
        for k, v in a.items():
            if merged.get(k) is None and v is not None:
                merged[k] = v
        return merged
    merged = dict(a)
    for k, v in b.items():
        if merged.get(k) is None and v is not None:
            merged[k] = v
    return merged


def scan_unit(row: dict, pipeline_outputs: list[str]) -> dict:
    scanned: dict = {}
    for key in candidate_batch_keys(pd.Series(row)):
        hit = scan_batch_pred(row["sample"], key, pipeline_outputs) or {}
        scanned = _better_scan(scanned, hit)
        if scanned.get("status") == "ok" and scanned.get("snp_mean_coverage") is not None:
            break
    if scanned.get("status") in {None, "no_zscore_dir", "no_report"}:
        root = bam_root(row.get("clean_bam"))
        if root is not None and (root / "zscore_downstream").is_dir():
            pipe = pipeline_root_from_bam(row.get("clean_bam"))
            batch_name = root.parent.name if root.parent else None
            if pipe and batch_name:
                hit = scan_batch_pred(row["sample"], batch_name, [pipe]) or {}
                scanned = _better_scan(scanned, hit)
    return scanned


def _combine_meta_frames(cur_meta: pd.DataFrame, orig_meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frames = []
    for df in (cur_meta, orig_meta):
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["sample"] = tmp["sample"].map(canonical_sample_id)
        keep = [c for c in cols if c in tmp.columns]
        if "sample" not in keep:
            continue
        frames.append(tmp[keep].drop_duplicates("sample", keep="first"))
    if not frames:
        return pd.DataFrame(columns=["sample"])
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="sample", how="outer", suffixes=("", "_orig"))
        for c in extra.columns:
            if c == "sample":
                continue
            oc = f"{c}_orig"
            if oc in out.columns:
                if c in out.columns:
                    out[c] = out[c].combine_first(out[oc])
                else:
                    out[c] = out[oc]
                out = out.drop(columns=[oc])
    return out


def sample_level_fields(cur_meta: pd.DataFrame, orig_meta: pd.DataFrame) -> pd.DataFrame:
    return _combine_meta_frames(cur_meta, orig_meta, ["sample", "purity", "set"])


def meta_score_table(cur_meta: pd.DataFrame, orig_meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["sample", *META_SCORE_COLS]
    src = _combine_meta_frames(cur_meta, orig_meta, cols)
    rename = {c: f"meta_{c}" for c in META_SCORE_COLS if c in src.columns}
    return src.rename(columns=rename)


def _is_na(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def read_hs_mean_target(clean_bam: object) -> Optional[float]:
    """Picard HsMetrics MEAN_TARGET_COVERAGE next to the clean BAM."""
    if not clean_bam or (isinstance(clean_bam, float) and np.isnan(clean_bam)):
        return None
    bam = Path(str(clean_bam))
    name = bam.name
    stem_id = name.replace(".clean.bam", "").replace(".bam", "")
    candidates = [
        bam.with_name(f"{stem_id}_hs_metrics.txt"),
        bam.with_name(name.replace(".bam", "_hs_metrics.txt")),
        bam.with_name(name.replace(".clean.bam", ".clean_hs_metrics.txt")),
    ]
    for disk in disk_sample_candidates(stem_id):
        candidates.append(bam.with_name(f"{disk}_hs_metrics.txt"))
        candidates.append(bam.with_name(f"{disk}.clean_hs_metrics.txt"))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for i, line in enumerate(lines):
        if "MEAN_TARGET_COVERAGE" not in line:
            continue
        headers = line.split("\t")
        if "MEAN_TARGET_COVERAGE" not in headers:
            continue
        if i + 1 >= len(lines):
            return None
        vals = lines[i + 1].split("\t")
        row = dict(zip(headers, vals))
        try:
            return float(row["MEAN_TARGET_COVERAGE"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def fill_hs_mean_target(units: pd.DataFrame) -> pd.DataFrame:
    out = units.copy()
    miss = out["mean_target_coverage"].isna()
    if not miss.any():
        return out
    filled = 0
    for idx in out.index[miss]:
        val = read_hs_mean_target(out.at[idx, "clean_bam"])
        if val is None:
            continue
        out.at[idx, "mean_target_coverage"] = val
        filled += 1
    print(f"Filled mean_target_coverage from HsMetrics for {filled} units")
    return out


def fill_single_batch_from_meta(units: pd.DataFrame, meta_scores: pd.DataFrame) -> pd.DataFrame:
    """Copy sample-level meta scores onto the unique *current* batch when zscore is missing.

    Restored/filtered extra batches must not block this: meta scores belong to the
    remaining analysis batch, not to dropped noisy copies.
    """
    out = units.merge(meta_scores, on="sample", how="left")
    n_current = out.groupby("sample")["in_current_mqres"].transform("sum")
    single = out["in_current_mqres"] & (n_current == 1)
    n_units = 0
    for col in ("ff_before_mq", "ff_after_mq", "pred_label", "mean_target_coverage", "snp_mean_coverage"):
        src = f"meta_{col}"
        if src not in out.columns:
            continue
        miss = single & out[col].isna() & out[src].notna()
        n_units = max(n_units, int(miss.sum()))
        out.loc[miss, col] = out.loc[miss, src]
    if DEFAULT_FF_DIR.is_dir():
        need_ff = single & out["ff_before_mq"].isna()
        for idx in out.index[need_ff]:
            loaded = load_ff_higher_precision(DEFAULT_FF_DIR, out.at[idx, "sample"])
            if not loaded:
                continue
            out.at[idx, "ff_before_mq"] = format_ff(loaded.get("ff_before_mq"))
            out.at[idx, "ff_after_mq"] = format_ff(loaded.get("ff_after_mq"))
    out["ff_before_mq"] = [format_ff(v) for v in out["ff_before_mq"]]
    out["ff_after_mq"] = [format_ff(v) for v in out["ff_after_mq"]]
    drop = [c for c in out.columns if c.startswith("meta_")]
    out = out.drop(columns=drop)
    print(f"Filled scores from sample-level meta for {n_units} unique-current-batch units")
    return out


def depth_qc_of(snp: object, mean_target: object) -> Optional[str]:
    """pass/fail from known coverage; None if coverage was never measured."""
    if not _is_na(snp):
        try:
            return "pass" if float(snp) >= SNP_DEPTH_PASS else "fail"
        except (TypeError, ValueError):
            pass
    if not _is_na(mean_target):
        try:
            return "pass" if float(mean_target) >= SNP_DEPTH_PASS else "fail"
        except (TypeError, ValueError):
            return None
    return None


def rank_tuple(row: pd.Series) -> tuple:
    """Higher is better: deepest SNP coverage, then fewer T signals, then target cov."""
    snp = row["snp_mean_coverage"]
    tgt = row["mean_target_coverage"]
    puc = row["puc19"]
    pred = row["pred_label"]
    n_t = 0 if _is_na(pred) else n_trisomy_signals(pred)
    try:
        date = int(row["batch"])
    except (TypeError, ValueError):
        date = 0
    return (
        -1e18 if _is_na(snp) else float(snp),
        -n_t,
        -1e18 if _is_na(tgt) else float(tgt),
        -1e18 if _is_na(puc) else float(puc),
        date,
    )


def assign_batch_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_batch"] = out.groupby("sample")["batch"].transform("nunique")
    single = out["n_batch"] == 1
    out["n_t_signals"] = [
        n_trisomy_signals(p) if not _is_na(p) else 0 for p in out["pred_label"]
    ]
    out["depth_qc"] = [
        depth_qc_of(s, t) for s, t in zip(out["snp_mean_coverage"], out["mean_target_coverage"])
    ]
    pred_known = ~out["pred_label"].map(_is_na)
    out["noisy_pred"] = pred_known & (out["n_t_signals"] >= NOISY_MIN_T)
    out["global_noisy"] = out["batch"].isin(NOISY_BAD_BATCHES)
    out["blacklist"] = out["sample"].isin(SAMPLE_BLACKLIST)
    out["filtered_in_reorg"] = ~out["in_current_mqres"]
    # Conversion QC (puc19/lambda) lives on meta as qc_dataset and does not drive
    # batch_status. Single-batch keepers: noisy pred / global noisy chemistry
    # is still the only available run; if depth_qc passed, use it as primary.
    noisy_bad = (out["noisy_pred"] | out["global_noisy"]) & ~single
    bad = (
        out["blacklist"]
        | out["filtered_in_reorg"]
        | (out["depth_qc"] == "fail")
        | noisy_bad
    )
    out["batch_status"] = np.where(bad, 0, -1).astype(int)

    for sample, g in out.groupby("sample", sort=False):
        good_idx = g.index[out.loc[g.index, "batch_status"] < 0]
        if len(good_idx) == 0:
            continue
        if len(good_idx) == 1:
            out.loc[good_idx, "batch_status"] = 1
            continue
        ranked = sorted(good_idx, key=lambda i: rank_tuple(out.loc[i]), reverse=True)
        out.loc[ranked[0], "batch_status"] = 1
        out.loc[ranked[1:], "batch_status"] = 2
    return out


def scan_all(units: pd.DataFrame, pipeline_outputs: list[str], workers: int) -> pd.DataFrame:
    recs = units.to_dict("records")
    results: list[Optional[dict]] = [None] * len(recs)
    n = len(recs)
    workers = max(1, workers)
    print(f"Scanning {n} sample×batch units ({workers} threads)...")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_unit, rec, pipeline_outputs): i for i, rec in enumerate(recs)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[i] = {"status": "error", "error": str(exc)}
            done += 1
            if done % 100 == 0 or done == n:
                print(f"  scanned {done}/{n}")
    pred, ff_b, ff_a, tgt, snp, st = [], [], [], [], [], []
    for sc in results:
        sc = sc or {}
        pred.append(sc.get("pred_label"))
        ff_b.append(sc.get("ff_before_mq"))
        ff_a.append(sc.get("ff_after_mq"))
        tgt.append(sc.get("mean_target_coverage"))
        snp.append(sc.get("snp_mean_coverage"))
        st.append(sc.get("status"))
    out = units.copy()
    out["pred_label"] = pred
    out["ff_before_mq"] = [format_ff(v) for v in ff_b]
    out["ff_after_mq"] = [format_ff(v) for v in ff_a]
    out["mean_target_coverage"] = tgt
    out["snp_mean_coverage"] = snp
    out["scan_status"] = st
    return out


def rewrite_mqres_with_datasets(
    cur_prepared: pd.DataFrame,
    orig_prepared: pd.DataFrame,
    units: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Rename mqres_batch→dataset (with path suffix) and restore status-0 rows."""
    cols = ["sample", "deconv_res", "clean_bam", "dataset"]
    cur = cur_prepared.copy()
    cur["dataset"] = cur["batch_raw"]
    cur_out = cur[cols]

    restored_keys = set(
        zip(
            units.loc[~units["in_current_mqres"], "sample"],
            units.loc[~units["in_current_mqres"], "batch"],
        )
    )
    extra = orig_prepared.copy()
    extra["dataset"] = extra["batch_raw"]
    extra = extra[
        extra.apply(lambda r: (r["sample"], r["batch"]) in restored_keys, axis=1)
    ]
    extra_out = extra[cols] if len(extra) else cur_out.iloc[0:0].copy()
    out = pd.concat([cur_out, extra_out], ignore_index=True)
    out = out.drop_duplicates(
        ["sample", "dataset", "clean_bam", "deconv_res"], keep="first"
    )
    out = _mqres_one_bam_per_dataset(out)
    out = out.sort_values(["sample", "dataset", "clean_bam", "deconv_res"]).reset_index(
        drop=True
    )
    out.to_csv(out_path, index=False)
    return extra_out


def _fill_qc_from_prev_meta(units: pd.DataFrame, prev_path: Path) -> pd.DataFrame:
    """After QC columns leave mqres, recover them from an existing meta_each sheet."""
    if not prev_path.is_file():
        return units
    prev = pd.read_csv(prev_path)
    need = [c for c in ("qc_dataset", "puc19", "lambda") if c in prev.columns]
    if not need or "dataset" not in prev.columns:
        return units
    src = prev[["sample", "dataset", *need]].drop_duplicates(["sample", "dataset"])
    out = units.merge(src, on=["sample", "dataset"], how="left", suffixes=("", "_prev"))
    for c in need:
        prev_c = f"{c}_prev"
        if prev_c in out.columns:
            if c in out.columns:
                out[c] = out[c].combine_first(out[prev_c])
            else:
                out[c] = out[prev_c]
            out = out.drop(columns=[prev_c])
    return out


def _mqres_one_bam_per_dataset(mqres: pd.DataFrame) -> pd.DataFrame:
    """Keep PE+SE deconv for a single clean_bam per (sample, dataset).

    Some datasets have replicate BAMs (``*_1.clean.bam`` / ``*_2.clean.bam``).
    Meta and intermediate are one row per (sample, dataset); mqres must match
    that key count (2 deconv rows each), using the same bam collapse as units
    (prefer non-SE + parquet).
    """
    df = mqres.copy()
    df["is_se"] = (
        df["deconv_res"].astype(str).str.contains("single_end", case=False, na=False)
    )
    df["_prio"] = df["deconv_res"].map(
        lambda p: 0 if str(p).endswith(".parquet") else 1
    )
    keep_idx = []
    for (sample, dataset), g in df.groupby(["sample", "dataset"], sort=False):
        g_pe = g[~g["is_se"]] if (~g["is_se"]).any() else g
        bam = g_pe.sort_values("_prio").iloc[0]["clean_bam"]
        sub = g[g["clean_bam"] == bam]
        for flag in (False, True):
            gg = sub[sub["is_se"] == flag]
            if gg.empty:
                continue
            keep_idx.append(gg.sort_values("_prio").index[0])
    out = df.loc[keep_idx].drop(columns=["is_se", "_prio"])
    return out


def rewrite_intermediate_modeA(outdir: Path, units: pd.DataFrame) -> int:
    """Rename batch→dataset and append status-0 units missing from the parquet."""
    path = outdir / "intermediate_each_batch_modeA.parquet"
    if not path.is_file():
        print(f"Skip intermediate update; missing {path}")
        return 0
    ib = pd.read_parquet(path)
    lu = units[["sample", "batch", "batch_raw"]].drop_duplicates(
        ["sample", "batch"], keep="first"
    )
    ib["date"] = ib["batch"].map(yyyymmdd)
    ib = ib.merge(
        lu.rename(columns={"batch": "date", "batch_raw": "dataset"}),
        on=["sample", "date"],
        how="left",
    )
    miss = ib["dataset"].isna()
    ib.loc[miss, "dataset"] = ib.loc[miss, "date"].map(
        lambda d: f"{d}-XML" if d else None
    )
    ib = ib.drop(columns=["batch", "date"])
    front = ["sample", "ff_before_mq", "ff_after_mq", "dataset"]
    ib = ib[front + [c for c in ib.columns if c not in front]]

    have = set(zip(ib["sample"], ib["dataset"].map(yyyymmdd)))
    extra_u = units[~units.apply(lambda r: (r["sample"], r["batch"]) in have, axis=1)]
    n_add = int(len(extra_u))
    if n_add:
        blank = {c: np.nan for c in ib.columns}
        rows = []
        for r in extra_u.itertuples():
            rec = dict(blank)
            rec["sample"] = r.sample
            rec["dataset"] = r.batch_raw
            rec["ff_before_mq"] = getattr(r, "ff_before_mq", np.nan)
            rec["ff_after_mq"] = getattr(r, "ff_after_mq", np.nan)
            rows.append(rec)
        ib = pd.concat([ib, pd.DataFrame(rows)], ignore_index=True)
    ib = ib.sort_values(["sample", "dataset"]).reset_index(drop=True)
    ib.to_parquet(path, index=False)
    return n_add


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    cur_meta_path = args.meta or (outdir / "meta_samplesheet.csv")
    cur_mqres_path = args.mqres or (outdir / "mqres_samplesheet.csv")

    cur_meta = pd.read_csv(cur_meta_path)
    cur_mqres = pd.read_csv(cur_mqres_path)
    orig_meta = pd.read_csv(args.orig_meta) if args.orig_meta.is_file() else pd.DataFrame()
    orig_mqres = pd.read_csv(args.orig_mqres) if args.orig_mqres.is_file() else pd.DataFrame()

    cur = _prepare_mqres(cur_mqres)
    orig = _prepare_mqres(orig_mqres) if len(orig_mqres) else cur.iloc[0:0].copy()
    units = union_units(cur, orig)
    pipeline_outputs = collect_pipeline_roots(units, args.pipeline_output)
    units = scan_all(units, pipeline_outputs, args.workers)
    units = fill_hs_mean_target(units)

    sample_meta = sample_level_fields(cur_meta, orig_meta)
    meta_scores = meta_score_table(cur_meta, orig_meta)
    units = units.merge(sample_meta, on="sample", how="left")
    units = fill_single_batch_from_meta(units, meta_scores)
    units = assign_batch_status(units)
    units["dataset"] = units["batch_raw"]
    units["dataset_status"] = units["batch_status"]
    if "qc_batch" in units.columns:
        units["qc_dataset"] = units["qc_batch"].map(lambda x: yyyymmdd(x) or None)
    elif "qc_dataset" not in units.columns:
        units["qc_dataset"] = None
    units = _fill_qc_from_prev_meta(units, outdir / "meta_each_batch_samplesheet.csv")

    out = units[OUT_COLS].sort_values(["sample", "dataset"]).reset_index(drop=True)
    out_path = outdir / "meta_each_batch_samplesheet.csv"
    out.to_csv(out_path, index=False)

    extra_mq = rewrite_mqres_with_datasets(
        cur, orig, units, outdir / "mqres_samplesheet.csv"
    )
    n_inter_add = rewrite_intermediate_modeA(outdir, units)

    n_primary = int((out["dataset_status"] == 1).sum())
    n_secondary = int((out["dataset_status"] == 2).sum())
    n_bad = int((out["dataset_status"] == 0).sum())
    n_samples = int(out["sample"].nunique())
    samples_no_primary = sorted(
        s
        for s, g in out.groupby("sample")
        if (g["dataset_status"] == 1).sum() == 0
    )
    restored = int((~units["in_current_mqres"]).sum())
    multi_primary = (
        out[out["dataset_status"] == 1].groupby("sample").size().gt(1).sum()
    )
    inconsistent = (
        units.groupby("sample")[["purity", "set"]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
        .sum()
    )
    summary = {
        "n_rows": int(len(out)),
        "n_samples": n_samples,
        "n_status_0": n_bad,
        "n_status_1": n_primary,
        "n_status_2": n_secondary,
        "n_restored_filtered_units": restored,
        "n_blacklist_rows": int(units["sample"].isin(SAMPLE_BLACKLIST).sum()),
        "n_samples_without_primary": len(samples_no_primary),
        "n_samples_with_multiple_primary": int(multi_primary),
        "n_samples_inconsistent_purity_or_set": int(inconsistent),
        "scan_status_counts": {
            str(k): int(v) for k, v in units["scan_status"].value_counts(dropna=False).items()
        },
        "depth_qc_counts": {
            str(k): int(v) for k, v in out["depth_qc"].value_counts(dropna=False).items()
        },
        "n_mqres_status0_rows_added": int(len(extra_mq)),
        "n_intermediate_modeA_rows_added": int(n_inter_add),
        "dataset_suffix_counts": {
            str(k): int(v)
            for k, v in out["dataset"]
            .map(lambda s: str(s)[8:] if isinstance(s, str) and len(str(s)) >= 8 else str(s))
            .value_counts()
            .items()
        },
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    if samples_no_primary:
        print(
            f"Samples with no primary batch (all status 0): {len(samples_no_primary)}"
        )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
