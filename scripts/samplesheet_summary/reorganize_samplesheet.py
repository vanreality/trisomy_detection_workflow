#!/usr/bin/env python3
"""Reorganize AIPT meta + mqres samplesheets: batch mapping, score QA, FF precision.

Reads:
  - temporary_updated_samplesheet.csv (meta; no batches column)
  - mqres_samplesheet.csv (candidate deconv_res / clean_bam)
  - episcore_result_samplesheet.csv (selected score file paths)
  - optional ff_higher_precision/*_ff.tsv

Writes under --outdir (default: /lustre1/cqyi/AIPT_2.0/results/samplesheet_summary):
  - meta_samplesheet.csv          (meta + batches + score_source/batch + FF@0.0001)
  - mqres_samplesheet.csv         (repaired paths; selected flag; batch column)
  - report_batch_score_inconsistency.csv
  - report_multibatch_wrong_selection.csv
  - report_ff_precision.csv
  - report_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_META = "/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet.csv"
DEFAULT_MQRES = "/lustre1/cqyi/syfan/nipt_article_plot/mqres_samplesheet.csv"
DEFAULT_EPISCORE = "/lustre1/cqyi/syfan/nipt_article_plot/episcore_result_samplesheet.csv"
DEFAULT_FF_DIR = "/lustre1/cqyi/syfan/nipt_article_plot/ff_higher_precision"
DEFAULT_OUTDIR = "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary"
DEFAULT_PIPELINE_OUTPUTS = (
    "/lustre1/cqyi/myli/bert/DNA_5mC_analysis_pipeline/output",
    "/appsnew/home/myli/lustre1/bert/DNA_5mC_analysis_pipeline/output",
)
DEFAULT_POST_CURATION_CUTOFF = "20260120"
DEFAULT_FF_PRECISION = 0.0001

_TECHNICAL_SUFFIX_RE = re.compile(r"(?:_rep\d+|_\d+|_[AHX])$")
_REPORT_NAME_RE = re.compile(
    r"(?:.*_)?(?P<sample>(?:JP|P)?TAY\d+[PB]\d+[HS]\d+(?:_rep\d+)?(?:_\d+)?|HCPT\d+)"
    r"_report.*\.(?:csv|tsv)$"
)


# ---------------------------------------------------------------------------
# Sample / batch helpers
# ---------------------------------------------------------------------------

def canonical_sample_id(sample: str) -> str:
    sample = str(sample).strip()
    while True:
        updated = _TECHNICAL_SUFFIX_RE.sub("", sample)
        if updated == sample:
            break
        sample = updated
    # B→P alias for modern PTAY/JPTAY IDs (leave HCPT alone)
    if re.match(r"^(?:J)?PTAY\d+", sample, re.IGNORECASE):
        sample = sample.replace("B", "P")
    return sample


def batch_date(batch: Optional[str]) -> str:
    if not batch:
        return ""
    m = re.match(r"(\d{8})", str(batch))
    return m.group(1) if m else ""


def normalize_batch_key(batch: Optional[str]) -> Optional[str]:
    """Map ``20251218-XML_igtc`` / ``20251218-XML.igtc`` → ``20251218-XML`` when possible."""
    if not batch or (isinstance(batch, float) and np.isnan(batch)):
        return None
    b = str(batch)
    m = re.match(r"(\d{8}-XML)", b.replace(".igtc", "").replace("_igtc", ""))
    if m:
        return m.group(1)
    m = re.match(r"(\d{8})", b)
    return m.group(1) if m else b


def extract_batch_from_path(path: object) -> Optional[str]:
    """Parse a sequencing / analysis batch id from mqres or score paths."""
    if not isinstance(path, str):
        return None
    patterns = [
        r"/DNA_5mC_analysis_pipeline/output/(?P<batch>[^/]+)/",
        r"/analysis_targets/220k/(?P<batch>\d{8}[^/]+)/",
        r"/MQ_deconvolution/220k/(?P<batch>[^/]+)/",
        r"/data_run/output/(?P<batch>\d{8}[^/]*)/",
        # embedded: .../20250808-XML_igtc:HCPT0081P.clean.bam_...
        r"/(?P<batch>\d{8}-XML(?:_igtc|\.igtc)?):",
        r"/(?P<batch>\d{8}-XML(?:_igtc|\.igtc)?)/",
    ]
    for pat in patterns:
        m = re.search(pat, path)
        if m:
            return m.group("batch")
    return None


def classify_score_source(path: Optional[str]) -> str:
    if not path or not isinstance(path, str):
        return "missing"
    if "20260123_early" in path:
        return "early_20260123"
    if "20260123_middle" in path:
        return "middle_20260123"
    m = re.search(r"/DNA_5mC_analysis_pipeline/output/([^/]+)/", path)
    if m:
        return f"pipeline:{m.group(1)}"
    m = re.search(r"beta_trisomy_detection/([^/]+)/", path)
    if m:
        return f"trisomy:{m.group(1)}"
    return "other"


def format_ff(value, precision: float = DEFAULT_FF_PRECISION) -> Optional[float]:
    """Round FF to the given absolute precision (default 0.0001)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    decimals = max(0, int(round(-np.log10(precision))))
    return round(v, decimals)


def ff_decimals_in_csv(raw: object) -> Optional[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if s.lower() in {"", "nan", "none", "na"}:
        return None
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1])


# ---------------------------------------------------------------------------
# Match / prediction helpers (mirrors update_samplesheet thresholds)
# ---------------------------------------------------------------------------

def pred_label_from_zscore_df(zscore_df: pd.DataFrame) -> str:
    chr_order = [str(i) for i in range(1, 23)]
    t_labels, gray_labels = [], []
    for _, row in zscore_df.iterrows():
        label_chr = str(row["chr"]).replace("chr", "")
        if label_chr not in chr_order:
            continue
        z = row.get("Z_sum_dist", np.nan)
        if pd.isna(z):
            continue
        if z > 4.5:
            t_labels.append(f"T{label_chr}")
        elif 3 <= z <= 4.5:
            gray_labels.append(f"Gray_T{label_chr}")
    parts = t_labels + gray_labels
    return ",".join(parts) if parts else "Normal"


def n_trisomy_signals(pred_label: object) -> int:
    if pred_label is None or (isinstance(pred_label, float) and np.isnan(pred_label)):
        return 0
    return sum(
        1
        for p in str(pred_label).split(",")
        if re.match(r"^(?:Gray_)?T\d+", p.strip())
    )


def calc_match_status(label: object, pred_label: object) -> Optional[str]:
    if label is None or pred_label is None:
        return None
    if isinstance(label, float) and np.isnan(label):
        return None
    if isinstance(pred_label, float) and np.isnan(pred_label):
        return None
    label_str = str(label)
    pred_str = ",".join(i.replace("Gray_T", "T") for i in str(pred_label).split(","))
    if label_str == "Unknown":
        return None
    if label_str == pred_str:
        return "match"
    if label_str in pred_str or pred_str in label_str:
        return "partially_match"
    return "mismatch"


def labels_equal(a: object, b: object) -> bool:
    if a is None or b is None:
        return False
    if (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return False

    def norm(x):
        return ",".join(p.replace("Gray_T", "T").strip() for p in str(x).split(",") if p.strip())

    return norm(a) == norm(b)


# ---------------------------------------------------------------------------
# Path repair (from samplesheet_workflow.repair_mqres_deconv_paths)
# ---------------------------------------------------------------------------

def repair_path(path: object) -> object:
    if not isinstance(path, str):
        return path
    candidates = [path]
    if path.startswith("/appsnew/home/myli/lustre1/"):
        candidates.append(path.replace("/appsnew/home/myli/lustre1/", "/lustre1/cqyi/", 1))
    for candidate in list(candidates):
        if candidate.endswith(".txt"):
            candidates.append(candidate[:-4] + ".parquet")
        elif candidate.endswith(".parquet"):
            candidates.append(candidate[:-8] + ".txt")
    for candidate in candidates:
        if Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return path


# ---------------------------------------------------------------------------
# Score scanning
# ---------------------------------------------------------------------------

def resolve_zscore_dir(batch: str, pipeline_outputs: Iterable[str]) -> Optional[Path]:
    key = normalize_batch_key(batch)
    for root in pipeline_outputs:
        root_p = Path(root)
        for cand in (batch, key, f"{key}_igtc", f"{key}.igtc"):
            if not cand:
                continue
            p = root_p / cand / "bwameth_results" / "zscore_downstream"
            if p.is_dir():
                return p
    return None


def p_to_b_sample(sample: str) -> str:
    """Convert the last P/B alias in a PTAY/JPTAY ID back to on-disk B form."""
    last_p = sample.rfind("P")
    if last_p != -1:
        return sample[:last_p] + "B" + sample[last_p + 1 :]
    return sample


def disk_sample_candidates(sample: str) -> list[str]:
    out, seen = [], set()
    b = p_to_b_sample(sample) if re.match(r"^(?:J)?PTAY", sample) else sample
    for c in (sample, b, f"{sample}_rep1", f"{b}_rep1"):
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def scan_batch_pred(sample: str, batch: str, pipeline_outputs: Iterable[str]) -> Optional[dict]:
    """Return pred_label / FF / zscore strings from a pipeline batch zscore_downstream."""
    zdir = resolve_zscore_dir(batch, pipeline_outputs)
    if zdir is None:
        return {"batch": batch, "status": "no_zscore_dir", "pred_label": None}

    report = None
    for f in zdir.iterdir():
        if not f.is_file() or f.suffix not in {".csv", ".tsv"}:
            continue
        if "report" not in f.name:
            continue
        if sample in f.name or any(c in f.name for c in disk_sample_candidates(sample)):
            report = f
            break
        m = _REPORT_NAME_RE.match(f.name)
        if m and canonical_sample_id(m.group("sample")) == canonical_sample_id(sample):
            report = f
            break

    pred = None
    beta_zscores = rc_zscores = final_zscores = None
    mean_target_coverage = None
    if report is not None:
        try:
            zscore_df = pd.read_csv(report)
            pred = pred_label_from_zscore_df(zscore_df)
            chr_order = [str(i) for i in range(1, 23)]
            chr_rows = {
                str(row["chr"]).replace("chr", ""): row for _, row in zscore_df.iterrows()
            }

            def combine(col_name):
                vals = []
                for i in chr_order:
                    row = chr_rows.get(i)
                    if row is not None and col_name in row:
                        vals.append(str(row[col_name]))
                    else:
                        vals.append("")
                return ",".join(vals)

            if "s_inter" in zscore_df.columns:
                beta_zscores = combine("s_inter")
            if "zscore" in zscore_df.columns:
                rc_zscores = combine("zscore")
            if "Z_sum_dist" in zscore_df.columns:
                final_zscores = combine("Z_sum_dist")
            if "Mean_Target_Cov" in zscore_df.columns and len(zscore_df):
                mean_target_coverage = float(zscore_df["Mean_Target_Cov"].values[0])
            # report-level FF aliases
            for before_col, after_col in (
                ("FF_Before", "FF_After"),
                ("ff_before_mq", "ff_after_mq"),
            ):
                if before_col in zscore_df.columns and after_col in zscore_df.columns:
                    # may be repeated per-chr; take first
                    break
        except Exception:
            pred = None

    ff_b = ff_a = None
    snp_cov = cpg_cov = None
    disk_hit = None
    ff_source = None
    for disk in disk_sample_candidates(sample):
        # Prefer higher-precision FF artifact under the sample workdir
        hp = (
            zdir
            / "beta_zscore"
            / disk
            / "estimate_ff_higher_precision"
            / f"{disk}_ff.tsv"
        )
        if hp.is_file():
            loaded = _ff_from_tsv(hp)
            if loaded:
                ff_b, ff_a = loaded["ff_before_mq"], loaded["ff_after_mq"]
                ff_source = str(hp)
        sr = zdir / "beta_zscore" / disk / "collect_reports" / "summary_report.tsv"
        if sr.is_file():
            cov = pd.read_csv(sr, sep="\t")
            if ff_b is None and "ff_before_mq" in cov.columns:
                ff_b = float(cov["ff_before_mq"].iloc[0])
                ff_a = float(cov["ff_after_mq"].iloc[0])
                ff_source = str(sr)
            if "snp_mean_coverage" in cov.columns:
                snp_cov = float(cov["snp_mean_coverage"].iloc[0])
            if "cpg_mean_coverage" in cov.columns:
                cpg_cov = float(cov["cpg_mean_coverage"].iloc[0])
            disk_hit = disk
            break

    if report is not None and ff_b is None:
        try:
            zscore_df = pd.read_csv(report)
            if "FF_Before" in zscore_df.columns:
                ff_b = float(zscore_df["FF_Before"].iloc[0])
                ff_a = float(zscore_df["FF_After"].iloc[0])
                ff_source = str(report)
        except Exception:
            pass

    if pred is None and disk_hit is None:
        return {"batch": batch, "status": "no_report", "pred_label": None, "zdir": str(zdir)}

    return {
        "batch": batch,
        "status": "ok" if pred is not None else "partial",
        "pred_label": pred,
        "beta_zscores": beta_zscores,
        "rc_zscores": rc_zscores,
        "final_zscores": final_zscores,
        "mean_target_coverage": mean_target_coverage,
        "snp_mean_coverage": snp_cov,
        "cpg_mean_coverage": cpg_cov,
        "ff_before_mq": ff_b,
        "ff_after_mq": ff_a,
        "ff_source": ff_source,
        "report": str(report) if report else None,
        "zdir": str(zdir),
        "disk_sample": disk_hit,
    }


def _ff_from_tsv(path: Path) -> Optional[dict]:
    df = pd.read_csv(path, sep="\t")
    if "ff_before_mq" not in df.columns or "ff_after_mq" not in df.columns:
        return None
    if "chr" in df.columns:
        all_row = df[df["chr"].astype(str) == "all"]
        if not all_row.empty:
            return {
                "ff_before_mq": float(all_row["ff_before_mq"].iloc[0]),
                "ff_after_mq": float(all_row["ff_after_mq"].iloc[0]),
                "source": str(path),
            }
    return {
        "ff_before_mq": float(df["ff_before_mq"].iloc[0]),
        "ff_after_mq": float(df["ff_after_mq"].iloc[0]),
        "source": str(path),
    }


def load_ff_higher_precision(ff_dir: Path, sample: str) -> Optional[dict]:
    path = ff_dir / f"{sample}_ff.tsv"
    if path.is_file():
        loaded = _ff_from_tsv(path)
        if loaded:
            return loaded
    return None


def find_ff_in_pipeline(
    sample: str,
    batches: Iterable[str],
    pipeline_outputs: Iterable[str],
) -> Optional[dict]:
    """Locate 0.0001-precision FF from pipeline mq/zscore output dirs."""
    for batch in batches:
        if not batch or (isinstance(batch, float) and np.isnan(batch)):
            continue
        batch = str(batch)
        zdir = resolve_zscore_dir(batch, pipeline_outputs)
        if zdir is None:
            continue
        for disk in disk_sample_candidates(sample):
            candidates = [
                zdir
                / "beta_zscore"
                / disk
                / "estimate_ff_higher_precision"
                / f"{disk}_ff.tsv",
                zdir / "beta_zscore" / disk / "snp_to_ff" / f"{disk}_ff.tsv",
                zdir / "beta_zscore" / disk / "collect_reports" / "summary_report.tsv",
            ]
            for path in candidates:
                if not path.is_file():
                    continue
                if path.name == "summary_report.tsv":
                    cov = pd.read_csv(path, sep="\t")
                    if "ff_before_mq" in cov.columns:
                        return {
                            "ff_before_mq": float(cov["ff_before_mq"].iloc[0]),
                            "ff_after_mq": float(cov["ff_after_mq"].iloc[0]),
                            "source": str(path),
                        }
                else:
                    loaded = _ff_from_tsv(path)
                    if loaded:
                        return loaded
        # top-level report_data.csv
        if zdir.is_dir():
            for f in zdir.iterdir():
                if not f.is_file() or "report" not in f.name:
                    continue
                if sample not in f.name and not any(
                    c in f.name for c in disk_sample_candidates(sample)
                ):
                    continue
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
                if "FF_Before" in df.columns and "FF_After" in df.columns:
                    return {
                        "ff_before_mq": float(df["FF_Before"].iloc[0]),
                        "ff_after_mq": float(df["FF_After"].iloc[0]),
                        "source": str(f),
                    }
    return None


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------

def build_mqres_with_batches(mqres_df: pd.DataFrame) -> pd.DataFrame:
    out = mqres_df.copy()
    out["sample"] = out["sample"].map(canonical_sample_id)
    out["deconv_res"] = out["deconv_res"].map(repair_path)
    out["clean_bam"] = out["clean_bam"].map(repair_path)
    out["batch"] = out["deconv_res"].map(extract_batch_from_path)
    # fallback: try clean_bam
    miss = out["batch"].isna()
    out.loc[miss, "batch"] = out.loc[miss, "clean_bam"].map(extract_batch_from_path)
    out["batch_key"] = out["batch"].map(normalize_batch_key)
    out["is_single_end"] = out["deconv_res"].astype(str).str.contains("single_end_", regex=False)
    return out


def sample_batch_map(mqres: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample, g in mqres.groupby("sample"):
        batches = sorted({b for b in g["batch"].dropna().unique()})
        keys = sorted({k for k in g["batch_key"].dropna().unique()})
        rows.append(
            {
                "sample": sample,
                "batches": ";".join(batches),
                "batch_keys": ";".join(keys),
                "n_batches": len(batches),
                "n_mqres_rows": len(g),
            }
        )
    return pd.DataFrame(rows)


def attach_score_source(meta: pd.DataFrame, episcore_df: pd.DataFrame) -> pd.DataFrame:
    epi = episcore_df.copy()
    epi["sample"] = epi["sample"].map(canonical_sample_id)
    epi = epi.drop_duplicates(subset=["sample"], keep="first")
    out = meta.merge(epi[["sample", "episcore_file"]], on="sample", how="left")
    out["score_source"] = out["episcore_file"].map(classify_score_source)
    out["score_batch"] = out["episcore_file"].map(extract_batch_from_path)
    out["score_batch_key"] = out["score_batch"].map(normalize_batch_key)
    return out


def flag_batch_score_inconsistencies(meta: pd.DataFrame) -> pd.DataFrame:
    """Samples whose selected score batch is not among mqres candidate batches."""
    rows = []
    for r in meta.itertuples():
        score_batch = getattr(r, "score_batch", None)
        if not score_batch or (isinstance(score_batch, float) and np.isnan(score_batch)):
            # early/middle reanalysis sources are expected not to be mqres batches
            if getattr(r, "score_source", "").startswith(("early_", "middle_", "trisomy:")):
                continue
            if getattr(r, "score_source", "") == "missing":
                rows.append(
                    {
                        "sample": r.sample,
                        "issue": "missing_score_file",
                        "score_source": r.score_source,
                        "score_batch": None,
                        "mqres_batches": r.batches,
                        "detail": "no episcore_file / symlink target",
                    }
                )
            continue

        mq_batches = [b for b in str(r.batches).split(";") if b]
        mq_keys = {normalize_batch_key(b) for b in mq_batches}
        sb = str(score_batch)
        sbk = normalize_batch_key(sb)
        if sb in mq_batches or (sbk and sbk in mq_keys):
            continue
        rows.append(
            {
                "sample": r.sample,
                "issue": "score_batch_not_in_mqres",
                "score_source": r.score_source,
                "score_batch": sb,
                "mqres_batches": r.batches,
                "detail": (
                    "Selected episcore/zscore path batch is not among mqres candidate "
                    "batches — scores may come from a re-run / different BAM than mqres."
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_multibatch_selection(
    meta: pd.DataFrame,
    pipeline_outputs: Iterable[str],
    cutoff: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare pred performance across pipeline batches for multi-batch samples.

    Scans **all** samples with ``n_batches > 1`` unless ``cutoff`` is set, in which
    case only samples with any batch on/after that date are included.
    """
    scan_rows = []
    focus = meta[meta["n_batches"] > 1].copy()
    if cutoff:
        focus = focus[
            focus["batch_keys"].map(
                lambda s: any(batch_date(b) >= cutoff for b in str(s).split(";") if b)
            )
        ]

    for r in focus.itertuples():
        keys = [k for k in str(r.batch_keys).split(";") if k]
        for key in keys:
            scanned = scan_batch_pred(r.sample, key, pipeline_outputs)
            if scanned is None:
                continue
            disk_pred = scanned.get("pred_label")
            scan_rows.append(
                {
                    "sample": r.sample,
                    "label": r.label,
                    "meta_pred": r.pred_label,
                    "meta_match": r.match_status,
                    "ff_before_mq": getattr(r, "ff_before_mq", None),
                    "batch": key,
                    "scan_status": scanned.get("status"),
                    "disk_pred": disk_pred,
                    "disk_match": calc_match_status(r.label, disk_pred),
                    "n_t_meta": n_trisomy_signals(r.pred_label),
                    "n_t_disk": n_trisomy_signals(disk_pred),
                    "scores_match_meta": labels_equal(disk_pred, r.pred_label)
                    if disk_pred is not None
                    else None,
                    "is_score_batch": (
                        normalize_batch_key(getattr(r, "score_batch", None)) == key
                    ),
                    "report": scanned.get("report"),
                }
            )

    scan_df = pd.DataFrame(scan_rows)
    suspects = []
    if scan_df.empty:
        return scan_df, pd.DataFrame(suspects)

    for sample, g in scan_df.groupby("sample"):
        meta_match = g["meta_match"].iloc[0]
        meta_pred = g["meta_pred"].iloc[0]
        label = g["label"].iloc[0]
        n_t_meta = n_trisomy_signals(meta_pred)
        good = g[g["disk_match"] == "match"]
        ok_rows = g[g["scan_status"] == "ok"].drop_duplicates("batch")
        cleaner = (
            ok_rows[ok_rows["n_t_disk"] < n_t_meta] if n_t_meta >= 3 else ok_rows.iloc[0:0]
        )

        reason = None
        good_batches: list[str] = []
        good_preds: list[str] = []

        if len(good) > 0 and meta_match != "match":
            good_batches = sorted(good["batch"].unique())
            good_preds = [f"{row.batch}:{row.disk_pred}" for row in good.itertuples()]
            reason = (
                "meta pred mismatches label (with trisomy signals) while another "
                "pipeline batch predicts the true label"
                if meta_match == "mismatch" and n_t_meta >= 1
                else "meta/selected pred is not match while another batch matches label"
            )
        elif len(cleaner) > 0 and n_t_meta >= 3:
            best = cleaner.sort_values(["n_t_disk", "batch"]).iloc[0]
            if int(best["n_t_disk"]) <= max(1, n_t_meta - 2):
                good_batches = sorted(cleaner["batch"].unique())
                good_preds = [
                    f"{row.batch}:{row.disk_pred}(n_t={int(row.n_t_disk)})"
                    for row in cleaner.itertuples()
                ]
                reason = (
                    f"meta pred has {n_t_meta} trisomy signals; another batch has fewer "
                    f"(best n_t={int(best['n_t_disk'])})"
                )

        if reason is None:
            continue

        suspects.append(
            {
                "sample": sample,
                "label": label,
                "meta_pred": meta_pred,
                "meta_match": meta_match,
                "n_t_meta": n_t_meta,
                "ff_before_mq": g["ff_before_mq"].iloc[0],
                "score_batch": g.loc[g["is_score_batch"], "batch"].iloc[0]
                if g["is_score_batch"].any()
                else None,
                "good_batches": ";".join(good_batches),
                "good_preds": ";".join(good_preds),
                "all_disk": ";".join(
                    f"{row.batch}:{row.disk_pred}/{row.disk_match}/n_t="
                    f"{int(row.n_t_disk) if pd.notna(row.n_t_disk) else 'NA'}"
                    for row in g.drop_duplicates("batch").itertuples()
                ),
                "reason": reason,
            }
        )

    return scan_df, pd.DataFrame(suspects)


def audit_noisy_high_ff_samples(
    meta: pd.DataFrame,
    scan_df: pd.DataFrame,
    ff_before_min: float = 0.01,
    min_t_signals: int = 3,
) -> pd.DataFrame:
    """High-FF + many abnormal pred signals; classify single- vs multi-batch cause."""
    rows = []
    for r in meta.itertuples():
        n_t = n_trisomy_signals(r.pred_label)
        try:
            ff_b = float(r.ff_before_mq) if pd.notna(r.ff_before_mq) else None
        except (TypeError, ValueError):
            ff_b = None
        if ff_b is None or ff_b < ff_before_min or n_t < min_t_signals:
            continue
        n_batches = int(getattr(r, "n_batches", 0) or 0)
        scan_sub = (
            scan_df[scan_df["sample"] == r.sample]
            if scan_df is not None and not scan_df.empty
            else pd.DataFrame()
        )
        better = None
        all_disk = None
        if not scan_sub.empty:
            ok = scan_sub[scan_sub["scan_status"] == "ok"].drop_duplicates("batch")
            all_disk = ";".join(
                f"{row.batch}:{row.disk_pred}/n_t={int(row.n_t_disk)}"
                for row in ok.itertuples()
            )
            cleaner = ok[ok["n_t_disk"] < n_t]
            if len(cleaner):
                best = cleaner.sort_values("n_t_disk").iloc[0]
                better = (
                    f"{best['batch']}:{best['disk_pred']}(n_t={int(best['n_t_disk'])})"
                )

        if n_batches <= 1:
            category = "single_batch_noisy"
        elif better is not None:
            category = "multi_batch_wrong_selection_candidate"
        else:
            category = "multi_batch_all_noisy_or_unscanned"

        rows.append(
            {
                "sample": r.sample,
                "label": r.label,
                "pred_label": r.pred_label,
                "n_t": n_t,
                "ff_before_mq": ff_b,
                "ff_after_mq": r.ff_after_mq,
                "match_status": r.match_status,
                "n_batches": n_batches,
                "batches": getattr(r, "batches", None),
                "score_source": getattr(r, "score_source", None),
                "score_batch": getattr(r, "score_batch", None),
                "category": category,
                "better_batch_pred": better,
                "all_disk": all_disk,
            }
        )
    return pd.DataFrame(rows)


# Manual score-batch retargets applied after attach_score_source.
DEFAULT_SCORE_BATCH_OVERRIDES = {
    "JPTAY1417P8H1": "20260210-XML",
}


def apply_manual_score_batch_overrides(
    meta: pd.DataFrame,
    overrides: dict[str, str],
    pipeline_outputs: Iterable[str],
) -> pd.DataFrame:
    """Force selected score batch and refresh pred/FF from that batch's report."""
    out = meta.copy()
    for sample, batch in overrides.items():
        mask = out["sample"] == sample
        if not mask.any():
            continue
        scanned = scan_batch_pred(sample, batch, pipeline_outputs)
        out.loc[mask, "score_batch"] = batch
        out.loc[mask, "score_batch_key"] = normalize_batch_key(batch)
        out.loc[mask, "score_source"] = f"pipeline:{batch}"
        for root in pipeline_outputs:
            found = False
            for disk in disk_sample_candidates(sample):
                zpath = (
                    Path(root)
                    / batch
                    / "bwameth_results"
                    / "zscore_downstream"
                    / "beta_zscore"
                    / disk
                    / "beta_to_zscore"
                    / f"{disk}_zscore.tsv"
                )
                if zpath.is_file():
                    out.loc[mask, "episcore_file"] = str(zpath.resolve())
                    found = True
                    break
            if found:
                break
        if scanned and scanned.get("pred_label") is not None:
            out.loc[mask, "pred_label"] = scanned["pred_label"]
            idxs = out.index[mask]
            for idx in idxs:
                out.at[idx, "match_status"] = calc_match_status(
                    out.at[idx, "label"], out.at[idx, "pred_label"]
                )
        if scanned and scanned.get("ff_before_mq") is not None:
            out.loc[mask, "ff_before_mq"] = scanned["ff_before_mq"]
            out.loc[mask, "ff_after_mq"] = scanned["ff_after_mq"]
    return out


_MATCH_RANK = {"match": 0, "partially_match": 1, "mismatch": 2}


def _batch_quality_key(disk_match: Optional[str], n_t_disk: int, batch: str):
    """Lower is better: exact match, then fewer trisomy signals, then newer batch."""
    rank = _MATCH_RANK.get(disk_match, 3) if disk_match else 3
    return (rank, int(n_t_disk), -int(batch_date(batch) or 0), batch)


def infer_preferred_batches_from_scan(
    meta: pd.DataFrame,
    scan_df: pd.DataFrame,
    ff_before_max: Optional[float] = 0.01,
) -> dict[str, str]:
    """Infer good batch for multi-batch samples (default: ff_before_mq <= ``ff_before_max``).

    Only overrides when a scannable batch is **strictly better** than current meta:
    better match_status, or same match with fewer trisomy signals.
    """
    if scan_df is None or scan_df.empty:
        return {}
    out: dict[str, str] = {}
    focus = meta[meta["n_batches"] > 1].copy()
    if ff_before_max is not None:
        focus = focus[
            focus["ff_before_mq"].map(
                lambda v: pd.notna(v) and float(v) <= ff_before_max
            )
        ]
    for r in focus.itertuples():
        g = scan_df[
            (scan_df["sample"] == r.sample) & (scan_df["scan_status"] == "ok")
        ].drop_duplicates("batch")
        if g.empty:
            continue
        scored = []
        for row in g.itertuples():
            scored.append(
                (
                    _batch_quality_key(row.disk_match, int(row.n_t_disk or 0), row.batch),
                    row.batch,
                    row.disk_pred,
                    row.disk_match,
                    int(row.n_t_disk or 0),
                )
            )
        scored.sort(key=lambda x: x[0])
        best_batch, best_pred, best_match, best_n_t = (
            scored[0][1],
            scored[0][2],
            scored[0][3],
            scored[0][4],
        )

        meta_match = getattr(r, "match_status", None)
        meta_rank = _MATCH_RANK.get(meta_match, 3)
        best_rank = _MATCH_RANK.get(best_match, 3)
        meta_n_t = n_trisomy_signals(r.pred_label)
        current = normalize_batch_key(
            getattr(r, "preferred_batch_key", None)
        ) or normalize_batch_key(getattr(r, "score_batch", None))

        improved = False
        if best_rank < meta_rank:
            improved = True
        elif best_rank == meta_rank and best_n_t < meta_n_t:
            improved = True
        elif (
            best_match == "match"
            and not labels_equal(r.pred_label, best_pred)
            and meta_rank >= 1
        ):
            improved = True
        elif best_match == "match" and current != best_batch:
            # Align mqres/score path to the matching pipeline batch
            improved = True

        if not improved:
            continue
        if current == best_batch and labels_equal(r.pred_label, best_pred):
            continue
        out[r.sample] = best_batch
    return out


def apply_ff_precision(
    meta: pd.DataFrame,
    ff_dir: Path,
    precision: float = DEFAULT_FF_PRECISION,
    pipeline_outputs: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure FF columns at ``precision``.

    Preference order:
      1. ``ff_dir/{sample}_ff.tsv`` (central higher-precision store)
      2. pipeline ``estimate_ff_higher_precision`` / ``summary_report`` / report CSV
      3. existing meta values
    """
    out = meta.copy()
    report_rows = []
    decimals = max(0, int(round(-np.log10(precision))))
    pipeline_outputs = list(pipeline_outputs or [])

    for idx, r in out.iterrows():
        sample = r["sample"]
        raw_b, raw_a = r.get("ff_before_mq"), r.get("ff_after_mq")
        hp = load_ff_higher_precision(ff_dir, sample)
        source = "meta"
        source_path = None
        new_b, new_a = raw_b, raw_a
        if hp is not None:
            new_b, new_a = hp["ff_before_mq"], hp["ff_after_mq"]
            source = "ff_higher_precision"
            source_path = hp.get("source")
        elif pipeline_outputs:
            batches = []
            for b in (
                r.get("preferred_batch_key"),
                r.get("score_batch"),
                *(str(r.get("batch_keys") or "").split(";")),
            ):
                if b and str(b) != "nan" and b not in batches:
                    batches.append(str(b))
            found = find_ff_in_pipeline(sample, batches, pipeline_outputs)
            if found is not None:
                new_b, new_a = found["ff_before_mq"], found["ff_after_mq"]
                source = "pipeline_report"
                source_path = found.get("source")

        fmt_b, fmt_a = format_ff(new_b, precision), format_ff(new_a, precision)
        out.at[idx, "ff_before_mq"] = fmt_b
        out.at[idx, "ff_after_mq"] = fmt_a

        dp_b = ff_decimals_in_csv(raw_b)
        dp_a = ff_decimals_in_csv(raw_a)
        low_display = (dp_b is not None and dp_b < decimals) or (
            dp_a is not None and dp_a < decimals
        )

        def _diff(a, b):
            if a is None or b is None or (isinstance(a, float) and np.isnan(a)):
                return None
            try:
                return abs(float(a) - float(b))
            except (TypeError, ValueError):
                return None

        report_rows.append(
            {
                "sample": sample,
                "ff_source": source,
                "ff_source_path": source_path,
                "raw_ff_before_mq": raw_b,
                "raw_ff_after_mq": raw_a,
                "raw_decimals_before": dp_b,
                "raw_decimals_after": dp_a,
                "new_ff_before_mq": fmt_b,
                "new_ff_after_mq": fmt_a,
                "had_low_display_precision": low_display,
                "diff_before": _diff(raw_b, fmt_b),
                "diff_after": _diff(raw_a, fmt_a),
                "missing_higher_precision_file": source == "meta",
            }
        )

    return out, pd.DataFrame(report_rows)


def append_missing_score_batch_mqres(
    mqres: pd.DataFrame,
    meta: pd.DataFrame,
    pipeline_outputs: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add mqres rows for score_batch when that batch exists on disk but not in mqres.

    This repairs cases where episcore was re-run in a newer batch while
    ``mqres_samplesheet.csv`` still pointed at an older BAM.
    """
    generate_samplesheet = _load_generate_samplesheet()
    added_rows = []
    if generate_samplesheet is None:
        return mqres, pd.DataFrame(added_rows)

    need = []
    for r in meta.itertuples():
        sb = getattr(r, "score_batch", None)
        if not sb or (isinstance(sb, float) and np.isnan(sb)):
            continue
        sb = str(sb)
        sbk = normalize_batch_key(sb)
        existing_keys = {
            normalize_batch_key(b) for b in str(getattr(r, "batches", "")).split(";") if b
        }
        if sbk in existing_keys or sb in existing_keys:
            continue
        found = False
        for root in pipeline_outputs:
            if (Path(root) / sb / "bwameth_results" / "mq_downstream").is_dir():
                found = True
                break
        if found:
            need.append((r.sample, sb))

    if not need:
        return mqres, pd.DataFrame(added_rows)

    by_batch: dict[str, list[str]] = defaultdict(list)
    for sample, batch in need:
        by_batch[batch].append(sample)

    frames = [mqres]
    for batch, samples in by_batch.items():
        parse_dir = None
        for root in pipeline_outputs:
            if (Path(root) / batch / "bwameth_results" / "mq_downstream").is_dir():
                parse_dir = root
                break
        if parse_dir is None:
            continue
        new_df = generate_samplesheet(
            [batch],
            black_list=None,
            white_list=samples,
            parse_dir=parse_dir,
        )
        if new_df is None or new_df.empty:
            continue
        new_df = build_mqres_with_batches(new_df)
        frames.append(new_df)
        for _, row in new_df.iterrows():
            added_rows.append(
                {
                    "sample": row["sample"],
                    "batch": row["batch"],
                    "deconv_res": row["deconv_res"],
                    "clean_bam": row["clean_bam"],
                    "reason": "score_batch_missing_from_mqres",
                }
            )

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["sample", "clean_bam", "deconv_res"]).reset_index(
        drop=True
    )
    return out, pd.DataFrame(added_rows)


def _load_generate_samplesheet():
    """Import generate_samplesheet from notebooks tools if available."""
    candidates = [
        Path(__file__).resolve().parents[2]
        / "notebooks"
        / "aipt_1.0"
        / "tools",
        Path(__file__).resolve().parents[2]
        / "notebooks"
        / "240k_dev"
        / "tools",
    ]
    import sys

    for tools_dir in candidates:
        if not tools_dir.is_dir():
            continue
        sys.path.insert(0, str(tools_dir.parent))
        try:
            from tools.update_samplesheet import generate_samplesheet  # type: ignore

            return generate_samplesheet
        except Exception:
            try:
                sys.path.insert(0, str(tools_dir))
                import update_samplesheet as us  # type: ignore

                return us.generate_samplesheet
            except Exception:
                continue
    return None


def select_mqres_rows(
    mqres: pd.DataFrame,
    meta: pd.DataFrame,
    prefer_batches: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Mark selected mqres rows.

    Preference order per sample:
      1. ``prefer_batches`` override (e.g. better-performing batch from QA)
      2. ``score_batch_key`` from episcore path
      3. latest ``batch_key`` by date
    """
    prefer_batches = prefer_batches or {}
    score_map = {
        r.sample: normalize_batch_key(r.score_batch)
        for r in meta.itertuples()
        if getattr(r, "score_batch", None)
        and not (isinstance(r.score_batch, float) and np.isnan(r.score_batch))
    }
    latest_map = {}
    for sample, g in mqres.groupby("sample"):
        keys = [k for k in g["batch_key"].dropna().unique()]
        if keys:
            latest_map[sample] = sorted(keys, key=lambda k: (batch_date(k), k))[-1]

    out = mqres.copy()
    selected = []
    chosen = []
    for r in out.itertuples():
        prefer = (
            prefer_batches.get(r.sample)
            or score_map.get(r.sample)
            or latest_map.get(r.sample)
        )
        chosen.append(prefer)
        ok = prefer is not None and r.batch_key == prefer
        selected.append(bool(ok))
    out["selected"] = selected
    out["selected_batch_key"] = chosen

    for sample, g in out.groupby("sample"):
        if not g["selected"].any():
            out.loc[g.index, "selected"] = True

    return out.sort_values(
        ["sample", "selected", "batch_key", "is_single_end", "clean_bam", "deconv_res"],
        ascending=[True, False, True, True, True, True],
    ).reset_index(drop=True)


def prefer_batches_from_suspects(suspects: pd.DataFrame) -> dict[str, str]:
    """Pick the latest good_batch for each suspect sample as mqres selection override."""
    out = {}
    if suspects is None or suspects.empty:
        return out
    for r in suspects.itertuples():
        goods = [b for b in str(getattr(r, "good_batches", "")).split(";") if b]
        if goods:
            out[r.sample] = sorted(goods, key=lambda k: (batch_date(k), k))[-1]
    return out


def refresh_meta_from_preferred_batches(
    meta: pd.DataFrame,
    prefer: dict[str, str],
    pipeline_outputs: Iterable[str],
    force: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rewrite meta scores / pred / FF / episcore path from preferred batch reports."""
    out = meta.copy()
    changes = []
    score_cols = (
        "pred_label",
        "beta_zscores",
        "rc_zscores",
        "final_zscores",
        "mean_target_coverage",
        "snp_mean_coverage",
        "cpg_mean_coverage",
        "ff_before_mq",
        "ff_after_mq",
    )
    for sample, batch in prefer.items():
        if not batch:
            continue
        mask = out["sample"] == sample
        if not mask.any():
            continue
        scanned = scan_batch_pred(sample, batch, pipeline_outputs)
        if not scanned or scanned.get("status") not in {"ok", "partial"}:
            continue
        if scanned.get("pred_label") is None and not force:
            continue
        idx = out.index[mask][0]
        old_pred = out.at[idx, "pred_label"]
        new_pred = scanned.get("pred_label", old_pred)
        changed_fields = []
        if new_pred is not None and (force or not labels_equal(old_pred, new_pred)):
            if not labels_equal(old_pred, new_pred):
                changed_fields.append("pred_label")
            out.at[idx, "pred_label"] = new_pred
            out.at[idx, "match_status"] = calc_match_status(out.at[idx, "label"], new_pred)
            if "conservative_match_status" in out.columns:
                # lightweight: reuse match for now when refreshing
                out.at[idx, "conservative_match_status"] = out.at[idx, "match_status"]

        for col in score_cols:
            if col == "pred_label":
                continue
            val = scanned.get(col)
            if val is None:
                continue
            old = out.at[idx, col] if col in out.columns else None
            out.at[idx, col] = val
            if old != val:
                changed_fields.append(col)

        out.at[idx, "score_batch"] = batch
        out.at[idx, "score_batch_key"] = normalize_batch_key(batch)
        out.at[idx, "score_source"] = f"pipeline:{batch}"
        for root in pipeline_outputs:
            found = False
            for disk in disk_sample_candidates(sample):
                zpath = (
                    Path(root)
                    / batch
                    / "bwameth_results"
                    / "zscore_downstream"
                    / "beta_zscore"
                    / disk
                    / "beta_to_zscore"
                    / f"{disk}_zscore.tsv"
                )
                if zpath.is_file():
                    out.at[idx, "episcore_file"] = str(zpath.resolve())
                    found = True
                    break
            if found:
                break
        if changed_fields or force:
            changes.append(
                {
                    "sample": sample,
                    "preferred_batch": batch,
                    "old_pred": old_pred,
                    "new_pred": new_pred,
                    "old_n_t": n_trisomy_signals(old_pred),
                    "new_n_t": n_trisomy_signals(new_pred),
                    "new_match_status": out.at[idx, "match_status"],
                    "changed_fields": ",".join(sorted(set(changed_fields))) or "score_batch",
                }
            )
    return out, pd.DataFrame(changes)


def write_ff_fixed_csv(df: pd.DataFrame, path: Path, precision: float = DEFAULT_FF_PRECISION):
    """Write CSV with FF columns formatted to fixed decimal places."""
    decimals = max(0, int(round(-np.log10(precision))))
    out = df.copy()

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return v

    for col in ("ff_before_mq", "ff_after_mq"):
        if col in out.columns:
            out[col] = out[col].map(fmt)
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--mqres", default=DEFAULT_MQRES)
    p.add_argument("--episcore", default=DEFAULT_EPISCORE)
    p.add_argument("--ff-dir", default=DEFAULT_FF_DIR)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument(
        "--pipeline-output",
        action="append",
        default=None,
        help="Pipeline output root (repeatable). Defaults to lustre + appsnew mirrors.",
    )
    p.add_argument(
        "--cutoff",
        default=None,
        help=(
            "Optional YYYYMMDD: only audit multi-batch samples with any batch on/after "
            "this date. Default: audit ALL multi-batch samples."
        ),
    )
    p.add_argument("--ff-precision", type=float, default=DEFAULT_FF_PRECISION)
    p.add_argument(
        "--skip-multibatch-scan",
        action="store_true",
        help="Skip on-disk zscore scanning (faster; still does path-level checks).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pipeline_outputs = args.pipeline_output or list(DEFAULT_PIPELINE_OUTPUTS)

    meta = pd.read_csv(args.meta)
    meta["sample"] = meta["sample"].map(canonical_sample_id)
    mqres_raw = pd.read_csv(args.mqres)
    episcore = pd.read_csv(args.episcore)

    mqres = build_mqres_with_batches(mqres_raw)
    batch_map = sample_batch_map(mqres)
    meta = meta.merge(batch_map, on="sample", how="left")
    meta["batches"] = meta["batches"].fillna("")
    meta["batch_keys"] = meta["batch_keys"].fillna("")
    meta["n_batches"] = meta["n_batches"].fillna(0).astype(int)
    meta["n_mqres_rows"] = meta["n_mqres_rows"].fillna(0).astype(int)

    meta = attach_score_source(meta, episcore)
    meta = apply_manual_score_batch_overrides(
        meta, DEFAULT_SCORE_BATCH_OVERRIDES, pipeline_outputs
    )

    inconsistency = flag_batch_score_inconsistencies(meta)

    if args.skip_multibatch_scan:
        scan_df, suspects = pd.DataFrame(), pd.DataFrame()
        noisy = pd.DataFrame()
        low_ff_prefer: dict = {}
    else:
        scan_df, suspects = audit_multibatch_selection(
            meta, pipeline_outputs, cutoff=args.cutoff
        )
        noisy = audit_noisy_high_ff_samples(
            meta, scan_df, ff_before_min=0.01, min_t_signals=2
        )

    # Fill mqres rows for score batches that were missing from the input sheet
    mqres, added_mqres = append_missing_score_batch_mqres(
        mqres, meta, pipeline_outputs
    )
    if not added_mqres.empty:
        # refresh batch map + meta batches after additions
        batch_map = sample_batch_map(mqres)
        drop_cols = [c for c in ("batches", "batch_keys", "n_batches", "n_mqres_rows") if c in meta.columns]
        meta = meta.drop(columns=drop_cols).merge(batch_map, on="sample", how="left")
        meta["batches"] = meta["batches"].fillna("")
        meta["batch_keys"] = meta["batch_keys"].fillna("")
        meta["n_batches"] = meta["n_batches"].fillna(0).astype(int)
        meta["n_mqres_rows"] = meta["n_mqres_rows"].fillna(0).astype(int)
        inconsistency = flag_batch_score_inconsistencies(meta)

    prefer = prefer_batches_from_suspects(suspects)
    prefer.update(
        {s: normalize_batch_key(b) for s, b in DEFAULT_SCORE_BATCH_OVERRIDES.items()}
    )
    # Low-FF multi-batch: infer good vs bad batch and replace scores/ff/mqres
    low_ff_prefer = infer_preferred_batches_from_scan(
        meta, scan_df, ff_before_max=0.01
    )
    prefer.update(low_ff_prefer)

    meta, pred_updates = refresh_meta_from_preferred_batches(
        meta, prefer, pipeline_outputs, force=True
    )

    meta, ff_report = apply_ff_precision(
        meta,
        Path(args.ff_dir),
        precision=args.ff_precision,
        pipeline_outputs=pipeline_outputs,
    )

    mqres_out = select_mqres_rows(mqres, meta, prefer_batches=prefer)

    meta["preferred_batch_key"] = [
        prefer.get(s)
        or (normalize_batch_key(sb) if pd.notna(sb) else None)
        for s, sb in zip(meta["sample"], meta["score_batch"])
    ]

    # Column order: keep original meta cols, append new
    original_cols = [
        c
        for c in pd.read_csv(args.meta, nrows=0).columns
        if c in meta.columns
    ]
    extra_cols = [
        c
        for c in [
            "batches",
            "batch_keys",
            "n_batches",
            "n_mqres_rows",
            "score_source",
            "score_batch",
            "score_batch_key",
            "preferred_batch_key",
            "episcore_file",
        ]
        if c in meta.columns
    ]
    meta_out = meta[original_cols + extra_cols]

    meta_path = outdir / "meta_samplesheet.csv"
    mqres_path = outdir / "mqres_samplesheet.csv"
    write_ff_fixed_csv(meta_out, meta_path, precision=args.ff_precision)
    mqres_out.to_csv(mqres_path, index=False)

    inconsistency.to_csv(outdir / "report_batch_score_inconsistency.csv", index=False)
    suspects.to_csv(outdir / "report_multibatch_wrong_selection.csv", index=False)
    if not scan_df.empty:
        scan_df.to_csv(outdir / "report_multibatch_scan_detail.csv", index=False)
    if not noisy.empty:
        noisy.to_csv(outdir / "report_noisy_high_ff.csv", index=False)
    if not pred_updates.empty:
        pred_updates.to_csv(outdir / "report_pred_label_updates.csv", index=False)
    ff_report.to_csv(outdir / "report_ff_precision.csv", index=False)
    if not added_mqres.empty:
        added_mqres.to_csv(outdir / "report_mqres_rows_added.csv", index=False)

    summary = {
        "n_meta_samples": int(len(meta_out)),
        "n_mqres_rows": int(len(mqres_out)),
        "n_mqres_selected": int(mqres_out["selected"].sum()),
        "n_mqres_rows_added_from_score_batch": int(len(added_mqres)),
        "n_batches_distribution": meta_out["n_batches"].value_counts().sort_index().to_dict(),
        "score_source_counts": meta_out["score_source"].value_counts().to_dict(),
        "n_score_batch_not_in_mqres": int(len(inconsistency)),
        "n_multibatch_samples_scanned": int((meta_out["n_batches"] > 1).sum()),
        "n_suspect_wrong_batch": int(len(suspects)),
        "suspect_preferred_overrides": prefer,
        "n_noisy_high_ff": int(len(noisy)),
        "noisy_high_ff_category_counts": (
            noisy["category"].value_counts().to_dict() if not noisy.empty else {}
        ),
        "n_pred_label_updates": int(len(pred_updates)),
        "n_low_ff_multibatch_overrides": int(len(low_ff_prefer)),
        "low_ff_multibatch_overrides": low_ff_prefer,
        "manual_score_batch_overrides": DEFAULT_SCORE_BATCH_OVERRIDES,
        "n_missing_ff_higher_precision": int(ff_report["missing_higher_precision_file"].sum()),
        "ff_source_counts": ff_report["ff_source"].value_counts().to_dict(),
        "n_ff_low_display_precision": int(ff_report["had_low_display_precision"].sum()),
        "cutoff": args.cutoff,
        "ff_precision": args.ff_precision,
        "outputs": {
            "meta": str(meta_path),
            "mqres": str(mqres_path),
            "inconsistency": str(outdir / "report_batch_score_inconsistency.csv"),
            "wrong_selection": str(outdir / "report_multibatch_wrong_selection.csv"),
            "noisy_high_ff": str(outdir / "report_noisy_high_ff.csv"),
            "ff": str(outdir / "report_ff_precision.csv"),
        },
    }
    with open(outdir / "report_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
