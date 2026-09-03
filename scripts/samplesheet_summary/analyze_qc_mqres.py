#!/usr/bin/env python3
"""Map DB conversion-QC table to mqres clean_bams and test puc19 vs trisomy match.

Reads:
  - DB ``质控数据生信分析结果`` via ``tools.db_helper``
  - ``meta_samplesheet.csv`` / ``mqres_samplesheet.csv`` under --outdir

Mapping policy (per sample):
  - 1 clean_bam + 1 QC → direct map (ignore date gap)
  - N clean_bam + 1 QC → all bams → that QC
  - multi QC → nearest QC by calendar date (no max window)

Writes under --outdir:
  - qc_bioinfo_table.csv
  - qc_bam_map.csv              (one row per clean_bam)
  - qc_mqres_batch_map.csv      (alias of bam map)
  - qc_samples_missing_from_db.csv
  - qc_puc19_performance.csv
  - qc_batch_summary.csv
  - qc_analysis_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOOLS_CANDIDATES = [
    ROOT / "notebooks" / "aipt_1.0",
    ROOT / "notebooks" / "240k_dev",
]
for tools_parent in TOOLS_CANDIDATES:
    if (tools_parent / "tools" / "db_helper.py").is_file():
        sys.path.insert(0, str(tools_parent))
        break

from tools.db_helper import (  # noqa: E402
    extract_batch_date,
    read_qc_bioinfo_df,
)

DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")
DEFAULT_MAX_DAY_DELTA = 7
PUC19_FAIL_LT = 94.0
LAMBDA_FAIL_GT = 1.0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--mqres", type=Path, default=None)
    p.add_argument("--max-day-delta", type=int, default=DEFAULT_MAX_DAY_DELTA)
    p.add_argument(
        "--qc-csv",
        type=Path,
        default=None,
        help="Optional cached QC CSV (skip DB). Must have sample/lambda/puc19/dataset.",
    )
    return p.parse_args(argv)


def _norm_yyyymmdd(date_val) -> Optional[str]:
    """Normalize CSV/DB dates to ``YYYYMMDD`` (handles float ``20241219.0``)."""
    if date_val is None or (isinstance(date_val, float) and np.isnan(date_val)):
        return None
    s = str(date_val).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    # float-like without trailing .0 already handled; also int
    if s.isdigit() and len(s) == 8:
        return s
    extracted = extract_batch_date(s)
    return extracted if extracted else None


def _date_to_ord(date_str: Optional[str]) -> Optional[int]:
    s = _norm_yyyymmdd(date_str)
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").toordinal()
    except ValueError:
        return None



def build_mqres_bam_level(mqres: pd.DataFrame) -> pd.DataFrame:
    """One row per unique ``clean_bam`` (typically 2 deconv_res share one bam)."""
    df = mqres.copy()
    if "batch_key" not in df.columns:
        df["batch_key"] = df.get("batch")
    df["batch_key"] = df["batch_key"].astype(str)
    df["batch_date"] = df["batch_key"].map(extract_batch_date).map(_norm_yyyymmdd)
    if "selected" in df.columns:
        df["selected"] = df["selected"].astype(bool)
    else:
        df["selected"] = True
    batch_src = "batch" if "batch" in df.columns else "batch_key"
    g = (
        df.groupby(["sample", "clean_bam"], as_index=False)
        .agg(
            batch_key=("batch_key", "first"),
            batch_date=("batch_date", "first"),
            batch_raw=(batch_src, "first"),
            n_deconv=("deconv_res", "size"),
            selected=("selected", "any"),
        )
    )
    # per-sample bam count for mapping policy
    g["n_bam_for_sample"] = g.groupby("sample")["clean_bam"].transform("nunique")
    return g


# keep alias for any external callers
def build_mqres_batch_level(mqres: pd.DataFrame) -> pd.DataFrame:
    return build_mqres_bam_level(mqres)


def _hs_swap(sample: str) -> Optional[str]:
    """Swap trailing H1 ↔ S1 when present."""
    if sample.endswith("H1"):
        return sample[:-2] + "S1"
    if sample.endswith("S1"):
        return sample[:-2] + "H1"
    return None


def _qc_rows_for_sample(
    sample: str, qc_by_sample: dict[str, pd.DataFrame]
) -> tuple[Optional[pd.DataFrame], Optional[str], str]:
    """Return (qc_rows, matched_qc_sample, match_note)."""
    if sample in qc_by_sample:
        return qc_by_sample[sample], sample, "exact_sample"
    alt = _hs_swap(sample)
    if alt and alt in qc_by_sample:
        return qc_by_sample[alt], alt, "hs_swap"
    return None, None, "missing"


def _prefer_qc_row(df: pd.DataFrame) -> pd.Series:
    """Prefer ``*-XML`` dataset names when several QC rows tie."""
    tmp = df.copy()
    tmp["_score"] = tmp["dataset"].map(
        lambda d: (
            0
            if str(d).endswith("-XML")
            else (1 if "-XML" in str(d) else 2)
        )
    )
    return tmp.sort_values(["_score", "dataset"]).iloc[0]


def _unique_qc_datasets(qsub: pd.DataFrame) -> pd.DataFrame:
    """One row per QC dataset (first / preferred)."""
    if qsub.empty:
        return qsub
    rows = []
    for _, g in qsub.groupby("dataset", dropna=False):
        rows.append(_prefer_qc_row(g))
    return pd.DataFrame(rows).reset_index(drop=True)


def _nearest_qc_row(qsub: pd.DataFrame, mq_ord: Optional[int]) -> tuple[pd.Series, Optional[int]]:
    """Pick QC row with smallest |Δdays|; ties broken by preferred dataset name."""
    candidates = []
    for _, q in qsub.iterrows():
        q_ord = _date_to_ord(q["dataset_date"])
        if mq_ord is None or q_ord is None:
            delta = 10**9
        else:
            delta = abs(q_ord - mq_ord)
        ds = str(q["dataset"])
        score = 0 if ds.endswith("-XML") else (1 if "-XML" in ds else 2)
        candidates.append((delta, score, ds, q))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    delta, _, _, hit = candidates[0]
    return hit, (None if delta >= 10**9 else int(delta))


def map_bam_to_qc(mqres_bams: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """Map each ``clean_bam`` to one QC result.

    Policy (per sample):
      - 1 bam + 1 QC dataset → direct map (ignore date gap)
      - N bam + 1 QC → all bams map to that QC
      - otherwise (multi QC) → nearest QC by calendar date (no max window)
    """
    qc = qc.copy()
    if "dataset_date" not in qc.columns:
        qc["dataset_date"] = None
    qc["dataset_date"] = qc["dataset_date"].map(_norm_yyyymmdd)
    from_ds = qc["dataset"].map(extract_batch_date)
    qc["dataset_date"] = qc["dataset_date"].fillna(from_ds).map(_norm_yyyymmdd)
    qc_by_sample: dict[str, pd.DataFrame] = {
        s: g.reset_index(drop=True) for s, g in qc.groupby("sample")
    }

    rows = []
    for r in mqres_bams.itertuples():
        sample = r.sample
        mq_date = _norm_yyyymmdd(r.batch_date)
        mq_ord = _date_to_ord(mq_date)
        qsub_raw, qc_sample, sample_match = _qc_rows_for_sample(sample, qc_by_sample)
        base = {
            "sample": sample,
            "clean_bam": r.clean_bam,
            "qc_sample": qc_sample,
            "sample_match": sample_match,
            "mqres_batch_key": r.batch_key,
            "mqres_batch_date": mq_date,
            "mqres_selected": bool(r.selected),
            "n_deconv": int(r.n_deconv),
            "n_bam_for_sample": int(r.n_bam_for_sample),
        }
        if qsub_raw is None or qsub_raw.empty:
            rows.append(
                {
                    **base,
                    "n_qc_for_sample": 0,
                    "map_status": "sample_not_in_qc",
                    "map_rule": "missing",
                    "qc_dataset": None,
                    "qc_dataset_date": None,
                    "day_delta": None,
                    "lambda": None,
                    "puc19": None,
                }
            )
            continue

        qsub = _unique_qc_datasets(qsub_raw)
        n_qc = int(len(qsub))
        n_bam = int(r.n_bam_for_sample)
        base["n_qc_for_sample"] = n_qc

        if n_qc == 1:
            hit = qsub.iloc[0]
            day_delta = None
            if mq_ord is not None:
                q_ord = _date_to_ord(hit["dataset_date"])
                if q_ord is not None:
                    day_delta = abs(q_ord - mq_ord)
            rule = "direct_singleton" if n_bam == 1 else "direct_single_qc"
            rows.append(
                {
                    **base,
                    "map_status": rule,
                    "map_rule": rule,
                    "qc_dataset": hit["dataset"],
                    "qc_dataset_date": _norm_yyyymmdd(hit["dataset_date"]),
                    "day_delta": day_delta,
                    "lambda": hit["lambda"],
                    "puc19": hit["puc19"],
                }
            )
            continue

        # multi QC → nearest by date (no window)
        hit, day_delta = _nearest_qc_row(qsub, mq_ord)
        rows.append(
            {
                **base,
                "map_status": "nearest_date",
                "map_rule": "nearest_date",
                "qc_dataset": hit["dataset"],
                "qc_dataset_date": _norm_yyyymmdd(hit["dataset_date"]),
                "day_delta": day_delta,
                "lambda": hit["lambda"],
                "puc19": hit["puc19"],
            }
        )
    return pd.DataFrame(rows)


def map_mqres_to_qc(
    mqres_batches: pd.DataFrame,
    qc: pd.DataFrame,
    max_day_delta: int = DEFAULT_MAX_DAY_DELTA,
) -> pd.DataFrame:
    """Backward-compatible wrapper → :func:`map_bam_to_qc` (``max_day_delta`` unused)."""
    del max_day_delta
    return map_bam_to_qc(mqres_batches, qc)


def qc_fail_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["puc19_fail"] = out["puc19"].map(
        lambda v: bool(pd.notna(v) and float(v) < PUC19_FAIL_LT)
    )
    out["lambda_fail"] = out["lambda"].map(
        lambda v: bool(pd.notna(v) and float(v) > LAMBDA_FAIL_GT)
    )
    out["qc_fail"] = out["puc19_fail"] | out["lambda_fail"]
    return out



def _n_t_signals(pred) -> int:
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return 0
    import re
    return sum(
        1 for p in str(pred).split(",")
        if re.match(r"^(?:Gray_)?T\d+", p.strip())
    )


def analyze_puc19_vs_performance(
    mapped: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Join selected mqres-QC rows to meta; also summarize all-batch QC effects."""
    meta_cols = [
        c for c in (
            "sample", "label", "pred_label", "match_status",
            "conservative_match_status", "set", "ff_before_mq",
            "preferred_batch_key", "score_batch",
        ) if c in meta.columns
    ]
    m = meta[meta_cols].drop_duplicates("sample")

    sel = mapped[mapped["mqres_selected"]].copy()
    sel = sel.sort_values(
        ["sample", "day_delta"], ascending=[True, True], na_position="last"
    ).drop_duplicates("sample", keep="first")

    joined = sel.merge(m, on="sample", how="left")
    joined = qc_fail_flags(joined)
    joined["n_t_pred"] = joined["pred_label"].map(_n_t_signals)
    joined["noisy_pred"] = joined["n_t_pred"] >= 3
    joined["evaluable"] = joined["match_status"].isin(
        ["match", "partially_match", "mismatch"]
    )
    joined["is_mismatch"] = joined["match_status"] == "mismatch"
    joined["is_not_match"] = joined["match_status"].isin(
        ["mismatch", "partially_match"]
    )

    def _match_rate(sub: pd.DataFrame, col: str) -> dict:
        ev = sub[sub["evaluable"]]
        n = len(ev)
        if n == 0:
            return {"n_evaluable": 0, "n_pos": 0, "rate": None}
        n_pos = int(ev[col].sum())
        return {"n_evaluable": n, "n_pos": n_pos, "rate": n_pos / n}

    def _flag_rate(sub: pd.DataFrame, col: str) -> dict:
        n = len(sub)
        if n == 0:
            return {"n": 0, "n_pos": 0, "rate": None}
        n_pos = int(sub[col].sum())
        return {"n": n, "n_pos": n_pos, "rate": n_pos / n}

    puc_fail = joined[joined["puc19_fail"]]
    puc_pass = joined[~joined["puc19_fail"] & joined["puc19"].notna()]

    summary = {
        "thresholds": {
            "puc19_fail_lt": PUC19_FAIL_LT,
            "lambda_fail_gt": LAMBDA_FAIL_GT,
            "noisy_pred_min_t_signals": 3,
        },
        "n_selected_mapped": int(len(joined)),
        "n_with_qc": int(joined["puc19"].notna().sum()),
        "n_puc19_fail": int(joined["puc19_fail"].sum()),
        "n_lambda_fail": int(joined["lambda_fail"].sum()),
        "n_qc_fail": int(joined["qc_fail"].sum()),
        "mismatch_rate_puc19_fail": _match_rate(puc_fail, "is_mismatch"),
        "mismatch_rate_puc19_pass": _match_rate(puc_pass, "is_mismatch"),
        "not_match_rate_puc19_fail": _match_rate(puc_fail, "is_not_match"),
        "not_match_rate_puc19_pass": _match_rate(puc_pass, "is_not_match"),
        "noisy_pred_rate_puc19_fail": _flag_rate(
            puc_fail[puc_fail["pred_label"].notna()], "noisy_pred"
        ),
        "noisy_pred_rate_puc19_pass": _flag_rate(
            puc_pass[puc_pass["pred_label"].notna()], "noisy_pred"
        ),
        "mismatch_rate_qc_fail": _match_rate(joined[joined["qc_fail"]], "is_mismatch"),
        "mismatch_rate_qc_pass": _match_rate(
            joined[~joined["qc_fail"] & joined["puc19"].notna()], "is_mismatch"
        ),
    }

    # All mapped rows for batch-level stats
    all_j = mapped.merge(m, on="sample", how="left")
    all_j = qc_fail_flags(all_j)
    all_j["n_t_pred"] = all_j["pred_label"].map(_n_t_signals)
    all_j["noisy_pred"] = all_j["n_t_pred"] >= 3
    all_j["evaluable"] = all_j["match_status"].isin(
        ["match", "partially_match", "mismatch"]
    )
    all_j["is_mismatch"] = all_j["match_status"] == "mismatch"

    batch_rows = []
    for key_col, name in (
        ("qc_dataset_date", "qc_date"),
        ("mqres_batch_key", "mqres_batch"),
    ):
        for key, g in all_j.groupby(key_col):
            if key is None or (isinstance(key, float) and np.isnan(key)):
                continue
            ev = g[g["evaluable"]]
            batch_rows.append({
                "group_type": name,
                "group": key,
                "n_samples": int(g["sample"].nunique()),
                "n_rows": int(len(g)),
                "n_evaluable": int(len(ev)),
                "n_puc19_fail": int(g["puc19_fail"].sum()),
                "n_lambda_fail": int(g["lambda_fail"].sum()),
                "median_puc19": float(g["puc19"].median()) if g["puc19"].notna().any() else None,
                "median_lambda": float(g["lambda"].median()) if g["lambda"].notna().any() else None,
                "n_mismatch": int(ev["is_mismatch"].sum()) if len(ev) else 0,
                "mismatch_rate": float(ev["is_mismatch"].mean()) if len(ev) else None,
                "n_noisy_pred": int(g["noisy_pred"].sum()),
                "noisy_pred_rate": float(g["noisy_pred"].mean()) if len(g) else None,
            })
    batch_summary = pd.DataFrame(batch_rows).sort_values(
        ["group_type", "n_puc19_fail", "mismatch_rate"],
        ascending=[True, False, False],
    )

    focus = all_j[
        all_j["mqres_batch_key"].astype(str).str.contains("20260602", na=False)
    ].copy()
    focus_u = focus.drop_duplicates("sample")
    summary["batch_20260602"] = {
        "n_samples": int(focus_u["sample"].nunique()),
        "n_rows": int(len(focus)),
        "median_puc19": float(focus["puc19"].median()) if focus["puc19"].notna().any() else None,
        "n_puc19_fail": int(focus["puc19_fail"].sum()),
        "mismatch": _match_rate(focus_u, "is_mismatch"),
        "noisy_pred": _flag_rate(focus_u[focus_u["pred_label"].notna()], "noisy_pred"),
        "match_status_counts": focus_u["match_status"].value_counts(dropna=False).to_dict(),
        "pred_label_top": focus_u["pred_label"].value_counts(dropna=False).head(8).to_dict(),
    }
    return joined, batch_summary, summary


def write_markdown(
    path: Path,
    map_df: pd.DataFrame,
    perf: pd.DataFrame,
    batch_summary: pd.DataFrame,
    summary: dict,
    max_day_delta: int,
) -> None:
    def pct(rate):
        if rate is None or (isinstance(rate, float) and np.isnan(rate)):
            return "n/a"
        return f"{100 * rate:.1f}%"

    mf = summary["mismatch_rate_puc19_fail"]
    mp = summary["mismatch_rate_puc19_pass"]
    nf = summary["not_match_rate_puc19_fail"]
    np_ = summary["not_match_rate_puc19_pass"]
    noisy_f = summary["noisy_pred_rate_puc19_fail"]
    noisy_p = summary["noisy_pred_rate_puc19_pass"]
    b26 = summary["batch_20260602"]

    map_counts = map_df["map_status"].value_counts().to_dict()
    rule_counts = (
        map_df["map_rule"].value_counts().to_dict()
        if "map_rule" in map_df.columns
        else map_counts
    )
    n_bam = len(map_df)
    n_mapped = int(map_df["puc19"].notna().sum()) if "puc19" in map_df.columns else 0
    n_multi = int((map_df.groupby("sample")["clean_bam"].nunique() > 1).sum()) if "clean_bam" in map_df.columns else int((map_df.groupby("sample")["mqres_batch_key"].nunique() > 1).sum())
    nearest = map_df[map_df["map_status"] == "nearest_date"]
    not_in_qc = sorted(
        map_df.loc[map_df["map_status"] == "sample_not_in_qc", "sample"].unique()
    )
    hs_swap_n = (
        int((map_df["sample_match"] == "hs_swap").sum())
        if "sample_match" in map_df.columns
        else 0
    )
    sing = map_df[map_df.get("map_rule", pd.Series(dtype=str)) == "direct_singleton"] if "map_rule" in map_df.columns else map_df.iloc[0:0]

    top_fail = batch_summary[
        (batch_summary["group_type"] == "mqres_batch")
        & (batch_summary["n_puc19_fail"] > 0)
    ].head(12)

    lines = [
        "# QC ↔ mqres mapping and pUC19 vs trisomy performance",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Data sources",
        "",
        "- DB `质控数据生信分析结果` → `sample`, `lambda`, `puc19`, `dataset`",
        "- Sample cleaning: `_QC` / `QC-` / `EM` / `UMBS` / `erro_`; `HCPT####P`→`HCPT####`; B→P; drop names with `V`; H1↔S1 fallback",
        "- `mqres_samplesheet.csv`: typically 2 `deconv_res` per `clean_bam` → map at **bam** level",
        "- `meta_samplesheet.csv` (`label` / `pred_label` / `match_status`)",
        "",
        "## QC fail rules",
        "",
        f"- **Fail** if `puc19 < {PUC19_FAIL_LT}%` **or** `lambda > {LAMBDA_FAIL_GT}%`",
        "",
        "## 1. BAM ↔ QC mapping",
        "",
        "Policy **per sample** (no ±7-day cutoff):",
        "",
        "1. **1 clean_bam + 1 QC** → map directly (e.g. `PTAY1454P7S1`: mqres `20260618` ↔ QC `20260418`)",
        "2. **N clean_bam + 1 QC** → all bams map to that QC",
        "3. **Multi QC** → each bam maps to nearest QC by calendar `|Δdays|`",
        "",
        f"- Unique `clean_bam`: **{n_bam}** (mapped with QC: **{n_mapped}**, unmapped: **{n_bam - n_mapped}**)",
        f"- Multi-bam samples: **{n_multi}**",
        f"- H1↔S1 fallback rows: **{hs_swap_n}**",
        "- Map rule counts:",
    ]
    for k, v in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{k}`: {v}")
    if len(nearest) and nearest["day_delta"].notna().any():
        lines.append(
            f"- Nearest-date `|Δdays|`: median **{nearest['day_delta'].median():.0f}**, "
            f"max **{int(nearest['day_delta'].max())}**"
        )
    if len(sing) and sing["day_delta"].notna().any():
        lines.append(
            f"- Direct singleton `|Δdays|` (informational): median **{sing['day_delta'].median():.0f}**, "
            f"max **{int(sing['day_delta'].max())}**, "
            f">7d **{int((sing['day_delta'] > 7).sum())}**, "
            f">30d **{int((sing['day_delta'] > 30).sum())}**"
        )
    lines += [
        "",
        f"### Samples with **no QC row in DB** ({len(not_in_qc)})",
        "",
    ]
    if not_in_qc:
        lines.append(
            "Absent from DB after name cleaning. See `qc_samples_missing_from_db.csv` / `qc_bams_unmapped.csv`."
        )
        lines.append("")
        for s in not_in_qc:
            lines.append(f"- `{s}`")
    else:
        lines.append("_None._")
    lines += [
        "",
        "Primary output: **`qc_bam_map.csv`** (one row per `clean_bam`).",
        "",
        "## 2. Does low pUC19 lead to bad trisomy detection?",
        "",
        "Performance join uses the **selected** clean_bam’s QC values.",
        "",
        "### Against ground-truth `match_status`",
        "",
        "| Group | N evaluable | Mismatch rate | Not-match (mismatch+partial) |",
        "|-------|------------:|--------------:|-----------------------------:|",
        f"| pUC19 fail (<{PUC19_FAIL_LT}%) | {mf['n_evaluable']} | {pct(mf['rate'])} ({mf['n_pos']}) | {pct(nf['rate'])} ({nf['n_pos']}) |",
        f"| pUC19 pass | {mp['n_evaluable']} | {pct(mp['rate'])} ({mp['n_pos']}) | {pct(np_['rate'])} ({np_['n_pos']}) |",
        "",
        "### Noisy-prediction proxy (≥3 T/Gray signals)",
        "",
        "Most pUC19-fail rows are `Unknown` label (esp. JPTAY emergency), so `match_status` is missing. "
        "As a proxy for “bad performance”, count preds with ≥3 trisomy/gray signals:",
        "",
        "| Group | N | Noisy-pred rate |",
        "|-------|--:|----------------:|",
        f"| pUC19 fail | {noisy_f['n']} | {pct(noisy_f['rate'])} ({noisy_f['n_pos']}) |",
        f"| pUC19 pass | {noisy_p['n']} | {pct(noisy_p['rate'])} ({noisy_p['n_pos']}) |",
        "",
    ]

    # Verdict
    if mf["rate"] is not None and mp["rate"] is not None and mf["n_evaluable"] >= 5:
        if mf["rate"] > mp["rate"] + 0.05:
            verdict = "Yes — higher mismatch among pUC19-fail (evaluable labels)."
        elif mf["rate"] < mp["rate"] - 0.05:
            verdict = "No — pUC19-fail does **not** show higher mismatch than pass."
        else:
            verdict = "Unclear — similar mismatch rates (<5 pp difference)."
    else:
        verdict = (
            "**No strong evidence that low pUC19 drives label↔pred mismatch.** "
            f"Only {mf['n_evaluable']} pUC19-fail samples have evaluable labels "
            f"(mismatch rate {pct(mf['rate'])}), vs {mp['n_evaluable']} pass "
            f"({pct(mp['rate'])}). Many fails are Unknown/emergency."
        )
        if noisy_f["rate"] is not None and noisy_p["rate"] is not None:
            if noisy_f["rate"] <= noisy_p["rate"] + 0.05:
                verdict += (
                    f" Noisy-pred rates are also not elevated "
                    f"({pct(noisy_f['rate'])} fail vs {pct(noisy_p['rate'])} pass)."
                )
            else:
                verdict += (
                    f" Noisy-pred rate is higher on fail "
                    f"({pct(noisy_f['rate'])} vs {pct(noisy_p['rate'])})."
                )
    lines += [f"**Verdict:** {verdict}", ""]

    lines += [
        "### Batch `20260602` (expired reagents)",
        "",
        f"- mqres rows with `batch_key` containing `20260602`: **{b26['n_samples']}** samples",
    ]
    if b26["median_puc19"] is not None:
        lines.append(
            f"- median pUC19 = **{b26['median_puc19']:.2f}%**; "
            f"pUC19-fail rows = **{b26['n_puc19_fail']}**"
        )
    lines.append(
        f"- evaluable mismatch = **{pct(b26['mismatch']['rate'])}** "
        f"({b26['mismatch']['n_pos']}/{b26['mismatch']['n_evaluable']}); "
        f"noisy-pred rate = **{pct(b26['noisy_pred']['rate'])}**"
    )
    lines.append(f"- match_status counts: `{b26['match_status_counts']}`")
    lines.append(f"- pred_label top: `{b26['pred_label_top']}`")
    lines += [
        "",
        "Interpretation: `20260602` clearly fails conversion QC (pUC19 often ~91–93%), "
        "but after preferred-batch curation many of these samples carry **Normal** preds. "
        "Ground-truth labels are mostly `Unknown`, so trisomy mismatch cannot be assessed "
        "directly. Low pUC19 marks a bad chemistry batch, but it is **not sufficient** "
        "evidence by itself that trisomy calling on the preferred scores is worse.",
        "",
        "### Batches with most pUC19 fails (all mqres↔QC rows)",
        "",
    ]
    if len(top_fail):
        lines += [
            "| mqres batch | n samples | n puc19 fail | median puc19 | mismatch rate | noisy-pred rate |",
            "|-------------|----------:|-------------:|-------------:|--------------:|----------------:|",
        ]
        for _, r in top_fail.iterrows():
            lines.append(
                f"| {r['group']} | {int(r['n_samples'])} | {int(r['n_puc19_fail'])} | "
                f"{r['median_puc19']:.2f} | {pct(r['mismatch_rate'])} | {pct(r['noisy_pred_rate'])} |"
            )
    else:
        lines.append("_No pUC19-fail batches found._")

    lines += [
        "",
        "## Outputs",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `qc_bioinfo_table.csv` | Cleaned QC table (`sample_raw` → `sample`) |",
        "| `qc_bam_map.csv` | **One row per clean_bam → QC** |",
        "| `qc_mqres_batch_map.csv` | Alias of `qc_bam_map.csv` |",
        "| `qc_bams_unmapped.csv` | Bams with no QC |",
        "| `qc_samples_missing_from_db.csv` | mqres samples with no QC row after cleaning |",
        "| `qc_puc19_performance.csv` | Selected-bam QC + meta performance |",
        "| `qc_batch_summary.csv` | Per-batch aggregates |",
        "| `qc_analysis_summary.md` | This report |",
        "| `qc_analysis_summary.json` | Machine-readable stats |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = args.meta or (outdir / "meta_samplesheet.csv")
    mqres_path = args.mqres or (outdir / "mqres_samplesheet.csv")

    if args.qc_csv and args.qc_csv.is_file():
        qc = pd.read_csv(args.qc_csv)
        if "dataset_date" not in qc.columns:
            qc["dataset_date"] = None
        qc["dataset_date"] = qc["dataset_date"].map(_norm_yyyymmdd)
        qc["dataset_date"] = qc["dataset_date"].fillna(
            qc["dataset"].map(extract_batch_date)
        ).map(_norm_yyyymmdd)
    else:
        qc = read_qc_bioinfo_df(as_pandas=True)
        qc["dataset_date"] = qc["dataset_date"].map(_norm_yyyymmdd)

    qc_path = outdir / "qc_bioinfo_table.csv"
    qc.to_csv(qc_path, index=False)

    meta = pd.read_csv(meta_path)
    mqres = pd.read_csv(mqres_path)
    mq_bams = build_mqres_bam_level(mqres)
    mapped = map_bam_to_qc(mq_bams, qc)
    mapped = qc_fail_flags(mapped)
    mapped.to_csv(outdir / "qc_bam_map.csv", index=False)
    mapped.to_csv(outdir / "qc_mqres_batch_map.csv", index=False)  # alias

    missing_samples = (
        mapped.loc[mapped["map_status"] == "sample_not_in_qc", ["sample"]]
        .drop_duplicates()
        .sort_values("sample")
    )
    missing_samples.to_csv(outdir / "qc_samples_missing_from_db.csv", index=False)

    unmapped_bams = mapped.loc[
        mapped["puc19"].isna(),
        ["sample", "clean_bam", "mqres_batch_key", "map_status"],
    ]
    unmapped_bams.to_csv(outdir / "qc_bams_unmapped.csv", index=False)

    perf, batch_summary, summary = analyze_puc19_vs_performance(mapped, meta)
    summary["n_mqres_rows"] = int(len(mqres))
    summary["n_mqres_samples"] = int(mqres["sample"].nunique())
    summary["n_unique_clean_bam"] = int(mqres["clean_bam"].nunique())
    summary["n_bam_mapped"] = int(mapped["puc19"].notna().sum())
    summary["n_bam_unmapped"] = int(mapped["puc19"].isna().sum())
    summary["map_rule_counts"] = mapped["map_rule"].value_counts().to_dict()
    summary["n_qc_samples_after_clean"] = int(qc["sample"].nunique())
    summary["n_samples_not_in_qc"] = int(len(missing_samples))
    summary["samples_not_in_qc"] = missing_samples["sample"].tolist()
    # example large-gap singleton
    sing = mapped[mapped["map_rule"] == "direct_singleton"].copy()
    if len(sing) and sing["day_delta"].notna().any():
        summary["direct_singleton_day_delta"] = {
            "median": float(sing["day_delta"].median()),
            "max": float(sing["day_delta"].max()),
            "n_gt_7d": int((sing["day_delta"] > 7).sum()),
            "n_gt_30d": int((sing["day_delta"] > 30).sum()),
        }
    perf.to_csv(outdir / "qc_puc19_performance.csv", index=False)
    batch_summary.to_csv(outdir / "qc_batch_summary.csv", index=False)

    write_markdown(
        outdir / "qc_analysis_summary.md",
        mapped,
        perf,
        batch_summary,
        summary,
        args.max_day_delta,
    )
    with open(outdir / "qc_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote markdown → {outdir / 'qc_analysis_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
