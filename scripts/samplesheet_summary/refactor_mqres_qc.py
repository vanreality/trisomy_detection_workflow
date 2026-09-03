#!/usr/bin/env python3
"""Refactor mqres samplesheet with QC join + multi-batch puc19 vs noisy-pred.

1. Prefer ``.parquet`` deconv over ``.txt`` per (clean_bam, is_single_end)
   so each bam has exactly 2 deconv rows (PE+SE).
2. Emit mqres with columns:
   ``sample, deconv_res, clean_bam, mqres_batch, qc_batch, puc19, lambda``
   (batch columns are YYYYMMDD only).
3. For multi-batch samples, scan per-batch pred_label and test whether
   lower puc19 tracks noisier predictions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "samplesheet_summary"))
from reorganize_samplesheet import (  # noqa: E402
    DEFAULT_PIPELINE_OUTPUTS,
    n_trisomy_signals,
    normalize_batch_key,
    scan_batch_pred,
)

TOOLS_CANDIDATES = [
    ROOT / "notebooks" / "aipt_1.0",
    ROOT / "notebooks" / "240k_dev",
]
for tools_parent in TOOLS_CANDIDATES:
    if (tools_parent / "tools" / "db_helper.py").is_file():
        sys.path.insert(0, str(tools_parent))
        break

from tools.db_helper import extract_batch_date  # noqa: E402

DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")
NOISY_MIN_T = 3
_DATE_RE = re.compile(r"^(\d{8})")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--mqres", type=Path, default=None)
    p.add_argument("--bam-map", type=Path, default=None)
    p.add_argument(
        "--pipeline-output",
        action="append",
        default=None,
        help="Pipeline output root(s) for per-batch pred scan (repeatable).",
    )
    return p.parse_args(argv)


def yyyymmdd(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    m = _DATE_RE.match(s) or re.search(r"(\d{8})", s)
    return m.group(1) if m else extract_batch_date(s)


def deconv_priority(path: object) -> int:
    """Lower is better: parquet → txt → other."""
    p = str(path).lower()
    if p.endswith(".parquet"):
        return 0
    if p.endswith(".txt"):
        return 1
    return 2


def prefer_parquet_deconv(mqres: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep best deconv source per (clean_bam, is_single_end).

    Returns (deduped_mqres, dropped_rows).
    """
    df = mqres.copy()
    if "is_single_end" not in df.columns:
        # infer from path
        df["is_single_end"] = df["deconv_res"].astype(str).str.contains(
            "single_end", case=False, na=False
        )
    df["_prio"] = df["deconv_res"].map(deconv_priority)
    # best priority available for this bam+end
    best = (
        df.groupby(["clean_bam", "is_single_end"], as_index=False)["_prio"]
        .min()
        .rename(columns={"_prio": "_best"})
    )
    merged = df.merge(best, on=["clean_bam", "is_single_end"], how="left")
    keep = merged[merged["_prio"] == merged["_best"]].drop(
        columns=["_prio", "_best"]
    )
    drop = merged[merged["_prio"] != merged["_best"]].drop(
        columns=["_prio", "_best"]
    )
    # if still >1 row per bam+end (duplicate same format), keep first
    keep = keep.drop_duplicates(["clean_bam", "is_single_end"], keep="first")
    return keep.reset_index(drop=True), drop.reset_index(drop=True)


def join_qc(mqres: pd.DataFrame, bam_map: pd.DataFrame) -> pd.DataFrame:
    """Attach qc_batch / puc19 / lambda per clean_bam."""
    qc_cols = [
        c
        for c in (
            "clean_bam",
            "qc_dataset_date",
            "puc19",
            "lambda",
            "mqres_batch_key",
            "map_rule",
            "qc_dataset",
        )
        if c in bam_map.columns
    ]
    q = bam_map[qc_cols].drop_duplicates("clean_bam")
    out = mqres.merge(q, on="clean_bam", how="left")
    # mqres_batch from batch_key / batch / mqres_batch_key
    if "batch_key" in out.columns:
        out["mqres_batch"] = out["batch_key"].map(yyyymmdd)
    elif "batch" in out.columns:
        out["mqres_batch"] = out["batch"].map(yyyymmdd)
    else:
        out["mqres_batch"] = out.get("mqres_batch_key", pd.Series(dtype=str)).map(
            yyyymmdd
        )
    # fill from map if needed
    if "mqres_batch_key" in out.columns:
        miss = out["mqres_batch"].isna()
        out.loc[miss, "mqres_batch"] = out.loc[miss, "mqres_batch_key"].map(yyyymmdd)
    out["qc_batch"] = out["qc_dataset_date"].map(yyyymmdd)
    # keep internal batch_key for pred scan
    if "batch_key" not in out.columns and "mqres_batch_key" in out.columns:
        out["batch_key"] = out["mqres_batch_key"]
    elif "batch_key" not in out.columns and "batch" in out.columns:
        out["batch_key"] = out["batch"].map(normalize_batch_key)
    return out


def build_mqres_out(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample",
        "deconv_res",
        "clean_bam",
        "mqres_batch",
        "qc_batch",
        "puc19",
        "lambda",
    ]
    out = df[cols].copy()
    out["mqres_batch"] = out["mqres_batch"].map(yyyymmdd)
    out["qc_batch"] = out["qc_batch"].map(yyyymmdd)
    out = out.sort_values(["sample", "mqres_batch", "clean_bam", "deconv_res"])
    return out.reset_index(drop=True)


def analyze_multibatch_puc19_noisy(
    mqres_full: pd.DataFrame,
    meta: pd.DataFrame,
    pipeline_outputs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per (sample, batch) puc19 vs scanned pred_label for multi-batch samples."""
    # one row per sample × mqres_batch (bam-level QC already joined)
    batch_df = (
        mqres_full.groupby(["sample", "mqres_batch"], as_index=False)
        .agg(
            batch_key=("batch_key", "first"),
            clean_bam=("clean_bam", "first"),
            puc19=("puc19", "first"),
            lambda_pct=("lambda", "first"),
            qc_batch=("qc_batch", "first"),
            n_deconv=("deconv_res", "size"),
        )
    )
    n_batch = batch_df.groupby("sample")["mqres_batch"].transform("nunique")
    multi = batch_df[n_batch > 1].copy()
    if multi.empty:
        return multi, pd.DataFrame(), {"n_multibatch_samples": 0}

    rows = []
    for r in multi.itertuples(index=False):
        key = normalize_batch_key(r.batch_key) or str(r.batch_key)
        scanned = scan_batch_pred(r.sample, key, pipeline_outputs) or {}
        pred = scanned.get("pred_label")
        n_t = n_trisomy_signals(pred)
        rows.append(
            {
                "sample": r.sample,
                "mqres_batch": r.mqres_batch,
                "batch_key": key,
                "qc_batch": r.qc_batch,
                "puc19": r.puc19,
                "lambda": r.lambda_pct,
                "pred_label": pred,
                "n_t_signals": n_t,
                "noisy_pred": n_t >= NOISY_MIN_T,
                "scan_status": scanned.get("status"),
                "clean_bam": r.clean_bam,
            }
        )
    detail = pd.DataFrame(rows)

    meta_cols = [c for c in ("sample", "label", "pred_label", "preferred_batch_key") if c in meta.columns]
    if meta_cols:
        detail = detail.merge(
            meta[meta_cols].rename(columns={"pred_label": "meta_pred_label"}),
            on="sample",
            how="left",
        )

    # per-sample contrast: lowest vs highest puc19 among batches with pred
    contrasts = []
    for sample, g in detail.groupby("sample"):
        g2 = g[g["pred_label"].notna() & g["puc19"].notna()].copy()
        if len(g2) < 2:
            contrasts.append(
                {
                    "sample": sample,
                    "n_batches": int(g["mqres_batch"].nunique()),
                    "n_batches_with_pred_and_qc": int(len(g2)),
                    "status": "insufficient",
                }
            )
            continue
        g2 = g2.sort_values("puc19")
        low = g2.iloc[0]
        high = g2.iloc[-1]
        # also: is the noisiest batch the lowest puc19?
        noisy = g2[g2["noisy_pred"]]
        clean = g2[~g2["noisy_pred"]]
        if len(noisy) and len(clean):
            pattern = "mixed_noisy_and_clean"
            noisy_lower_puc = bool(
                float(noisy["puc19"].min()) < float(clean["puc19"].min())
            )
            # strict: max noisy puc19 < min clean puc19
            noisy_all_lower = bool(
                float(noisy["puc19"].max()) < float(clean["puc19"].min())
            )
            noisy_higher_puc = bool(
                float(noisy["puc19"].min()) > float(clean["puc19"].max())
            )
        elif len(noisy) and not len(clean):
            pattern = "all_noisy"
            noisy_lower_puc = None
            noisy_all_lower = None
            noisy_higher_puc = None
        elif not len(noisy) and len(clean):
            pattern = "all_clean"
            noisy_lower_puc = None
            noisy_all_lower = None
            noisy_higher_puc = None
        else:
            pattern = "no_pred"
            noisy_lower_puc = None
            noisy_all_lower = None
            noisy_higher_puc = None

        same_puc = abs(float(high["puc19"]) - float(low["puc19"])) < 1e-6
        contrasts.append(
            {
                "sample": sample,
                "n_batches": int(g["mqres_batch"].nunique()),
                "n_batches_with_pred_and_qc": int(len(g2)),
                "status": pattern,
                "low_puc19": float(low["puc19"]),
                "low_puc19_batch": low["mqres_batch"],
                "low_puc19_pred": low["pred_label"],
                "low_puc19_noisy": bool(low["noisy_pred"]),
                "high_puc19": float(high["puc19"]),
                "high_puc19_batch": high["mqres_batch"],
                "high_puc19_pred": high["pred_label"],
                "high_puc19_noisy": bool(high["noisy_pred"]),
                "puc19_delta": float(high["puc19"] - low["puc19"]),
                "same_puc19": same_puc,
                "low_batch_noisier": (
                    (not same_puc)
                    and bool(low["noisy_pred"])
                    and not bool(high["noisy_pred"])
                ),
                "high_batch_noisier": (
                    (not same_puc)
                    and bool(high["noisy_pred"])
                    and not bool(low["noisy_pred"])
                ),
                "noisy_batch_has_lower_puc19": noisy_lower_puc,
                "noisy_batch_has_higher_puc19": noisy_higher_puc,
                "all_noisy_batches_lower_puc19_than_clean": noisy_all_lower,
            }
        )
    contrast_df = pd.DataFrame(contrasts)

    mixed = contrast_df[contrast_df["status"] == "mixed_noisy_and_clean"]
    def _sum_true(series) -> int:
        if series is None or len(series) == 0:
            return 0
        return int(series.fillna(False).astype(bool).sum())

    summary = {
        "n_multibatch_samples": int(detail["sample"].nunique()),
        "n_batch_rows": int(len(detail)),
        "n_with_pred_and_qc": int(
            (detail["pred_label"].notna() & detail["puc19"].notna()).sum()
        ),
        "status_counts": contrast_df["status"].value_counts().to_dict(),
        "n_mixed_noisy_clean": int(len(mixed)),
        "n_mixed_where_low_puc_is_noisier": (
            _sum_true(mixed["low_batch_noisier"]) if len(mixed) else 0
        ),
        "n_mixed_where_high_puc_is_noisier": (
            _sum_true(mixed["high_batch_noisier"]) if len(mixed) else 0
        ),
        "n_mixed_where_noisy_has_lower_puc19": (
            _sum_true(mixed["noisy_batch_has_lower_puc19"]) if len(mixed) else 0
        ),
        "n_mixed_where_noisy_has_higher_puc19": (
            _sum_true(mixed["noisy_batch_has_higher_puc19"]) if len(mixed) else 0
        ),
        "n_mixed_where_all_noisy_lower_than_clean": (
            _sum_true(mixed["all_noisy_batches_lower_puc19_than_clean"])
            if len(mixed)
            else 0
        ),
        "n_mixed_same_puc19": (
            _sum_true(mixed["same_puc19"]) if len(mixed) else 0
        ),
        "noisy_min_t_signals": NOISY_MIN_T,
    }
    if len(mixed):
        # among mixed: mean puc19 noisy vs clean batches
        m_samples = set(mixed["sample"])
        d = detail[detail["sample"].isin(m_samples) & detail["puc19"].notna()]
        summary["mixed_median_puc19_noisy"] = (
            float(d.loc[d["noisy_pred"], "puc19"].median())
            if d["noisy_pred"].any()
            else None
        )
        summary["mixed_median_puc19_clean"] = (
            float(d.loc[~d["noisy_pred"], "puc19"].median())
            if (~d["noisy_pred"]).any()
            else None
        )
    return detail, contrast_df, summary


def write_markdown(
    path: Path,
    dedupe_info: dict,
    mqres_out: pd.DataFrame,
    contrast: pd.DataFrame,
    mb_summary: dict,
) -> None:
    mixed = contrast[contrast["status"] == "mixed_noisy_and_clean"] if len(contrast) else contrast
    lines = [
        "# Refactored mqres + multi-batch pUC19 vs noisy pred",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Deconv dedupe (parquet preferred over txt)",
        "",
        f"- Input mqres rows: **{dedupe_info['n_in']}**",
        f"- Dropped `.txt` (or lower-priority) rows: **{dedupe_info['n_dropped']}**",
        f"- Output rows: **{len(mqres_out)}**",
        f"- Unique `clean_bam`: **{mqres_out['clean_bam'].nunique()}**",
        f"- Deconv per bam: {dedupe_info.get('deconv_per_bam')}",
        "",
    ]
    if dedupe_info.get("dropped_samples"):
        lines.append("Dropped rows were from:")
        for s, n in dedupe_info["dropped_samples"].items():
            lines.append(f"- `{s}`: {n} row(s)")
        lines.append("")

    lines += [
        "## 2. Refactored `mqres_samplesheet.csv` columns",
        "",
        "`sample`, `deconv_res`, `clean_bam`, `mqres_batch` (YYYYMMDD), "
        "`qc_batch` (YYYYMMDD), `puc19`, `lambda`",
        "",
        f"- Rows with QC (`puc19` not null): **{int(mqres_out['puc19'].notna().sum())}** / {len(mqres_out)}",
        f"- Unique samples: **{mqres_out['sample'].nunique()}**",
        "",
        "## 3. Multi-batch: does lower pUC19 explain noisy preds?",
        "",
        f"Noisy = pred_label with ≥{NOISY_MIN_T} T/Gray_T signals (scanned from pipeline zscore reports).",
        "",
        f"- Multi-batch samples: **{mb_summary.get('n_multibatch_samples', 0)}**",
        f"- Status counts: `{mb_summary.get('status_counts', {})}`",
        "",
    ]
    n_mixed = mb_summary.get("n_mixed_noisy_clean", 0)
    n_low_noisy = mb_summary.get("n_mixed_where_low_puc_is_noisier", 0)
    n_high_noisy = mb_summary.get("n_mixed_where_high_puc_is_noisier", 0)
    n_noisy_lower = mb_summary.get("n_mixed_where_noisy_has_lower_puc19", 0)
    n_noisy_higher = mb_summary.get("n_mixed_where_noisy_has_higher_puc19", 0)
    n_same = mb_summary.get("n_mixed_same_puc19", 0)
    lines += [
        f"- Mixed (some noisy + some clean batches): **{n_mixed}**",
        f"  - lowest-puc19 batch is noisier: **{n_low_noisy}**",
        f"  - highest-puc19 batch is noisier: **{n_high_noisy}**",
        f"  - noisy batch(es) have **lower** puc19 than clean: **{n_noisy_lower}**",
        f"  - noisy batch(es) have **higher** puc19 than clean: **{n_noisy_higher}**",
        f"  - same puc19 on compared batches: **{n_same}**",
        "",
    ]
    if mb_summary.get("mixed_median_puc19_noisy") is not None:
        lines.append(
            f"- Among mixed samples: median puc19 noisy batches = "
            f"**{mb_summary['mixed_median_puc19_noisy']:.2f}%**, "
            f"clean batches = **{mb_summary['mixed_median_puc19_clean']:.2f}%**"
        )
        lines.append("")

    if n_mixed == 0:
        verdict = (
            "No multi-batch sample has both a noisy and a clean pred across batches "
            "(with scannable preds + QC), so pUC19 cannot be confirmed as the explanation."
        )
    elif n_noisy_higher > n_noisy_lower:
        verdict = (
            f"**No — lower pUC19 does not explain noisier preds within sample.** "
            f"In **{n_noisy_higher}/{n_mixed}** mixed samples the noisy batch(es) actually have "
            f"**higher** puc19 than clean ones (only {n_noisy_lower} show noisy-at-lower-puc19). "
            f"Classic pattern: `20260602` (puc19~91–92%, QC fail) often predicts Normal, while "
            f"an earlier higher-puc19 batch is multi-T noisy. Batch choice / model run differs; "
            f"conversion QC alone does not track within-sample noise."
        )
    elif n_low_noisy >= max(1, n_mixed * 0.6):
        verdict = (
            f"Partially supported: in **{n_low_noisy}/{n_mixed}** mixed samples the lowest-puc19 "
            "batch is noisier. Still not a reliable sole explanation."
        )
    else:
        verdict = (
            f"Weak / not supported: only **{n_low_noisy}/{n_mixed}** mixed samples have the "
            "lowest-puc19 batch as the noisier one."
        )
    lines += [f"**Verdict:** {verdict}", ""]

    if len(mixed):
        lines += [
            "### Mixed samples (noisy vs clean batches)",
            "",
            "| sample | low puc19 (batch, pred) | high puc19 (batch, pred) | who noisier? |",
            "|--------|-------------------------|--------------------------|--------------|",
        ]
        for _, r in mixed.sort_values("sample").iterrows():
            if r.get("same_puc19"):
                who = "tie puc19"
            elif r.get("high_batch_noisier"):
                who = "high puc19"
            elif r.get("low_batch_noisier"):
                who = "low puc19"
            else:
                who = "mixed/other"
            lines.append(
                f"| {r['sample']} | {r['low_puc19']:.1f}% ({r['low_puc19_batch']}, "
                f"`{r['low_puc19_pred']}`) | {r['high_puc19']:.1f}% ({r['high_puc19_batch']}, "
                f"`{r['high_puc19_pred']}`) | {who} |"
            )
        lines.append("")

    lines += [
        "## Outputs",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `mqres_samplesheet.csv` | Refactored mqres |",
        "| `mqres_deconv_dropped.csv` | Removed txt (or lower-priority) deconv rows |",
        "| `report_multibatch_puc19_noisy_detail.csv` | Per sample×batch pred + puc19 |",
        "| `report_multibatch_puc19_noisy_contrast.csv` | Per-sample low vs high puc19 |",
        "| `report_multibatch_puc19_noisy.md` | This report |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    mqres_path = args.mqres or (outdir / "mqres_samplesheet.csv")
    bam_map_path = args.bam_map or (outdir / "qc_bam_map.csv")
    meta_path = outdir / "meta_samplesheet.csv"
    pipeline_outputs = args.pipeline_output or list(DEFAULT_PIPELINE_OUTPUTS)

    mqres_in = pd.read_csv(mqres_path)
    # If already refactored (no is_single_end / batch_key), try to recover from backup
    backup = outdir / "mqres_samplesheet_pre_qc_refactor.csv"
    if "deconv_res" in mqres_in.columns and "batch_key" not in mqres_in.columns:
        if backup.is_file():
            print(f"Input looks refactored; reloading backup {backup}")
            mqres_in = pd.read_csv(backup)
        else:
            raise SystemExit(
                "mqres lacks batch_key/is_single_end and no backup found. "
                f"Expected {backup}"
            )
    elif "batch_key" in mqres_in.columns and not backup.is_file():
        mqres_in.to_csv(backup, index=False)
        print(f"Wrote backup → {backup}")

    keep, drop = prefer_parquet_deconv(mqres_in)
    drop.to_csv(outdir / "mqres_deconv_dropped.csv", index=False)

    bam_map = pd.read_csv(bam_map_path)
    full = join_qc(keep, bam_map)
    mqres_out = build_mqres_out(full)
    mqres_out.to_csv(outdir / "mqres_samplesheet.csv", index=False)

    deconv_per_bam = (
        mqres_out.groupby("clean_bam").size().value_counts().sort_index().to_dict()
    )
    dedupe_info = {
        "n_in": int(len(mqres_in)),
        "n_dropped": int(len(drop)),
        "dropped_samples": drop["sample"].value_counts().to_dict() if len(drop) else {},
        "deconv_per_bam": {int(k): int(v) for k, v in deconv_per_bam.items()},
    }

    meta = pd.read_csv(meta_path) if meta_path.is_file() else pd.DataFrame()
    detail, contrast, mb_summary = analyze_multibatch_puc19_noisy(
        full, meta, pipeline_outputs
    )
    detail.to_csv(outdir / "report_multibatch_puc19_noisy_detail.csv", index=False)
    contrast.to_csv(outdir / "report_multibatch_puc19_noisy_contrast.csv", index=False)
    write_markdown(
        outdir / "report_multibatch_puc19_noisy.md",
        dedupe_info,
        mqres_out,
        contrast,
        mb_summary,
    )
    summary = {
        "dedupe": dedupe_info,
        "n_mqres_out": int(len(mqres_out)),
        "n_unique_bam": int(mqres_out["clean_bam"].nunique()),
        "n_with_qc": int(mqres_out["puc19"].notna().sum()),
        "multibatch_puc19_noisy": mb_summary,
    }
    with open(outdir / "report_multibatch_puc19_noisy.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote mqres → {outdir / 'mqres_samplesheet.csv'}")
    print(f"Wrote markdown → {outdir / 'report_multibatch_puc19_noisy.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
