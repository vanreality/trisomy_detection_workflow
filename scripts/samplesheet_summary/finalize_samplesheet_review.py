#!/usr/bin/env python3
"""Finalize samplesheet review: QC special-case, drop noisy multi-batch, clean outdir.

Writes only:
  - samplesheet_review.md
  - meta_samplesheet.csv
  - mqres_samplesheet.csv
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

DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")

# mqres_batch YYYYMMDD → QC dataset_date to use (inverted 20260528 ↔ 20260602 pairing)
SPECIAL_MQRES_QC_DATE = {
    "20260528": "20260602",  # noisy / lower puc19 chemistry
    "20260602": "20260529",  # better / higher puc19 (QC labeled 20260529)
}

# Whole-batch drop for multi-batch samples only
NOISY_BAD_BATCHES = {
    "20260203",  # lambda ~60% (>>1% QC fail)
    "20260528",  # multi-T noisy (after special-case QC = low puc19)
    "20260623",  # multi-T noisy vs cleaner 20260626
}

SAMPLE_BLACKLIST = [
    "PTAY1330P7S1",
    "PTAY0652P7H1",
    "PTAY0620P8S1",
    "PTAY0535P8S1",
    "PTAY1351P8S1",
]

META_DROP_COLS = [
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

NOISY_MIN_T = 3
KEEP_FILES = {
    "samplesheet_review.md",
    "meta_samplesheet.csv",
    "mqres_samplesheet.csv",
}
_DATE_RE = re.compile(r"(\d{8})")


def yyyymmdd(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    m = _DATE_RE.search(s)
    return m.group(1) if m else None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--pipeline-output",
        action="append",
        default=None,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete other files in outdir.",
    )
    return p.parse_args(argv)


def load_qc_table(outdir: Path) -> pd.DataFrame:
    path = outdir / "qc_bioinfo_table.csv"
    if not path.is_file():
        raise SystemExit(f"Missing {path}; run analyze_qc_mqres.py first.")
    qc = pd.read_csv(path)
    qc["dataset_date"] = qc["dataset_date"].map(yyyymmdd)
    return qc


def apply_special_qc_mapping(mqres: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """Re-assign puc19/lambda/qc_batch for SPECIAL_MQRES_QC_DATE pairs."""
    out = mqres.copy()
    out["mqres_batch"] = out["mqres_batch"].map(yyyymmdd)
    out["qc_batch"] = out["qc_batch"].map(yyyymmdd)

    qc_by = {}
    for s, g in qc.groupby("sample"):
        by_date: dict[str, pd.Series] = {}
        for _, r in g.iterrows():
            d = yyyymmdd(r["dataset_date"])
            if not d:
                continue
            # prefer *-XML plain dataset
            ds = str(r["dataset"])
            score = 0 if ds.endswith("-XML") else (1 if "-XML" in ds else 2)
            prev = by_date.get(d)
            if prev is None or score < prev["_score"]:
                row = r.copy()
                row["_score"] = score
                by_date[d] = row
        qc_by[s] = by_date

    n_applied = 0
    for idx, r in out.iterrows():
        mb = r["mqres_batch"]
        if mb not in SPECIAL_MQRES_QC_DATE:
            continue
        target = SPECIAL_MQRES_QC_DATE[mb]
        by_date = qc_by.get(r["sample"], {})
        hit = by_date.get(target)
        if hit is None:
            continue
        out.at[idx, "qc_batch"] = target
        out.at[idx, "puc19"] = hit["puc19"]
        out.at[idx, "lambda"] = hit["lambda"]
        n_applied += 1
    return out, n_applied


def scan_multibatch_noisy(
    mqres: pd.DataFrame,
    pipeline_outputs: list[str],
    backup: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Return per (sample, mqres_batch) noisy flag for multi-batch samples."""
    n_batch = mqres.groupby("sample")["mqres_batch"].transform("nunique")
    multi = mqres.loc[n_batch > 1, ["sample", "mqres_batch"]].drop_duplicates()
    # recover batch_key for scanning
    key_map = {}
    if backup is not None and "batch_key" in backup.columns:
        b = backup.copy()
        b["_date"] = b["batch_key"].map(yyyymmdd)
        for r in b.itertuples():
            key_map[(r.sample, r._date)] = normalize_batch_key(r.batch_key)

    rows = []
    for r in multi.itertuples(index=False):
        key = key_map.get((r.sample, r.mqres_batch)) or f"{r.mqres_batch}-XML"
        scanned = scan_batch_pred(r.sample, key, pipeline_outputs) or {}
        pred = scanned.get("pred_label")
        n_t = n_trisomy_signals(pred)
        rows.append(
            {
                "sample": r.sample,
                "mqres_batch": r.mqres_batch,
                "batch_key": key,
                "pred_label": pred,
                "n_t_signals": n_t,
                "noisy_pred": bool(n_t >= NOISY_MIN_T),
            }
        )
    return pd.DataFrame(rows)


def drop_noisy_multibatch_rows(
    mqres: pd.DataFrame,
    noisy_detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop bad/noisy batches for multi-batch samples only."""
    df = mqres.copy()
    df["mqres_batch"] = df["mqres_batch"].map(yyyymmdd)
    n_batch = df.groupby("sample")["mqres_batch"].transform("nunique")
    is_multi = n_batch > 1

    # 1) global bad batches
    drop_global = is_multi & df["mqres_batch"].isin(NOISY_BAD_BATCHES)

    # 2) per-sample noisy pred batches
    noisy_pairs: set[tuple[str, str]] = set()
    if len(noisy_detail):
        nd = noisy_detail.copy()
        nd["mqres_batch"] = nd["mqres_batch"].map(yyyymmdd)
        for r in nd.itertuples(index=False):
            if bool(r.noisy_pred) and r.mqres_batch:
                noisy_pairs.add((r.sample, r.mqres_batch))
    drop_scan = is_multi & df.apply(
        lambda r: (r["sample"], r["mqres_batch"]) in noisy_pairs, axis=1
    )

    drop = drop_global | drop_scan
    # never empty a multi-batch sample
    for sample, g in df[is_multi].groupby("sample"):
        idx = g.index
        if not drop.loc[idx].all():
            continue
        safe = idx[~df.loc[idx, "mqres_batch"].isin(NOISY_BAD_BATCHES)]
        if len(safe):
            drop.loc[safe] = False
            continue
        sub = df.loc[idx]
        if sub["puc19"].notna().any():
            keep_idx = sub["puc19"].astype(float).idxmax()
        else:
            keep_idx = idx[0]
        bam = df.at[keep_idx, "clean_bam"]
        drop.loc[idx[df.loc[idx, "clean_bam"] == bam]] = False

    removed = df.loc[drop].copy()
    kept = df.loc[~drop].copy()
    info = {
        "n_rows_removed": int(drop.sum()),
        "n_samples_affected": int(df.loc[drop, "sample"].nunique()),
        "removed_by_global_bad_batch": int(drop_global.sum()),
        "removed_by_noisy_pred_scan": int((drop_scan & ~drop_global).sum()),
        "removed_batch_counts": (
            {str(k): int(v) for k, v in removed["mqres_batch"].value_counts().items()}
            if len(removed)
            else {}
        ),
    }
    return kept, removed, info


def finalize_meta(meta: pd.DataFrame, mqres: pd.DataFrame) -> pd.DataFrame:
    out = meta[~meta["sample"].isin(SAMPLE_BLACKLIST)].copy()
    avail = (
        mqres.groupby("sample")["mqres_batch"]
        .apply(lambda s: ",".join(sorted({yyyymmdd(x) for x in s if yyyymmdd(x)})))
        .to_dict()
    )
    out["available_batches"] = out["sample"].map(avail).fillna("")
    drop = [c for c in META_DROP_COLS if c in out.columns]
    out = out.drop(columns=drop)
    # put available_batches near sample
    cols = list(out.columns)
    if "available_batches" in cols and "sample" in cols:
        cols.remove("available_batches")
        cols.insert(cols.index("sample") + 1, "available_batches")
        out = out[cols]
    return out.reset_index(drop=True)


def write_review_md(
    path: Path,
    *,
    special_n: int,
    drop_info: dict,
    mqres: pd.DataFrame,
    meta: pd.DataFrame,
    removed: pd.DataFrame,
    noisy_detail: pd.DataFrame,
) -> None:
    lines = [
        "# AIPT samplesheet review — final summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "This note records the full samplesheet maintenance job for AIPT 1.0 meta + mqres,",
        "including batch/score QA, QC conversion-rate join, and cleanup of noisy multi-batch rows.",
        "",
        "## Final deliverables",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `meta_samplesheet.csv` | Meta + `available_batches` (batches still in mqres) |",
        "| `mqres_samplesheet.csv` | Clean mqres: `sample, deconv_res, clean_bam, mqres_batch, qc_batch, puc19, lambda` |",
        "| `samplesheet_review.md` | This document |",
        "",
        f"- meta samples: **{len(meta)}**",
        f"- mqres rows: **{len(mqres)}** / unique bam: **{mqres['clean_bam'].nunique()}** / samples: **{mqres['sample'].nunique()}**",
        "",
        "## Pipeline of work (chronological)",
        "",
        "1. **Reorganize samplesheets** (`reorganize_samplesheet.py`): map samples→batches from mqres,",
        "   score↔mqres consistency, multi-batch preferred-batch QA, FF at precision `0.0001`.",
        "2. **QC table** (`质控数据生信分析结果` via `db_helper`): clean sample names",
        "   (`_QC` / `QC-` / `EM` / `HCPT####P`, B→P, drop names containing `V`), columns",
        "   `sample, lambda, puc19, dataset`.",
        "3. **BAM↔QC map** (`analyze_qc_mqres.py`): one QC per `clean_bam`;",
        "   1 bam + 1 QC → direct map (no ±7-day cutoff); multi → nearest by date.",
        "4. **Deconv dedupe** (`refactor_mqres_qc.py`): prefer `.parquet` over `.txt`",
        "   (dropped 4 txt rows for `PTAY0452P20H1` / `PTAY0519P13H1`) → always 2 deconv / bam.",
        "5. **This finalize step**: special-case QC pairing, drop noisy multi-batch rows,",
        "   blacklist samples, slim columns, clean output directory.",
        "",
        "## QC fail rules",
        "",
        "- Fail if `puc19 < 94%` **or** `lambda > 1%`.",
        "",
        "### Notable bad chemistry",
        "",
        "- **`20260203`**: `JPTAY1400P7H1` has **lambda = 60.4%** (≫1%). Batch is multi-T noisy;",
        "  for multi-batch samples this batch is removed. Single-batch samples on this batch are kept.",
        "",
        "## Special-case mqres ↔ QC mapping",
        "",
        "QC table has **no** `20260528` dataset (nearest high-puc19 label is often `20260529`).",
        "Date-nearest mapping had incorrectly paired:",
        "",
        "| mqres_batch | Wrong QC (before) | Correct QC (special case) | Interpretation |",
        "|-------------|-------------------|---------------------------|----------------|",
        "| `20260528` | ~`20260529` (high puc19) | **`20260602`** (low puc19) | Noisy multi-T mqres |",
        "| `20260602` | `20260602` (low puc19) | **`20260529`** (high puc19) | Cleaner / better mqres |",
        "",
        "Recorded override:",
        "",
        "```",
        "SPECIAL_MQRES_QC_DATE = {",
        "    '20260528': '20260602',",
        "    '20260602': '20260529',",
        "}",
        "```",
        "",
        f"Applied to **{special_n}** mqres rows (sample×deconv).",
        "",
        "After correction, within-sample pattern matches chemistry: lower puc19 ↔ noisier pred",
        "for the `20260528` / `20260602` pair (e.g. JPTAY168x).",
        "",
        "## Removing noisy batches (multi-batch only)",
        "",
        "**Policy:** If a sample has ≥2 mqres batches, drop noisy/bad batches.",
        "Samples with only one batch are **unchanged** even if that batch is noisy.",
        "",
        "### Global noisy-bad batches (dropped for all multi-batch samples)",
        "",
    ]
    for b in sorted(NOISY_BAD_BATCHES):
        lines.append(f"- `{b}`")
    lines += [
        "",
        "### Additional per-sample drops",
        "",
        f"Any multi-batch sample whose scanned pred_label has ≥{NOISY_MIN_T} T/Gray_T signals",
        "on a batch also loses that batch (e.g. occasional `20260310` noisiness).",
        "",
        f"- Rows removed: **{drop_info['n_rows_removed']}**",
        f"- Samples affected: **{drop_info['n_samples_affected']}**",
        f"- Removed by global bad batch: **{drop_info['removed_by_global_bad_batch']}**",
        f"- Removed by noisy-pred scan only: **{drop_info['removed_by_noisy_pred_scan']}**",
        f"- Removed batch counts: `{drop_info['removed_batch_counts']}`",
        "",
    ]
    if len(removed):
        lines += [
            "### Examples of removed multi-batch rows (unique sample×batch)",
            "",
            "| sample | mqres_batch | qc_batch | puc19 | lambda |",
            "|--------|-------------|----------|------:|-------:|",
        ]
        ex = (
            removed.groupby(["sample", "mqres_batch"], as_index=False)
            .agg(qc_batch=("qc_batch", "first"), puc19=("puc19", "first"), **{"lambda": ("lambda", "first")})
            .sort_values(["mqres_batch", "sample"])
            .head(40)
        )
        for _, r in ex.iterrows():
            lam = r["lambda"]
            lam_s = f"{lam:.1f}" if pd.notna(lam) else ""
            puc = r["puc19"]
            puc_s = f"{puc:.1f}" if pd.notna(puc) else ""
            lines.append(
                f"| {r['sample']} | {r['mqres_batch']} | {r['qc_batch']} | {puc_s} | {lam_s} |"
            )
        lines.append("")

    lines += [
        "## Sample blacklist (removed from meta + mqres)",
        "",
    ]
    for s in SAMPLE_BLACKLIST:
        lines.append(f"- `{s}`")
    lines += [
        "",
        "## meta_samplesheet.csv columns",
        "",
        "Dropped: "
        + ", ".join(f"`{c}`" for c in META_DROP_COLS)
        + ".",
        "",
        "Added: `available_batches` — comma-separated `YYYYMMDD` values still present in mqres",
        "for that sample (empty if sample has no mqres rows).",
        "",
        "## mqres_samplesheet.csv columns",
        "",
        "`sample`, `deconv_res`, `clean_bam`, `mqres_batch` (YYYYMMDD), "
        "`qc_batch` (YYYYMMDD), `puc19`, `lambda`",
        "",
        "## Scripts (repo)",
        "",
        "| Script | Role |",
        "|--------|------|",
        "| `scripts/samplesheet_summary/reorganize_samplesheet.py` | Initial meta/mqres reorg + FF |",
        "| `scripts/samplesheet_summary/analyze_qc_mqres.py` | QC fetch + bam↔QC map |",
        "| `scripts/samplesheet_summary/refactor_mqres_qc.py` | Parquet prefer + QC join columns |",
        "| `scripts/samplesheet_summary/finalize_samplesheet_review.py` | This finalize + dir cleanup |",
        "| `notebooks/aipt_1.0/tools/db_helper.py` | QC table read/clean helpers |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def clean_outdir(outdir: Path, *, dry_run: bool) -> list[str]:
    removed = []
    for p in sorted(outdir.iterdir()):
        if p.name in KEEP_FILES:
            continue
        removed.append(p.name)
        if not dry_run:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                # do not recursively wipe unexpected dirs unless empty-ish; skip dirs
                pass
    return removed


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    pipeline_outputs = args.pipeline_output or list(DEFAULT_PIPELINE_OUTPUTS)

    mqres_path = outdir / "mqres_samplesheet.csv"
    meta_path = outdir / "meta_samplesheet.csv"
    backup_path = outdir / "mqres_samplesheet_pre_qc_refactor.csv"

    mqres = pd.read_csv(mqres_path)
    meta = pd.read_csv(meta_path)
    qc = load_qc_table(outdir)
    backup = pd.read_csv(backup_path) if backup_path.is_file() else None

    mqres, special_n = apply_special_qc_mapping(mqres, qc)

    # Prefer cached noisy detail if present and fresh-ish; else rescan
    detail_path = outdir / "report_multibatch_puc19_noisy_detail.csv"
    if detail_path.is_file():
        noisy_detail = pd.read_csv(detail_path)
        noisy_detail["mqres_batch"] = noisy_detail["mqres_batch"].map(yyyymmdd)
        # recompute noisy with same threshold
        if "n_t_signals" in noisy_detail.columns:
            noisy_detail["noisy_pred"] = noisy_detail["n_t_signals"] >= NOISY_MIN_T
    else:
        noisy_detail = scan_multibatch_noisy(mqres, pipeline_outputs, backup)

    mqres_clean, removed, drop_info = drop_noisy_multibatch_rows(mqres, noisy_detail)
    mqres_clean = mqres_clean[~mqres_clean["sample"].isin(SAMPLE_BLACKLIST)].copy()
    removed_bl = mqres[mqres["sample"].isin(SAMPLE_BLACKLIST)]
    if len(removed_bl):
        removed = pd.concat([removed, removed_bl], ignore_index=True)

    # final column set / date normalize
    for c in ("mqres_batch", "qc_batch"):
        mqres_clean[c] = mqres_clean[c].map(yyyymmdd)
    mqres_out = mqres_clean[
        ["sample", "deconv_res", "clean_bam", "mqres_batch", "qc_batch", "puc19", "lambda"]
    ].sort_values(["sample", "mqres_batch", "clean_bam", "deconv_res"])
    mqres_out = mqres_out.reset_index(drop=True)

    meta_out = finalize_meta(meta, mqres_out)

    # write keepers first
    mqres_out.to_csv(outdir / "mqres_samplesheet.csv", index=False)
    meta_out.to_csv(outdir / "meta_samplesheet.csv", index=False)
    write_review_md(
        outdir / "samplesheet_review.md",
        special_n=special_n,
        drop_info=drop_info,
        mqres=mqres_out,
        meta=meta_out,
        removed=removed,
        noisy_detail=noisy_detail,
    )

    deleted = clean_outdir(outdir, dry_run=args.dry_run)
    summary = {
        "special_qc_rows_applied": special_n,
        "drop_info": drop_info,
        "n_meta": int(len(meta_out)),
        "n_mqres": int(len(mqres_out)),
        "n_bam": int(mqres_out["clean_bam"].nunique()),
        "blacklist": SAMPLE_BLACKLIST,
        "noisy_bad_batches": sorted(NOISY_BAD_BATCHES),
        "special_mqres_qc_date": SPECIAL_MQRES_QC_DATE,
        "deleted_files": deleted,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {outdir / 'samplesheet_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
