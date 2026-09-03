#!/usr/bin/env python3
"""Reorder meta, remap QC with H/S aliases, write samplesheet summary markdown."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "notebooks" / "aipt_1.0"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "scripts" / "samplesheet_summary"))

from tools.db_helper import extract_batch_date, read_qc_bioinfo_df  # noqa: E402

DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")

# Same special-case as finalize_samplesheet_review.py
SPECIAL_MQRES_QC_DATE = {
    "20260528": "20260602",
    "20260602": "20260529",
}

IGNORE_LABELS = {"Twin", "Unknown", "XO", "M21"}
META_COL_ORDER = [
    "sample",
    "ff_before_mq",
    "ff_after_mq",
    "label",
    "pred_label",
    "purity",
    "depth_qc",
    "available_batches",
    "match_status",
    "conservative_match_status",
    "set",
    "week",
    "timepoint",
    "state",
    "age",
    "conception_mode",
    "reproductive_history",
    "HCG",
    "ref_type",
    "mean_target_coverage",
    "cpg_mean_coverage",
    "snp_mean_coverage",
    "beta_zscores",
    "rc_zscores",
    "final_zscores",
]
_DATE_RE = re.compile(r"(\d{8})")


def yyyymmdd(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    m = _DATE_RE.search(s)
    return m.group(1) if m else extract_batch_date(s)


def hs_swap(sample: str) -> Optional[str]:
    if sample.endswith("H1"):
        return sample[:-2] + "S1"
    if sample.endswith("S1"):
        return sample[:-2] + "H1"
    return None


def _date_ord(d: Optional[str]) -> Optional[int]:
    d = yyyymmdd(d)
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y%m%d").toordinal()
    except ValueError:
        return None


def prefer_qc_row(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    tmp["_score"] = tmp["dataset"].map(
        lambda d: 0 if str(d).endswith("-XML") else (1 if "-XML" in str(d) else 2)
    )
    return tmp.sort_values(["_score", "dataset"]).iloc[0]


def build_qc_index(qc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    qc = qc.copy()
    qc["dataset_date"] = qc["dataset_date"].map(yyyymmdd)
    return {s: g.reset_index(drop=True) for s, g in qc.groupby("sample")}


def qc_candidates_for_sample(
    sample: str, qc_by: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, str]:
    """Exact sample + H↔S alias QC rows (tag source)."""
    frames = []
    if sample in qc_by:
        g = qc_by[sample].copy()
        g["_alias"] = "exact"
        frames.append(g)
    alt = hs_swap(sample)
    if alt and alt in qc_by:
        g = qc_by[alt].copy()
        g["_alias"] = "hs_swap"
        g["_alias_sample"] = alt
        frames.append(g)
    if not frames:
        return pd.DataFrame(), "missing"
    out = pd.concat(frames, ignore_index=True)
    # unique by dataset
    rows = []
    for _, g in out.groupby("dataset", dropna=False):
        hit = prefer_qc_row(g)
        rows.append(hit)
    return pd.DataFrame(rows), ("exact+hs" if len(frames) > 1 else frames[0]["_alias"].iloc[0])


def remap_mqres_qc(mqres: pd.DataFrame, qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remap each mqres row's qc_batch/puc19/lambda using exact+HS nearest date."""
    qc_by = build_qc_index(qc)
    out = mqres.copy()
    out["mqres_batch"] = out["mqres_batch"].map(yyyymmdd)
    audit_rows = []

    cache: dict[str, tuple[pd.DataFrame, str]] = {}
    for idx, r in out.iterrows():
        sample = r["sample"]
        if sample not in cache:
            cache[sample] = qc_candidates_for_sample(sample, qc_by)
        qsub, how = cache[sample]
        old = {
            "qc_batch": yyyymmdd(r.get("qc_batch")),
            "puc19": r.get("puc19"),
            "lambda": r.get("lambda"),
        }
        if qsub.empty:
            out.at[idx, "qc_batch"] = None
            out.at[idx, "puc19"] = np.nan
            out.at[idx, "lambda"] = np.nan
            audit_rows.append(
                {
                    "sample": sample,
                    "mqres_batch": r["mqres_batch"],
                    "status": "no_qc",
                    "alias_mode": how,
                    **{f"old_{k}": v for k, v in old.items()},
                }
            )
            continue

        mq_date = r["mqres_batch"]
        # special-case override target date when present in candidate pool
        target = SPECIAL_MQRES_QC_DATE.get(mq_date)
        hit = None
        alias_used = None
        day_delta = None
        rule = None
        if target is not None:
            special = qsub[qsub["dataset_date"] == target]
            if len(special):
                hit = prefer_qc_row(special)
                alias_used = hit.get("_alias", how)
                day_delta = (
                    abs(_date_ord(mq_date) - _date_ord(hit["dataset_date"]))
                    if _date_ord(mq_date) and _date_ord(hit["dataset_date"])
                    else None
                )
                rule = "special_case"

        if hit is None:
            mq_ord = _date_ord(mq_date)
            # 1 bam-equivalent: if only one unique QC date among candidates → direct
            uniq_dates = [d for d in qsub["dataset_date"].unique() if d]
            n_bam = out.loc[out["sample"] == sample, "clean_bam"].nunique()
            if len(uniq_dates) == 1 and n_bam == 1:
                hit = prefer_qc_row(qsub)
                rule = "direct_singleton_alias"
                alias_used = hit.get("_alias", how)
                day_delta = (
                    abs(mq_ord - _date_ord(hit["dataset_date"]))
                    if mq_ord and _date_ord(hit["dataset_date"])
                    else None
                )
            else:
                best = None
                for _, q in qsub.iterrows():
                    q_ord = _date_ord(q["dataset_date"])
                    if mq_ord is None or q_ord is None:
                        delta = 10**9
                    else:
                        delta = abs(q_ord - mq_ord)
                    ds = str(q["dataset"])
                    score = 0 if ds.endswith("-XML") else (1 if "-XML" in ds else 2)
                    # prefer exact alias over hs_swap on tie
                    alias_pen = 0 if q.get("_alias") == "exact" else 1
                    cand = (delta, alias_pen, score, ds, q)
                    if best is None or cand[:4] < best[:4]:
                        best = cand
                assert best is not None
                delta, _, _, _, hit = best
                day_delta = None if delta >= 10**9 else int(delta)
                alias_used = hit.get("_alias", how)
                rule = "nearest_alias"

        new_qc = yyyymmdd(hit["dataset_date"])
        new_puc = hit["puc19"]
        new_lam = hit["lambda"]
        out.at[idx, "qc_batch"] = new_qc
        out.at[idx, "puc19"] = new_puc
        out.at[idx, "lambda"] = new_lam
        changed = (
            old["qc_batch"] != new_qc
            or (pd.isna(old["puc19"]) != pd.isna(new_puc))
            or (
                pd.notna(old["puc19"])
                and pd.notna(new_puc)
                and abs(float(old["puc19"]) - float(new_puc)) > 1e-9
            )
        )
        if changed or alias_used == "hs_swap":
            audit_rows.append(
                {
                    "sample": sample,
                    "mqres_batch": mq_date,
                    "status": "remapped" if changed else "hs_confirmed",
                    "rule": rule,
                    "alias_used": alias_used,
                    "alias_sample": hit.get("_alias_sample"),
                    "day_delta": day_delta,
                    "old_qc_batch": old["qc_batch"],
                    "new_qc_batch": new_qc,
                    "old_puc19": old["puc19"],
                    "new_puc19": new_puc,
                    "old_lambda": old["lambda"],
                    "new_lambda": new_lam,
                }
            )

    return out, pd.DataFrame(audit_rows)


def reorder_meta(meta: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in META_COL_ORDER if c in meta.columns]
    rest = [c for c in meta.columns if c not in ordered]
    return meta[ordered + rest]


def is_evaluable_label(label) -> bool:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return False
    s = str(label).strip()
    if not s or s in IGNORE_LABELS:
        return False
    if "," in s:  # multi-trisomy label
        return False
    return True


def perfect_match_table(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    m["_eval"] = m["label"].map(is_evaluable_label)
    m["_ff_ge"] = m["ff_before_mq"].astype(float) >= 0.01
    m["_match"] = m["_eval"] & (m["label"].astype(str) == m["pred_label"].astype(str))
    rows = []
    for set_name, g in m.groupby("set", dropna=False):
        for ff_name, ff_mask in (
            ("ff_before_mq >= 1%", g["_ff_ge"]),
            ("ff_before_mq < 1%", ~g["_ff_ge"]),
        ):
            sub = g.loc[ff_mask & g["_eval"]]
            n = len(sub)
            n_perf = int(sub["_match"].sum()) if n else 0
            rows.append(
                {
                    "set": set_name,
                    "ff_bin": ff_name,
                    "n_evaluable": n,
                    "n_perfect_match": n_perf,
                    "perfect_match_rate": (n_perf / n) if n else None,
                }
            )
    # overall
    for ff_name, ff_mask in (
        ("ff_before_mq >= 1%", m["_ff_ge"]),
        ("ff_before_mq < 1%", ~m["_ff_ge"]),
    ):
        sub = m.loc[ff_mask & m["_eval"]]
        n = len(sub)
        n_perf = int(sub["_match"].sum()) if n else 0
        rows.append(
            {
                "set": "ALL",
                "ff_bin": ff_name,
                "n_evaluable": n,
                "n_perfect_match": n_perf,
                "perfect_match_rate": (n_perf / n) if n else None,
            }
        )
    return pd.DataFrame(rows)


def batch_set_distribution(meta: pd.DataFrame) -> pd.DataFrame:
    def n_batch(s):
        if pd.isna(s) or str(s).strip() == "":
            return 0
        return len([x for x in str(s).split(",") if x.strip()])

    m = meta.copy()
    m["n_batches"] = m["available_batches"].map(n_batch)
    m["batch_type"] = m["n_batches"].map(
        lambda n: "0" if n == 0 else ("1" if n == 1 else ("2" if n == 2 else "3+"))
    )
    rows = []
    for bt, g in m.groupby("batch_type"):
        row = {"batch_type": bt, "n_samples": int(len(g))}
        for sname, sg in g.groupby("set"):
            row[f"set_{sname}"] = int(len(sg))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("batch_type")


def parse_noisy_removed(review_md: Path) -> Optional[int]:
    if not review_md.is_file():
        return None
    text = review_md.read_text()
    m = re.search(r"Samples affected:\s*\*\*(\d+)\*\*", text)
    return int(m.group(1)) if m else None


def write_summary_md(
    path: Path,
    meta: pd.DataFrame,
    mqres: pd.DataFrame,
    dist: pd.DataFrame,
    perfect: pd.DataFrame,
    audit: pd.DataFrame,
    n_noisy_removed_samples: Optional[int],
) -> None:
    def pct(r):
        if r is None or (isinstance(r, float) and np.isnan(r)):
            return "n/a"
        return f"{100 * r:.1f}%"

    n_miss = int(mqres["puc19"].isna().sum())
    n_miss_s = int(mqres.loc[mqres["puc19"].isna(), "sample"].nunique())
    hs_fix = audit[audit.get("alias_used", pd.Series(dtype=str)) == "hs_swap"] if len(audit) and "alias_used" in audit.columns else audit.iloc[0:0]
    remapped = audit[audit["status"] == "remapped"] if len(audit) else audit

    lines = [
        "# Samplesheet summary (meta + mqres)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- meta samples: **{len(meta)}**",
        f"- mqres rows: **{len(mqres)}** / unique bam: **{mqres['clean_bam'].nunique()}** / samples: **{mqres['sample'].nunique()}**",
        f"- mqres rows still without QC: **{n_miss}** ({n_miss_s} samples; absent from QC DB)",
        "",
        "## 1. Batch-count × set distribution",
        "",
        "Batch count = number of distinct `YYYYMMDD` in meta `available_batches` (post noisy-batch cleanup).",
        "",
    ]
    # table header dynamic
    set_cols = [c for c in dist.columns if c.startswith("set_")]
    sets = [c[len("set_") :] for c in set_cols]
    lines.append(
        "| batch type | n samples | " + " | ".join(sets) + " |"
    )
    lines.append("|------------|----------:|" + "|".join(["----------:"] * len(sets)) + "|")
    for _, r in dist.iterrows():
        cells = [str(r["batch_type"]), str(int(r["n_samples"]))]
        for sc in set_cols:
            cells.append(str(int(r[sc])) if pd.notna(r.get(sc)) else "0")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # also set marginal
    lines += ["### Set marginal", "", "| set | n |", "|-----|--:|"]
    for s, n in meta["set"].value_counts().items():
        lines.append(f"| {s} | {int(n)} |")
    lines.append("")

    lines += [
        "## 2. Multi-batch samples with noisy mqres removed",
        "",
    ]
    if n_noisy_removed_samples is not None:
        lines.append(
            f"- Multi-batch samples that lost ≥1 noisy/bad mqres batch in finalize: "
            f"**{n_noisy_removed_samples}**"
        )
        lines.append(
            "- Global bad batches dropped for multi-batch samples: "
            "`20260203` (lambda), `20260528` (noisy), `20260623` (noisy); "
            "plus per-sample noisy-pred drops (e.g. `20260310`)."
        )
    else:
        lines.append("- (count unavailable — see `samplesheet_review.md`)")
    lines.append("")

    lines += [
        "## 3. Perfect match by set × FF",
        "",
        "Perfect match: `label == pred_label`.",
        "",
        "Excluded from evaluable:",
        "- `label` in `{Twin, Unknown, XO, M21}`",
        "- multi-trisomy `label` containing `,` (e.g. `T1,T2`)",
        "",
        "FF threshold: `ff_before_mq >= 0.01` (1%) vs `< 0.01`.",
        "",
        "| set | FF bin | n evaluable | n perfect | rate |",
        "|-----|--------|------------:|----------:|-----:|",
    ]
    for _, r in perfect.iterrows():
        lines.append(
            f"| {r['set']} | {r['ff_bin']} | {int(r['n_evaluable'])} | "
            f"{int(r['n_perfect_match'])} | {pct(r['perfect_match_rate'])} |"
        )
    lines.append("")

    lines += [
        "## 4. QC remap note (H/S aliases)",
        "",
        "QC sample IDs may use `B` (→`P`) and `H1`/`S1` that differ from mqres.",
        "Example: mqres `PTAY0666P7S1` @ `20250930` was wrongly tied to QC `PTAY0666P7S1` @ `20260327`;",
        "correct chemistry QC is `PTAY0666B7H1` → `PTAY0666P7H1` @ `20250926` (Δ=4d).",
        "",
        "Remap policy: for each mqres row, consider QC from **exact sample + H↔S alias**,",
        "pick nearest `dataset_date` to `mqres_batch` (plus special-case "
        "`20260528`↔`20260602`/`20260529`).",
        "",
        f"- Audit rows logged: **{len(audit)}**",
        f"- Remapped (qc/puc19 changed): **{len(remapped)}**",
        f"- Used H↔S alias: **{len(hs_fix)}**",
        "",
    ]
    if len(remapped):
        lines += [
            "| sample | mqres_batch | old qc | new qc | old puc19 | new puc19 | alias |",
            "|--------|-------------|--------|--------|----------:|----------:|-------|",
        ]
        show = remapped.drop_duplicates(["sample", "mqres_batch"]).sort_values(
            ["sample", "mqres_batch"]
        )
        for _, r in show.iterrows():
            lines.append(
                f"| {r['sample']} | {r['mqres_batch']} | {r.get('old_qc_batch')} | "
                f"{r.get('new_qc_batch')} | {r.get('old_puc19')} | {r.get('new_puc19')} | "
                f"{r.get('alias_used')} |"
            )
        lines.append("")

    lines += [
        "## Files",
        "",
        "| File | Role |",
        "|------|------|",
        "| `meta_samplesheet.csv` | Reordered human-readable columns |",
        "| `mqres_samplesheet.csv` | QC remapped with H/S aliases |",
        "| `samplesheet_summary.md` | This summary |",
        "| `samplesheet_review.md` | Full review job history |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = p.parse_args(argv)
    outdir = args.outdir

    meta = pd.read_csv(outdir / "meta_samplesheet.csv")
    mqres = pd.read_csv(outdir / "mqres_samplesheet.csv")

    # refresh QC from DB
    qc = read_qc_bioinfo_df(as_pandas=True)
    qc["dataset_date"] = qc["dataset_date"].map(yyyymmdd)

    mqres2, audit = remap_mqres_qc(mqres, qc)
    for c in ("mqres_batch", "qc_batch"):
        mqres2[c] = mqres2[c].map(yyyymmdd)
    mqres2 = mqres2[
        ["sample", "deconv_res", "clean_bam", "mqres_batch", "qc_batch", "puc19", "lambda"]
    ].sort_values(["sample", "mqres_batch", "clean_bam", "deconv_res"])
    mqres2.to_csv(outdir / "mqres_samplesheet.csv", index=False)

    meta2 = reorder_meta(meta)
    meta2.to_csv(outdir / "meta_samplesheet.csv", index=False)

    dist = batch_set_distribution(meta2)
    perfect = perfect_match_table(meta2)
    n_noisy = parse_noisy_removed(outdir / "samplesheet_review.md")

    write_summary_md(
        outdir / "samplesheet_summary.md",
        meta2,
        mqres2,
        dist,
        perfect,
        audit,
        n_noisy,
    )
    # do not keep qc table / audit in outdir (user prefers clean dir) — audit embedded in md
    print("meta columns:", list(meta2.columns)[:12], "...")
    print("PTAY0666:", mqres2[mqres2["sample"] == "PTAY0666P7S1"].drop_duplicates(
        ["mqres_batch", "qc_batch", "puc19"]
    ).to_string(index=False))
    print("wrote", outdir / "samplesheet_summary.md")
    print("audit remapped", int((audit["status"] == "remapped").sum()) if len(audit) else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
