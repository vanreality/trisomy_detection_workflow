#!/usr/bin/env python3
"""Collect conventional / middle-ref recall curves and the feature matrix."""

from __future__ import annotations

import re
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

import config as cfg

console = Console()
RECALL_DIR_RE = re.compile(r"^recall_([\d.]+)$")


def _collect_score(
    recall_dir: Path,
    score_col: str,
    out_col: str,
    labels: pd.DataFrame,
    early_only: bool = True,
) -> pd.DataFrame:
    rows = []
    for out_dir in sorted(recall_dir.glob("recall_*")):
        m = RECALL_DIR_RE.match(out_dir.name)
        if not m:
            continue
        recall = float(m.group(1))
        path = out_dir / "_analyze_zscore.tsv.gz"
        if not path.is_file():
            continue
        df = pd.read_csv(path, sep="\t")
        if score_col not in df.columns:
            console.print(f"[yellow]Skip[/yellow] no {score_col} in {path}")
            continue
        for _, row in df.iterrows():
            sample = row["sample"]
            lab = labels.loc[sample] if sample in labels.index else None
            if early_only and lab is not None and lab.get("dataset") == "middle":
                continue
            rows.append(
                {
                    "sample": sample,
                    "recall": recall,
                    out_col: row[score_col],
                    "label": None if lab is None else lab["label"],
                    "cohort": None if lab is None else lab["cohort"],
                    "dataset": None if lab is None else lab.get("dataset"),
                    "ff_before_mq": None,
                }
            )
    return pd.DataFrame(rows)


def _attach_ff(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not cfg.CHRY_FF_TSV.is_file():
        return df
    ff = (
        pd.read_csv(cfg.CHRY_FF_TSV, sep="\t", usecols=["sample", "ff_before_mq"])
        .drop_duplicates("sample")
        .set_index("sample")["ff_before_mq"]
    )
    df = df.copy()
    df["ff_before_mq"] = df["sample"].map(ff)
    return df


def _load_intra_table(path: Path) -> pd.DataFrame:
    cols = [
        "sample",
        "chrX_hypo_z_intra",
        "chrX_hyper_z_intra",
        "chrX_s_intra",
    ]
    df = pd.read_csv(path, sep="\t")
    keep = [c for c in cols if c in df.columns]
    return df[keep]


def _features_from_middle_refs(labels: pd.DataFrame) -> list[pd.DataFrame]:
    """Build per-recall feature frames from male_ref + female_ref outputs.

    ``chrX_percentage`` comes from zscore analyze tables; ``z_intra`` / ``s_intra``
    from episcore analyze + reference tables. Male-ref and female-ref together
    cover all early + middle samples.
    """
    frames: list[pd.DataFrame] = []
    z_dirs = [cfg.MALE_REF_ZSCORE_DIR, cfg.FEMALE_REF_ZSCORE_DIR]
    e_dirs = [cfg.MALE_REF_EPISCORE_DIR, cfg.FEMALE_REF_EPISCORE_DIR]
    recalls = set()
    for d in z_dirs + e_dirs:
        if d.is_dir():
            for p in d.glob("recall_*"):
                m = RECALL_DIR_RE.match(p.name)
                if m:
                    recalls.add(m.group(1))
    for recall_s in sorted(recalls, key=lambda x: float(x)):
        recall = float(recall_s)
        pct_parts = []
        for d in z_dirs:
            path = d / f"recall_{recall_s}" / "_analyze_zscore.tsv.gz"
            if path.is_file():
                pct_parts.append(
                    pd.read_csv(path, sep="\t", usecols=["sample", "chrX_percentage"])
                )
        intra_parts = []
        for d in e_dirs:
            for name in ("_analyze_zscore.tsv.gz", "_reference_zscore.tsv.gz"):
                path = d / f"recall_{recall_s}" / name
                if path.is_file():
                    intra_parts.append(_load_intra_table(path))
        if not pct_parts or not intra_parts:
            continue
        pct = pd.concat(pct_parts, ignore_index=True).drop_duplicates("sample", keep="last")
        intra = pd.concat(intra_parts, ignore_index=True).drop_duplicates(
            "sample", keep="last"
        )
        merged = pct.merge(intra, on="sample", how="outer")
        merged["recall"] = recall
        merged["label"] = merged["sample"].map(
            labels["label"] if "label" in labels.columns else None
        )
        merged["cohort"] = merged["sample"].map(
            labels["cohort"] if "cohort" in labels.columns else None
        )
        merged["dataset"] = merged["sample"].map(
            labels["dataset"] if "dataset" in labels.columns else None
        )
        frames.append(merged)
    return frames


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    labels = pd.read_csv(cfg.COHORT_LABELS).set_index("sample")
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Conventional episcore
    if cfg.EPISCORE_RECALL_DIR.is_dir():
        epi = _collect_score(
            cfg.EPISCORE_RECALL_DIR, "chrX_s_inter", "chrX_episcore", labels
        )
        # Merge prior curves for samples not recomputed
        if cfg.PRIOR_EPISCORE_TSV.is_file():
            prior = pd.read_csv(cfg.PRIOR_EPISCORE_TSV, sep="\t")
            prior["cohort"] = prior["cohort"].replace({"old": "old_early", "new": "old_early"})
            prior["dataset"] = "early"
            have = set(zip(epi["sample"], epi["recall"])) if len(epi) else set()
            extra = prior[
                ~prior.apply(lambda r: (r["sample"], r["recall"]) in have, axis=1)
            ]
            epi = pd.concat([epi, extra], ignore_index=True)
        epi = _attach_ff(epi)
        epi.to_csv(cfg.EPISCORE_COLLECTED, sep="\t", index=False)
        console.print(
            f"[green]Wrote[/green] {cfg.EPISCORE_COLLECTED}  rows={len(epi)}"
        )

    # Conventional zscore
    if cfg.ZSCORE_RECALL_DIR.is_dir():
        zdf = _collect_score(
            cfg.ZSCORE_RECALL_DIR, "chrX_zscore", "chrX_zscore", labels
        )
        if cfg.PRIOR_ZSCORE_TSV.is_file():
            prior = pd.read_csv(cfg.PRIOR_ZSCORE_TSV, sep="\t")
            prior["cohort"] = prior["cohort"].replace({"old": "old_early", "new": "old_early"})
            prior["dataset"] = "early"
            have = set(zip(zdf["sample"], zdf["recall"])) if len(zdf) else set()
            extra = prior[
                ~prior.apply(lambda r: (r["sample"], r["recall"]) in have, axis=1)
            ]
            zdf = pd.concat([zdf, extra], ignore_index=True)
        zdf = _attach_ff(zdf)
        zdf.to_csv(cfg.ZSCORE_COLLECTED, sep="\t", index=False)
        console.print(f"[green]Wrote[/green] {cfg.ZSCORE_COLLECTED}  rows={len(zdf)}")

    # Male-ref / female-ref
    for recall_dir, col, out_path, score_name in (
        (
            cfg.MALE_REF_EPISCORE_DIR,
            "chrX_s_inter",
            cfg.MALE_REF_EPISCORE_COLLECTED,
            "chrX_episcore",
        ),
        (
            cfg.MALE_REF_ZSCORE_DIR,
            "chrX_zscore",
            cfg.MALE_REF_ZSCORE_COLLECTED,
            "chrX_zscore",
        ),
        (
            cfg.FEMALE_REF_EPISCORE_DIR,
            "chrX_s_inter",
            cfg.FEMALE_REF_EPISCORE_COLLECTED,
            "chrX_episcore",
        ),
        (
            cfg.FEMALE_REF_ZSCORE_DIR,
            "chrX_zscore",
            cfg.FEMALE_REF_ZSCORE_COLLECTED,
            "chrX_zscore",
        ),
    ):
        if not recall_dir.is_dir():
            continue
        df = _collect_score(recall_dir, col, score_name, labels, early_only=True)
        df = _attach_ff(df)
        df.to_csv(out_path, sep="\t", index=False)
        console.print(f"[green]Wrote[/green] {out_path}  rows={len(df)}")

    # Feature matrix: prefer dedicated features_recall outputs; else assemble from
    # male_ref + female_ref (union covers all early + middle).
    feat_rows = []
    if cfg.FEATURES_DIR.is_dir():
        for out_dir in sorted(cfg.FEATURES_DIR.glob("recall_*")):
            m = RECALL_DIR_RE.match(out_dir.name)
            if not m:
                continue
            path = out_dir / "features.tsv.gz"
            if not path.is_file():
                continue
            df = pd.read_csv(path, sep="\t")
            df["recall"] = float(m.group(1))
            feat_rows.append(df)

    if not feat_rows:
        feat_rows = _features_from_middle_refs(labels)
        if feat_rows:
            console.print(
                "[cyan]Assembled feature matrix from male_ref + female_ref outputs[/cyan]"
            )

    if feat_rows:
        feat = pd.concat(feat_rows, ignore_index=True)
        feat = _attach_ff(feat)
        feat["gender"] = feat["label"]
        feat.to_csv(cfg.FEATURE_MATRIX_TSV, sep="\t", index=False)
        console.print(
            f"[green]Wrote[/green] {cfg.FEATURE_MATRIX_TSV}  "
            f"rows={len(feat)} recalls={feat['recall'].nunique()}"
        )


if __name__ == "__main__":
    main()
