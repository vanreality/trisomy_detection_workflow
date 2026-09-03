#!/usr/bin/env python3
"""Build Set A from meta_samplesheet.csv using the 20260813 filter.

Filter (one row per sample):
  depth_qc == pass
  set ∈ {dev, test}
  ff_before_mq >= 0.01
  label not in {Unknown, XO, Twin, M21}
  label does not contain ','

Attaches a score ``unit_id`` from batch-QC viz units:
  preferred_batch_key → score_batch_key → is_preferred_batch → sole unit
  → latest batch_key (multi-batch with no preferred key).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_OUT,
    EXCLUDE_LABELS,
    FF_MIN,
    META_SHEET,
    VIZ_UNITS,
)

console = Console()


def apply_setA_filter(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta.copy()
    df["sample"] = df["sample"].astype(str)
    df["label"] = df["label"].astype(str)
    df["set"] = df["set"].astype(str)
    df["depth_qc"] = df["depth_qc"].astype(str)
    df["ff_before_mq"] = pd.to_numeric(df["ff_before_mq"], errors="coerce")
    lab = df["label"]
    keep = (
        df["depth_qc"].eq("pass")
        & df["set"].isin(["dev", "test"])
        & (df["ff_before_mq"] >= FF_MIN)
        & ~lab.isin(list(EXCLUDE_LABELS))
        & ~lab.str.contains(",", na=False)
    )
    return df.loc[keep].copy()


def _blank(val: object) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip() in ("", "nan", "None", "<NA>")


def pick_unit_id(row: pd.Series, units: pd.DataFrame) -> tuple[str | None, str]:
    sample = str(row["sample"])
    sub = units.loc[units["sample"].astype(str) == sample].copy()
    if sub.empty:
        return None, "no_viz_unit"
    for col, tag in (("preferred_batch_key", "preferred"), ("score_batch_key", "score_batch")):
        key = row.get(col)
        if _blank(key):
            continue
        hit = sub.loc[sub["batch_key"].astype(str) == str(key)]
        if len(hit):
            return str(hit.iloc[0]["unit_id"]), tag
    if "is_preferred_batch" in sub.columns and sub["is_preferred_batch"].astype(bool).any():
        hit = sub.loc[sub["is_preferred_batch"].astype(bool)]
        return str(hit.iloc[0]["unit_id"]), "is_preferred"
    if len(sub) == 1:
        return str(sub.iloc[0]["unit_id"]), "single_batch"
    sub = sub.sort_values("batch_key")
    return str(sub.iloc[-1]["unit_id"]), "latest_batch"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--meta", default=str(META_SHEET), type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--viz-units", default=str(VIZ_UNITS), type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, path_type=Path))
def main(meta: Path, viz_units: Path, output_dir: Path | None) -> None:
    out = output_dir or (DEFAULT_OUT / "cohort")
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(meta)
    raw["sample"] = raw["sample"].astype(str)
    set_a = apply_setA_filter(raw)
    units = pd.read_csv(viz_units)
    units["sample"] = units["sample"].astype(str)

    unit_ids = []
    sources = []
    batch_keys = []
    for _, r in set_a.iterrows():
        uid, src = pick_unit_id(r, units)
        unit_ids.append(uid or "")
        sources.append(src)
        if uid:
            bk = units.loc[units["unit_id"].astype(str) == uid, "batch_key"]
            batch_keys.append(str(bk.iloc[0]) if len(bk) else "")
        else:
            batch_keys.append("")
    set_a = set_a.copy()
    set_a["unit_id"] = unit_ids
    set_a["unit_source"] = sources
    set_a["score_batch_key_used"] = batch_keys

    n_dev = int((set_a["set"] == "dev").sum())
    n_test = int((set_a["set"] == "test").sum())
    n_t = int(set_a["label"].astype(str).str.match(r"^T\d").sum())
    n_n = int((set_a["label"] == "Normal").sum())
    n_no_unit = int((set_a["unit_id"] == "").sum())

    set_a.to_csv(out / "set_A.csv", index=False)
    summary = (
        f"source={meta}\n"
        f"filter=depth_qc==pass; set in {{dev,test}}; ff_before_mq>={FF_MIN}; "
        f"label not in {list(EXCLUDE_LABELS)}; label has no comma\n"
        f"n={len(set_a)} (dev={n_dev}, test={n_test})\n"
        f"N={n_n} T={n_t} other={len(set_a) - n_n - n_t}\n"
        f"no_unit={n_no_unit}\n"
        f"unit_source={set_a['unit_source'].value_counts().to_dict()}\n"
        "note=expected 100 dev + ~198 test; comma-label exclusion drops test from 197 to 194\n"
    )
    (out / "set_A_filter.txt").write_text(summary)
    console.print(summary)
    console.print(f"[green]OK[/green] {out / 'set_A.csv'}")


if __name__ == "__main__":
    main()
