#!/usr/bin/env python3
"""Push meta / mqres / intermediate sheets into the NIPT NocoDB MySQL database.

Table mapping:
  meta_each_batch_samplesheet.csv          → 生信分析基础信息
  mqres_samplesheet.csv                    → 生信分析文件路径
  intermediate_each_batch_modeA.parquet    → 中游数据

Creates the tables if missing, then replaces all rows.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks" / "aipt_1.0"))
from tools.db_helper import AIPTDatabase  # noqa: E402

DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")
_DATE_RE = re.compile(r"(\d{8})")

TABLES = {
    "meta": "生信分析基础信息",
    "mqres": "生信分析文件路径",
    "intermediate": "中游数据",
}


def yyyymmdd(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if s.lower() in {"", "nan", "none", "nat"}:
        return None
    m = _DATE_RE.search(s)
    return m.group(1) if m else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--mqres", type=Path, default=None)
    p.add_argument("--intermediate", type=Path, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Create tables if needed but do not write rows.",
    )
    return p.parse_args(argv)


def mysql_type(col: str, series: pd.Series) -> str:
    if col in {"deconv_res", "clean_bam", "pred_label"}:
        return "TEXT"
    if col in {"sample", "dataset", "qc_dataset", "depth_qc", "set"}:
        return "VARCHAR(64)"
    if col == "dataset_status":
        return "INT"
    if pd.api.types.is_integer_dtype(series.dropna()):
        return "INT"
    if pd.api.types.is_float_dtype(series) or col.startswith("chr"):
        return "DOUBLE"
    return "TEXT"


def load_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    outdir = args.outdir
    meta = pd.read_csv(args.meta or (outdir / "meta_each_batch_samplesheet.csv"))
    mqres = pd.read_csv(args.mqres or (outdir / "mqres_samplesheet.csv"))
    inter = pd.read_parquet(
        args.intermediate or (outdir / "intermediate_each_batch_modeA.parquet")
    )
    inter = inter.drop(columns=["ff_before_mq", "ff_after_mq"], errors="ignore")
    if "qc_dataset" in meta.columns:
        meta["qc_dataset"] = meta["qc_dataset"].map(yyyymmdd)
    if "dataset_status" in meta.columns:
        meta["dataset_status"] = pd.to_numeric(meta["dataset_status"], errors="coerce")
    return {"meta": meta, "mqres": mqres, "intermediate": inter}


def to_polars(df: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(df, nan_to_null=True)


def ensure_table(db: AIPTDatabase, name: str, df: pd.DataFrame) -> None:
    columns = {c: mysql_type(c, df[c]) for c in df.columns}
    existing = db.list_tables()
    if name not in existing:
        db.create_table(name, columns, if_not_exists=True)
        return
    # Replace contents; schema is assumed to match (created by this script).
    db.clear_table(name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames = load_frames(args)
    with AIPTDatabase() as db:
        for key, table in TABLES.items():
            pdf = frames[key]
            print(f"\n=== {table}  ({key}: {len(pdf)} rows × {len(pdf.columns)} cols) ===")
            ensure_table(db, table, pdf)
            if args.dry_run:
                print("dry-run: skip insert")
                continue
            summary = db.insert_rows(table, to_polars(pdf))
            print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
