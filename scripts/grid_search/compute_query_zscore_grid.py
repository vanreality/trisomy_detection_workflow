#!/usr/bin/env python3
"""Compute the full zscore (percentage) grid for one query unit.

Reads the deconv table once, then evaluates every (threshold, recall) in
``--combos`` (the previous grid-search combo list). Writes a long parquet with
the same schema as ``zscore_grid_search.parquet``:

    sample, chr, threshold, recall, percentage
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
import polars as pl
from rich.console import Console

from t21_combo_common import CHR_LIST, DEFAULT_CPG_DIR, fmt_combo, read_combo_csv

_BIN = Path(__file__).resolve().parents[2] / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from calc_zscore_flexible import combo_percentages, read_recall_positions  # noqa: E402

console = Console()


def _read_deconv(path: Path, mtcount: float, min_prob: float) -> pl.DataFrame:
    cols = ["chr", "start", "end", "text", "prob_class_1", "mTcount"]
    is_parquet = path.suffix == ".parquet" or str(path).endswith(".parquet")
    console.print(f"  reading {path.name} ({'parquet' if is_parquet else 'txt'}) ...")
    if is_parquet:
        lf = pl.scan_parquet(path).select(cols)
    else:
        lf = pl.scan_csv(path, separator="\t", infer_schema_length=10000).select(cols)
    lf = lf.with_columns(
        [
            pl.col("chr").cast(pl.Utf8),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            pl.col("prob_class_1").cast(pl.Float64),
            pl.col("mTcount").cast(pl.Float64),
        ]
    )
    lf = lf.with_columns(
        pl.when(pl.col("chr").str.starts_with("chr"))
        .then(pl.col("chr"))
        .otherwise(pl.lit("chr") + pl.col("chr"))
        .alias("chr")
    )
    lf = lf.filter((pl.col("mTcount") >= mtcount) & (pl.col("prob_class_1") >= min_prob))
    try:
        df = lf.collect(engine="streaming")
    except TypeError:
        df = lf.collect(streaming=True)
    console.print(f"  collected n={df.height:,}; dedup ...")
    df = df.unique(subset=["chr", "start", "end", "text"], keep="first")
    console.print(f"  after dedup n={df.height:,}")
    return df


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--combos", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--cpg-recall-dir", default=str(DEFAULT_CPG_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--mtcount", default=1.0, show_default=True, type=float)
@click.option("--unit-id", default=None)
@click.option("--index", default=None, type=int)
@click.option("--force", is_flag=True, default=False)
def main(
    units: str,
    combos: str,
    output_dir: str,
    cpg_recall_dir: str,
    mtcount: float,
    unit_id: str | None,
    index: int | None,
    force: bool,
) -> None:
    udf = pd.read_csv(units)
    if unit_id is not None:
        udf = udf[udf["unit_id"].astype(str) == unit_id]
    elif index is not None:
        udf = udf.iloc[[index]]
    if len(udf) != 1:
        raise click.ClickException(f"Expected one unit, got {len(udf)}")
    row = udf.iloc[0]
    uid = str(row["unit_id"])
    deconv = Path(str(row["deconv_res"]))
    if not deconv.is_file():
        raise click.ClickException(f"Missing deconv: {deconv}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{uid}.parquet"
    if out_path.is_file() and not force:
        console.print(f"skip existing {out_path}")
        return

    combo_list = read_combo_csv(Path(combos))
    thresholds = sorted({t for t, _ in combo_list})
    recalls = sorted({r for _, r in combo_list})
    min_thr = min(thresholds)
    console.print(
        f"{uid}: {len(combo_list)} combos "
        f"({len(thresholds)} thr x {len(recalls)} recall unique)"
    )

    df = _read_deconv(deconv, mtcount, min_prob=min_thr)
    recall_dir = Path(cpg_recall_dir)
    recall_cache: dict[str, dict] = {}
    for rec in recalls:
        key = fmt_combo(rec)
        console.print(f"  load recall {key}")
        recall_cache[key] = read_recall_positions(recall_dir, rec)

    records = []
    n = len(combo_list)
    for i, (thr, rec) in enumerate(combo_list, start=1):
        if i == 1 or i % 100 == 0 or i == n:
            console.print(f"  combo {i}/{n} thr={fmt_combo(thr)} recall={fmt_combo(rec)}")
        pct = combo_percentages(df, thr, recall_cache[fmt_combo(rec)], mtcount)
        for chrom, value in zip(CHR_LIST, pct):
            records.append(
                {
                    "sample": uid,
                    "chr": chrom,
                    "threshold": float(thr),
                    "recall": float(rec),
                    "percentage": float(value),
                }
            )

    out = pd.DataFrame.from_records(records)
    out.to_parquet(out_path, index=False, compression="snappy")
    console.print(f"[green]OK[/green] {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()
