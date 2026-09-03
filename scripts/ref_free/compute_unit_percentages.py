#!/usr/bin/env python3
"""Compute per-chr read-count percentages for batch-QC units.

For each unit in ``unit_samplesheet.csv``, filter deconv reads at
``--threshold`` overlapping CpGs in the recall list, then write:

  {out}/{unit_id}.percentage.tsv   columns: chr, percentage

Optional ``--unit-id`` / ``--index`` for SLURM array shards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console

# Reuse overlap logic from pipeline bin/
_BIN = Path(__file__).resolve().parents[2] / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from calc_zscore_flexible import (  # noqa: E402
    CHR_LIST,
    combo_percentages,
    read_recall_positions,
)

console = Console()

DEFAULT_CPG_DIR = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260525-grid_search_240k_panel_240k_model/recall_list_220k"
)


def _read_deconv(
    path: Path,
    mtcount: float,
    min_prob: float | None = None,
) -> pl.DataFrame:
    """Load deconv reads; optionally pre-filter by prob to shrink huge TXT tables.

    Filter+collect first, then ``unique`` in memory. Streaming ``unique`` has
    hung on some large Lustre TXT/parquet inputs in batch-QC arrays.
    """
    cols = ["chr", "start", "end", "text", "prob_class_1", "mTcount"]
    is_parquet = path.suffix == ".parquet" or str(path).endswith(".parquet")
    console.print(f"  reading {path.name} ({'parquet' if is_parquet else 'txt'}) ...")
    if is_parquet:
        lf = pl.scan_parquet(path).select(cols)
    else:
        lf = pl.scan_csv(
            path,
            separator="\t",
            infer_schema_length=10000,
        ).select(cols)

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
    filt = pl.col("mTcount") >= mtcount
    if min_prob is not None:
        filt = filt & (pl.col("prob_class_1") >= min_prob)
    lf = lf.filter(filt)
    # Do not use the streaming engine: it has hung for hours on some Lustre
    # parquet reads (e.g. JPTAY1823P7H1 after-MQ %) while in-memory collect
    # of the same file finishes in seconds.
    df = lf.collect()
    console.print(f"  collected n={df.height:,}; dedup ...")
    df = df.unique(subset=["chr", "start", "end", "text"], keep="first")
    console.print(f"  after dedup n={df.height:,}")
    return df


def _compute_one(
    unit_id: str,
    deconv_res: str,
    out_path: Path,
    threshold: float,
    cpg_positions: dict,
    mtcount: float,
    force: bool,
) -> None:
    if out_path.is_file() and not force:
        console.print(f"  skip existing {out_path.name}")
        return
    # Pre-filter at z-threshold so multi-GB TXT deconvs stay tractable.
    df = _read_deconv(Path(deconv_res), mtcount, min_prob=threshold)
    pct = combo_percentages(df, threshold, cpg_positions, mtcount)
    out = pd.DataFrame({"chr": CHR_LIST, "percentage": pct})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False, float_format="%.8f")
    console.print(f"  [green]OK[/green] {unit_id} -> {out_path.name}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--threshold", required=True, type=float)
@click.option("--recall", required=True, type=float)
@click.option("--cpg-recall-dir", default=DEFAULT_CPG_DIR, type=click.Path(exists=True, file_okay=False))
@click.option("--mtcount", default=1.0, show_default=True, type=float)
@click.option("--unit-id", default=None, help="Process a single unit_id.")
@click.option("--index", default=None, type=int, help="0-based row index into units CSV.")
@click.option("--force", is_flag=True, default=False)
def main(
    units: str,
    output_dir: str,
    threshold: float,
    recall: float,
    cpg_recall_dir: str,
    mtcount: float,
    unit_id: str | None,
    index: int | None,
    force: bool,
) -> None:
    udf = pd.read_csv(units)
    if "unit_id" not in udf.columns:
        raise click.ClickException("units CSV needs unit_id column")
    if unit_id is not None:
        udf = udf[udf["unit_id"].astype(str) == unit_id]
    elif index is not None:
        if index < 0 or index >= len(udf):
            raise click.ClickException(f"index {index} out of range 0..{len(udf)-1}")
        udf = udf.iloc[[index]]

    if udf.empty:
        raise click.ClickException("No units to process")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cpg_positions = read_recall_positions(Path(cpg_recall_dir), recall)
    console.print(
        f"units={len(udf)} thr={threshold} recall={recall} out={out_root}"
    )

    for _, r in udf.iterrows():
        uid = str(r["unit_id"])
        deconv = str(r["deconv_res"])
        if not Path(deconv).is_file():
            console.print(f"[red]Missing deconv[/red] {uid}: {deconv}")
            continue
        _compute_one(
            uid,
            deconv,
            out_root / f"{uid}.percentage.tsv",
            threshold,
            cpg_positions,
            mtcount,
            force,
        )


if __name__ == "__main__":
    main()
