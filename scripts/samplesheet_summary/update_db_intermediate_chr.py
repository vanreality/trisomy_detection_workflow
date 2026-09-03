#!/usr/bin/env python3
"""Patch missing chr columns in DB table ``中游数据`` from the local parquet.

Fetches the live table, copies chromosome features from
``intermediate_each_batch_modeA.parquet`` onto rows where those cells are
empty, and writes those columns back. Fetal-fraction columns are never
updated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
import polars as pl
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks" / "aipt_1.0"))
from tools.db_helper import AIPTDatabase  # noqa: E402

console = Console()
DEFAULT_PARQUET = Path(
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/"
    "intermediate_each_batch_modeA.parquet"
)
TABLE = "中游数据"
KEY_COLS = ["sample", "dataset"]


def _chr_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if str(c).startswith("chr")]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--parquet",
    default=str(DEFAULT_PARQUET),
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--dry-run", is_flag=True, default=False)
def main(parquet: str, dry_run: bool) -> None:
    src = pd.read_parquet(parquet)
    src_chr = _chr_cols(list(src.columns))
    if not src_chr:
        raise click.ClickException(f"No chr columns in {parquet}")

    console.print(f"Fetching `{TABLE}` ...")
    with AIPTDatabase() as db:
        db_pl = db.fetch_table(TABLE, drop_system_cols=True)
        pk_col = [c for c in db_pl.columns if str(c).endswith("_id") and "中游数据" in c]
        if not pk_col:
            raise click.ClickException(f"No PK column in fetch: {db_pl.columns[:8]}")
        pk = pk_col[0]
        db_df = db_pl.to_pandas()
        db_chr = _chr_cols(list(db_df.columns))
        shared_chr = [c for c in db_chr if c in src_chr]
        extra_db = sorted(set(db_chr) - set(src_chr))
        extra_src = sorted(set(src_chr) - set(db_chr))
        console.print(
            f"DB rows={len(db_df)} chr={len(db_chr)}  "
            f"parquet rows={len(src)} chr={len(src_chr)}  shared={len(shared_chr)}"
        )
        if extra_db:
            console.print(f"[yellow]chr in DB not in parquet (left unchanged): {len(extra_db)}[/yellow]")
        if extra_src:
            console.print(f"[yellow]chr in parquet not in DB (skipped): {len(extra_src)}[/yellow]")
        if not shared_chr:
            raise click.ClickException("No overlapping chr columns")

        # Rows where DB has NA chr that parquet has filled (not only all-empty rows).
        look = src[["sample", "dataset", *shared_chr]].copy()
        look = look.rename(columns={c: f"_src_{c}" for c in shared_chr})
        merged = db_df.merge(look, on=KEY_COLS, how="inner")
        need = pd.Series(False, index=merged.index)
        n_filled = 0
        for c in shared_chr:
            src_c = f"_src_{c}"
            hit = merged[c].isna() & merged[src_c].notna()
            n_filled += int(hit.sum())
            need = need | hit
            merged.loc[hit, c] = merged.loc[hit, src_c]
        patch = merged.loc[need].copy()
        console.print(
            f"DB rows with chr NA fillable from parquet: {len(patch)}  "
            f"cells={n_filled}"
        )
        if patch.empty:
            console.print("[green]Nothing to update[/green]")
            return

        out = patch[[pk, *KEY_COLS, *shared_chr]].copy()
        out_pl = pl.from_pandas(out, nan_to_null=True)
        console.print(
            f"writing chr cells with new values ≈ {n_filled} "
            f"(dry_run={dry_run})"
        )
        summary = db.update_rows_columns(
            TABLE,
            out_pl,
            columns=shared_chr,
            verify_columns=KEY_COLS,
            dry_run=dry_run,
        )
        console.print(summary)


if __name__ == "__main__":
    main()
