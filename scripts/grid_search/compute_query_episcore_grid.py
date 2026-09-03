#!/usr/bin/env python3
"""Compute the episcore (z_intra) recall grid for one query unit at one threshold.

Requires a full-panel beta file from Nextflow ``EXTRACT_BETA`` (grid_search
``CpG_final_recall.txt``), not the production recall-0.65 beta. For every
recall listed for ``--threshold`` in ``--combos``, filter CpGs, aggregate
hypo/hyper beta, and compute within-sample ``z_intra``.

Writes a long parquet matching ``episcore_grid_search.parquet``:

    sample, chr, threshold, recall,
    hypo_z_intra, hyper_z_intra, hypo_cpgs_count, hyper_cpgs_count
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from beta_to_episcore import (
    calculate_chr_level_beta,
    calculate_s_intra,
    read_beta_filtered,
)
from t21_combo_common import (
    BETA_DEPTH,
    CHR_LIST,
    DEFAULT_CPG_DIR,
    find_nf_beta,
    fmt_combo,
    read_combo_csv,
)

console = Console()

BETA_COLS = [
    "chr",
    "start",
    "end",
    "target_meth_count",
    "target_unmeth_count",
    "raw_meth_count",
    "raw_unmeth_count",
    "raw_total_count",
    "meandiff",
]


def _load_cpg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["chr", "start", "end"])
    df["chr"] = df["chr"].astype(str)
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    return df


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--combos", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--threshold", required=True, type=float)
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--cpg-recall-dir", default=str(DEFAULT_CPG_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--nf-outdir", default=None, type=click.Path(file_okay=False))
@click.option("--beta-path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--depth", default=BETA_DEPTH, show_default=True, type=int)
@click.option("--unit-id", default=None)
@click.option("--index", default=None, type=int)
@click.option("--force", is_flag=True, default=False)
def main(
    units: str,
    combos: str,
    threshold: float,
    output_dir: str,
    cpg_recall_dir: str,
    nf_outdir: str | None,
    beta_path: str | None,
    depth: int,
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

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{uid}.parquet"
    if out_path.is_file() and not force:
        console.print(f"skip existing {out_path}")
        return

    recalls = sorted({r for t, r in read_combo_csv(Path(combos)) if abs(t - threshold) < 1e-9})
    if not recalls:
        raise click.ClickException(f"No recalls in combo list for threshold={threshold:g}")

    beta = None
    if beta_path:
        beta = Path(beta_path)
    elif nf_outdir:
        beta = find_nf_beta(Path(nf_outdir), uid)
    if beta is None or not beta.is_file():
        raise click.ClickException(
            f"No beta for {uid} at thr={threshold:g}. "
            "Run Nextflow EXTRACT_BETA (grid_search profile) first."
        )
    console.print(f"{uid} thr={fmt_combo(threshold)} beta={beta} recalls={len(recalls)}")

    available = set(pd.read_csv(beta, sep="\t", compression="gzip", nrows=0).columns)
    usecols = [c for c in BETA_COLS if c in available]
    for required in ("chr", "start", "end", "meandiff"):
        if required not in usecols:
            usecols.append(required)

    df0 = read_beta_filtered(
        str(beta),
        usecols=usecols,
        chr_list=CHR_LIST,
        cpg_filter_df=None,
        filter_depth=depth,
        depth_col="raw_total_count" if "raw_total_count" in usecols else None,
    )
    if "raw_meth_count" in df0.columns and "raw_unmeth_count" in df0.columns:
        meth_col, unmeth_col = "raw_meth_count", "raw_unmeth_count"
    elif "target_meth_count" in df0.columns and "target_unmeth_count" in df0.columns:
        meth_col, unmeth_col = "target_meth_count", "target_unmeth_count"
    else:
        raise click.ClickException(f"No meth/unmeth columns in {beta}")
    console.print(f"  beta rows after chr/depth filter: {len(df0):,}")

    cpg_dir = Path(cpg_recall_dir)
    records = []
    for rec in recalls:
        cpg_path = cpg_dir / f"220k_cpg_recall_{fmt_combo(rec)}.txt"
        if not cpg_path.is_file():
            raise click.ClickException(f"Missing CpG list: {cpg_path}")
        cpg = _load_cpg(cpg_path)
        df = df0.merge(cpg, on=["chr", "start", "end"], how="inner", copy=False)
        hypo_beta, hyper_beta, hypo_counts, hyper_counts = calculate_chr_level_beta(
            df, CHR_LIST, meth_col, unmeth_col
        )
        hypo_z, hyper_z, _s_intra = calculate_s_intra(
            hypo_beta, hyper_beta, hypo_counts, hyper_counts
        )
        for i, chrom in enumerate(CHR_LIST):
            records.append(
                {
                    "sample": uid,
                    "chr": chrom,
                    "threshold": float(threshold),
                    "recall": float(rec),
                    "hypo_z_intra": float(hypo_z[i]),
                    "hyper_z_intra": float(hyper_z[i]),
                    "hypo_cpgs_count": int(hypo_counts[i]),
                    "hyper_cpgs_count": int(hyper_counts[i]),
                }
            )
        console.print(f"  recall={fmt_combo(rec)} n_cpg={len(df):,}")

    out = pd.DataFrame.from_records(records)
    out.to_parquet(out_path, index=False, compression="snappy")
    console.print(f"[green]OK[/green] {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()
