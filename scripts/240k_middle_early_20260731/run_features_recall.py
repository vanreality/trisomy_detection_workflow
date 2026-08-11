#!/usr/bin/env python3
"""Per-recall feature table: chrX_percentage + chrX_*_z_intra / s_intra for early+middle.

Outputs ``{output_prefix}/features.tsv.gz``.

Note: prefer assembling the feature matrix from male_ref/female_ref outputs via
``collect_tables.py`` (avoids re-reading all deconv files). This script remains
for a dedicated features sweep if needed; uses spawn to avoid fork+polars hangs.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR / "episcore") not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR / "episcore"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _pct_worker(args):
    """Worker entry: import zmod inside child (spawn-safe)."""
    import run_zscore_recall as zmod

    sample, deconv_paths, cutoff, cpg_list = args
    zmod._init_worker(cpg_list)
    df = zmod.read_deconv_paths(deconv_paths, cutoff=cutoff, mtcount=1.0)
    pct = zmod.chromosome_percentages(df, zmod._CPG_POSITIONS)
    return sample, pct.get("chrX", float("nan"))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--samples-meta", required=True, type=click.Path(exists=True))
@click.option("--cpg-list", required=True, type=click.Path(exists=True))
@click.option("--output-prefix", required=True, type=str)
@click.option("--cutoff", type=float, default=0.85, show_default=True)
@click.option("--ncpus", type=int, default=4, show_default=True)
def main(
    samples_meta: str,
    cpg_list: str,
    output_prefix: str,
    cutoff: float,
    ncpus: int,
) -> None:
    from episcore_fast_calculator import _prepare_worker_ctx
    import beta_to_zscore as b2z

    meta = pd.read_csv(samples_meta)
    out_dir = Path(output_prefix.rstrip("/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.tsv.gz"
    if out_path.is_file():
        console.print(f"[yellow]Skip existing[/yellow] {out_path}")
        return

    console.print(f"Samples: {len(meta)}  cpus={ncpus}  cutoff={cutoff}")

    # --- percentages (spawn avoids fork deadlock with polars/singularity) ---
    pct_tasks = [
        (str(r.sample), str(r.deconv_paths), float(cutoff), str(cpg_list))
        for r in meta.itertuples(index=False)
        if pd.notna(r.deconv_paths)
    ]
    pct_map: dict[str, float] = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=ncpus, mp_context=ctx) as pool:
        futs = {pool.submit(_pct_worker, t): t[0] for t in pct_tasks}
        for fut in as_completed(futs):
            sample, px = fut.result()
            pct_map[sample] = px
            console.print(f"  pct {sample} chrX={px:.6g}")

    # --- z_intra from betas ---
    beta_samples = {
        str(r.sample): str(r.beta_path)
        for r in meta.itertuples(index=False)
        if pd.notna(getattr(r, "beta_path", None))
    }
    chromosomes, _ = _prepare_worker_ctx(
        chr_list="1-22,X",
        beta_cols=(
            "chr,start,end,target_meth_count,target_unmeth_count,"
            "raw_total_count,meandiff"
        ),
        cpg_list=cpg_list,
        exclude_cpg_list=None,
        depth=None,
        depth_col="raw_total_count",
    )
    intra_map: dict[str, dict[str, float]] = {}
    if beta_samples:
        results = b2z._run_pool(beta_samples, ncpus, "Intra samples")
        chr_idx = {c: i for i, c in enumerate(chromosomes)}
        xi = chr_idx.get("chrX")
        if xi is None:
            raise click.ClickException("chrX missing from chromosome list")
        for r in results:
            if r.get("error") is not None:
                console.print(f"[yellow]beta fail[/yellow] {r['sample']}: {r['error']}")
                continue
            intra_map[r["sample"]] = {
                "chrX_hypo_z_intra": float(r["hypo_z_intra"][xi]),
                "chrX_hyper_z_intra": float(r["hyper_z_intra"][xi]),
                "chrX_s_intra": float(r["s_intra"][xi]),
            }
            console.print(f"  intra {r['sample']} s_intra={r['s_intra'][xi]:.4g}")

    rows = []
    for _, row in meta.iterrows():
        sample = row["sample"]
        intra = intra_map.get(sample, {})
        rows.append(
            {
                "sample": sample,
                "dataset": row.get("dataset"),
                "cohort": row.get("cohort"),
                "label": row.get("label"),
                "chrX_percentage": pct_map.get(sample, np.nan),
                "chrX_hypo_z_intra": intra.get("chrX_hypo_z_intra", np.nan),
                "chrX_hyper_z_intra": intra.get("chrX_hyper_z_intra", np.nan),
                "chrX_s_intra": intra.get("chrX_s_intra", np.nan),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_path, sep="\t", index=False, float_format="%.6f", compression="gzip"
    )
    console.print(f"[green]Wrote[/green] {out_path}  rows={len(rows)}")


if __name__ == "__main__":
    main()
