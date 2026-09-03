#!/usr/bin/env python3
"""Backfill blacklist rows into pool abnormality_signal_ratio.tsv files.

Old sweeps excluded blacklist from eval, so those samples are missing from
per-pool TSVs. This re-runs the fixed-combo repeat loop (same seeds) and
merges only blacklist rows into existing TSVs without rewriting other samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from pool_size_auc_sweep import (
    CHR_LIST,
    DEFAULT_BLACKLIST,
    _load_fixed_combo_arrays,
    _parse_pool_sizes,
    _run_one_pool,
)

console = Console()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-dir",
    default="/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--sweep-base",
    default="/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--pool-sizes",
    default="20,160,2",
    show_default=True,
    help="min,max,step or comma list of even pool sizes",
)
@click.option(
    "--pool-size",
    default=None,
    type=int,
    help="Run a single pool size (SLURM array task)",
)
@click.option("--total-repeats", default=20000, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--fill-seed", default=7, show_default=True, type=int)
@click.option("--n-jobs", default=8, show_default=True, type=int)
@click.option("--force", is_flag=True, help="Recompute even if blacklist rows already present")
@click.option(
    "--blacklist",
    default=",".join(DEFAULT_BLACKLIST),
    show_default=True,
)
def main(
    input_dir: Path,
    sweep_base: Path,
    pool_sizes: str,
    pool_size: int | None,
    total_repeats: int,
    seed: int,
    fill_seed: int,
    n_jobs: int,
    force: bool,
    blacklist: str,
) -> None:
    parts = [p.strip() for p in pool_sizes.split(",") if p.strip()]
    if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
        lo, hi, step = map(int, parts)
        sizes = list(range(lo, hi + 1, step))
    else:
        sizes = _parse_pool_sizes(pool_sizes)
    if pool_size is not None:
        if pool_size not in sizes:
            sizes = _parse_pool_sizes(str(pool_size))
        else:
            sizes = [pool_size]

    bl = [s.strip() for s in blacklist.split(",") if s.strip()]
    mode_dir = sweep_base / "fixed"

    meta = pd.read_csv(input_dir / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    ep_df = pd.read_parquet(input_dir / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_dir / "zscore_grid_search.parquet")
    universe = sorted(
        set(meta["sample"])
        & set(ep_df["sample"].astype(str))
        & set(z_df["sample"].astype(str))
    )
    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}
    meta_idx = meta.set_index("sample").reindex(universe)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    ff_arr = pd.to_numeric(meta_idx["ff_before_mq"], errors="coerce").to_numpy()

    ep_arrays, z_array = _load_fixed_combo_arrays(
        ep_df, z_df, 0.5, 0.65, 0.85, 0.95, sample_index, chr_index
    )

    console.print(
        f"Backfill blacklist={bl} pools={sizes[0]}..{sizes[-1]} "
        f"repeats={total_repeats} n_jobs={n_jobs}"
    )

    for p in sizes:
        tsv = mode_dir / f"pool_{p}" / "abnormality_signal_ratio.tsv"
        if not tsv.is_file():
            console.print(f"[yellow]skip missing[/yellow] {tsv}")
            continue
        existing = pd.read_csv(tsv, sep="\t")
        existing["sample"] = existing["sample"].astype(str)
        already = set(existing["sample"]) & set(bl)
        if (not force) and already == set(bl):
            console.print(f"pool={p}: blacklist already present, skip compute")
            continue

        console.rule(f"pool={p}")
        pack = _run_one_pool(
            pool_size=p,
            total_repeats=total_repeats,
            seed=seed,
            fill_seed=fill_seed,
            cutoff=3.0,
            ez_cutoff=4.5,
            ff_min=0.01,
            combo_mode="fixed",
            ep_arrays=ep_arrays,
            z_array_or_all=z_array,
            ep_combos=[(0.5, 0.65)],
            z_combos=[(0.85, 0.95)],
            ez_pairs=[(0, 0)],
            set_arr=set_arr,
            label_arr=label_arr,
            ff_arr=ff_arr,
            universe=universe,
            use_fixed=True,
            blacklist=(),  # include blacklist in result
            n_jobs=n_jobs,
        )
        fresh = pack["result"]
        fresh["sample"] = fresh["sample"].astype(str)
        bl_rows = fresh.loc[fresh["sample"].isin(bl)].copy()
        if bl_rows.empty:
            console.print(f"[red]pool={p}: no blacklist rows in result[/red]")
            continue

        keep = existing.loc[~existing["sample"].isin(bl)].copy()
        out = pd.concat([keep, bl_rows], ignore_index=True)
        # stable order: original samples then blacklist
        out.to_csv(tsv, sep="\t", index=False, float_format="%.6f")
        meta_path = mode_dir / f"pool_{p}" / "blacklist_backfill.json"
        meta_path.write_text(
            json.dumps(
                {
                    "pool_size": p,
                    "total_repeats": total_repeats,
                    "seed": seed,
                    "fill_seed": fill_seed,
                    "blacklist": bl,
                    "n_blacklist_rows": int(len(bl_rows)),
                },
                indent=2,
            )
            + "\n"
        )
        console.print(
            f"[green]OK[/green] pool={p} wrote {len(bl_rows)} blacklist rows → {tsv.name}"
        )
        console.print(
            bl_rows[["sample", "ezscore_signal_ratio"]]
            .sort_values("sample")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
