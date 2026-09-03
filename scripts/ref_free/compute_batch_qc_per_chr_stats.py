#!/usr/bin/env python3
"""Per-chr ref_free signal ratios + score mean/std for batch-QC units.

Replays the same 40+40 draws as ``ref_free_ezscore`` (fixed combo) and writes a
long table:

  sample, chr, ep_signal_ratio, z_signal_ratio,
  ez_signal_ratio_<cutoff>..., ep_mean, ep_std, z_mean, z_std, ez_mean, ez_std

Only check units (sample id contains ``__``) are emitted.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
if str(REF40_DIR) not in sys.path:
    sys.path.insert(0, str(REF40_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_search_ref40 import (  # noqa: E402
    CHR_LIST,
    compute_episcore,
    compute_zscore,
)
from ref_free_ezscore import (  # noqa: E402
    _compute_ezscore,
    _generate_half_partitions,
    _load_fixed_combo_arrays,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--result-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-tsv", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--check-only/--all-eval",
    default=True,
    help="Keep only unit_ids containing '__' (batch-QC check samples).",
)
def main(input_dir: str, result_dir: str, output_tsv: str, check_only: bool) -> None:
    input_path = Path(input_dir)
    ref_dir = Path(result_dir) / "ref_free_ezscore"
    cfg = json.loads((ref_dir / "run_config.json").read_text())
    if cfg.get("combo_mode") != "fixed":
        raise click.ClickException("Only combo_mode=fixed supported")

    total_repeats = int(cfg["total_repeats"])
    ref_n = int(cfg["ref_n"])
    seed = int(cfg["seed"])
    cutoff = float(cfg.get("cutoff", 3.0))
    ep_threshold = float(cfg["ep_threshold"])
    ep_recall = float(cfg["ep_recall"])
    z_threshold = float(cfg["z_threshold"])
    z_recall = float(cfg["z_recall"])
    ez_cutoffs = [float(x) for x in cfg.get("ez_cutoffs", [])]
    if not ez_cutoffs:
        ez_cutoffs = [round(3.0 + 0.1 * i, 1) for i in range(16)]

    meta = pd.read_csv(input_path / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    ep_df = pd.read_parquet(input_path / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_path / "zscore_grid_search.parquet")

    # Match universe construction loosely: all samples present in both tables + meta
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
    is_dev_normal = (set_arr == "dev") & (label_arr == "Normal")
    ref_pool_idx = np.flatnonzero(is_dev_normal)
    if ref_pool_idx.size < 2 * ref_n:
        raise click.ClickException(
            f"Need >= {2 * ref_n} dev Normal, found {ref_pool_idx.size}"
        )

    if check_only:
        eval_ids = [i for i, s in enumerate(universe) if "__" in s]
    else:
        eval_ids = list(range(len(universe)))
    if not eval_ids:
        raise click.ClickException("No check samples in universe")
    eval_idx = np.asarray(eval_ids, dtype=np.int64)
    n_eval = eval_idx.size
    n_chr = len(CHR_LIST)

    ep_arrays, z_array = _load_fixed_combo_arrays(
        ep_df,
        z_df,
        ep_threshold,
        ep_recall,
        z_threshold,
        z_recall,
        sample_index,
        chr_index,
    )

    rng = np.random.default_rng(seed)
    ref_local_draws, ez_local_draws = _generate_half_partitions(
        pool_size=ref_pool_idx.size,
        half=ref_n,
        n_repeats=total_repeats,
        rng=rng,
    )

    ep_abn = np.zeros((n_chr, n_eval), dtype=np.int64)
    z_abn = np.zeros((n_chr, n_eval), dtype=np.int64)
    ez_abn = {c: np.zeros((n_chr, n_eval), dtype=np.int64) for c in ez_cutoffs}
    ep_sum = np.zeros((n_chr, n_eval), dtype=np.float64)
    ep_sumsq = np.zeros((n_chr, n_eval), dtype=np.float64)
    z_sum = np.zeros((n_chr, n_eval), dtype=np.float64)
    z_sumsq = np.zeros((n_chr, n_eval), dtype=np.float64)
    ez_sum = np.zeros((n_chr, n_eval), dtype=np.float64)
    ez_sumsq = np.zeros((n_chr, n_eval), dtype=np.float64)

    console.print(
        f"per-chr stats: check={n_eval} repeats={total_repeats} "
        f"ep_cut={cutoff} ez_cutoffs={ez_cutoffs[0]}..{ez_cutoffs[-1]}"
    )
    for r in range(total_repeats):
        ref_idx = ref_pool_idx[ref_local_draws[r]]
        ez_ref_idx = ref_pool_idx[ez_local_draws[r]]
        episcore = compute_episcore(
            np.expand_dims(ep_arrays[0], 0),
            np.expand_dims(ep_arrays[1], 0),
            np.expand_dims(ep_arrays[2], 0),
            np.expand_dims(ep_arrays[3], 0),
            ref_idx,
        )[0]  # [n_chr, n_samples]
        zscore = compute_zscore(np.expand_dims(z_array, 0), ref_idx)[0]
        ez = _compute_ezscore(episcore, zscore, ez_ref_idx)

        ep_e = episcore[:, eval_idx]
        z_e = zscore[:, eval_idx]
        ez_e = ez[:, eval_idx]

        ep_abn += (ep_e > cutoff).astype(np.int64)
        z_abn += (z_e > cutoff).astype(np.int64)
        for c in ez_cutoffs:
            ez_abn[c] += (ez_e > c).astype(np.int64)

        ep_sum += ep_e
        ep_sumsq += ep_e * ep_e
        z_sum += z_e
        z_sumsq += z_e * z_e
        ez_sum += ez_e
        ez_sumsq += ez_e * ez_e

        if (r + 1) % 1000 == 0:
            console.print(f"  {r + 1}/{total_repeats}")

    denom = float(total_repeats)
    rows = []
    for ji, j in enumerate(eval_idx):
        sid = universe[int(j)]
        for hi, chrom in enumerate(CHR_LIST):
            ep_mean = ep_sum[hi, ji] / denom
            z_mean = z_sum[hi, ji] / denom
            ez_mean = ez_sum[hi, ji] / denom
            ep_var = max(ep_sumsq[hi, ji] / denom - ep_mean * ep_mean, 0.0)
            z_var = max(z_sumsq[hi, ji] / denom - z_mean * z_mean, 0.0)
            ez_var = max(ez_sumsq[hi, ji] / denom - ez_mean * ez_mean, 0.0)
            row = {
                "sample": sid,
                "chr": chrom,
                "ep_signal_ratio": ep_abn[hi, ji] / denom,
                "z_signal_ratio": z_abn[hi, ji] / denom,
                "ep_mean": ep_mean,
                "ep_std": float(np.sqrt(ep_var)),
                "z_mean": z_mean,
                "z_std": float(np.sqrt(z_var)),
                "ez_mean": ez_mean,
                "ez_std": float(np.sqrt(ez_var)),
            }
            for c in ez_cutoffs:
                key = f"ez_signal_ratio_{c:g}"
                row[key] = ez_abn[c][hi, ji] / denom
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    console.print(f"[green]OK[/green] Wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
