#!/usr/bin/env python3
"""Per-chromosome ref_free signal ratios for one sample (fixed combo).

Replays the same seed / 40+40 draws as ``ref_free_ezscore`` and counts, per
chromosome, the fraction of repeats where that chromosome exceeds the cutoff.
"""

from __future__ import annotations

import json
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
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Slim fixed-combo input (meta + ep/z parquets)",
)
@click.option(
    "--result-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="fixed_combo result dir containing ref_free_ezscore/run_config.json",
)
@click.option("--sample", required=True, type=str)
@click.option("--output-tsv", default=None, type=click.Path(dir_okay=False))
def main(input_dir: str, result_dir: str, sample: str, output_tsv: str | None) -> None:
    input_path = Path(input_dir)
    ref_dir = Path(result_dir) / "ref_free_ezscore"
    cfg = json.loads((ref_dir / "run_config.json").read_text())
    if cfg.get("combo_mode") != "fixed":
        raise click.ClickException("Only combo_mode=fixed is supported")

    total_repeats = int(cfg["total_repeats"])
    ref_n = int(cfg["ref_n"])
    seed = int(cfg["seed"])
    cutoff = float(cfg.get("cutoff", 3.0))
    ep_threshold = float(cfg["ep_threshold"])
    ep_recall = float(cfg["ep_recall"])
    z_threshold = float(cfg["z_threshold"])
    z_recall = float(cfg["z_recall"])
    ez_cutoffs = [float(x) for x in cfg.get("ez_cutoffs", [3.0, 4.5])]
    want_ez = [c for c in (3.0, 4.5) if c in ez_cutoffs]
    if not want_ez:
        want_ez = [ez_cutoffs[0], ez_cutoffs[-1]]

    meta = pd.read_csv(input_path / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    ep_df = pd.read_parquet(input_path / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_path / "zscore_grid_search.parquet")

    universe = sorted(
        set(meta["sample"])
        & set(ep_df["sample"].astype(str))
        & set(z_df["sample"].astype(str))
    )
    if sample not in universe:
        raise click.ClickException(f"{sample} not in input universe")

    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}
    meta_idx = meta.set_index("sample").reindex(universe)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    is_normal = label_arr == "Normal"
    is_dev_normal = (set_arr == "dev") & is_normal
    ref_pool_idx = np.flatnonzero(is_dev_normal)
    sample_i = sample_index[sample]

    if ref_pool_idx.size < 2 * ref_n:
        raise click.ClickException(
            f"Need >= {2 * ref_n} dev Normal, found {ref_pool_idx.size}"
        )

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

    n_chr = len(CHR_LIST)
    ep_counts = np.zeros(n_chr, dtype=np.int64)
    z_counts = np.zeros(n_chr, dtype=np.int64)
    ez_counts = {c: np.zeros(n_chr, dtype=np.int64) for c in want_ez}

    console.print(
        f"Per-chr signal for {sample}: repeats={total_repeats} "
        f"cutoff={cutoff} ez={want_ez}"
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
        )[0]
        zscore = compute_zscore(np.expand_dims(z_array, 0), ref_idx)[0]
        ez = _compute_ezscore(episcore, zscore, ez_ref_idx)

        ep_counts += (episcore[:, sample_i] > cutoff).astype(np.int64)
        z_counts += (zscore[:, sample_i] > cutoff).astype(np.int64)
        for c in want_ez:
            ez_counts[c] += (ez[:, sample_i] > c).astype(np.int64)

        if (r + 1) % 1000 == 0:
            console.print(f"  {r + 1}/{total_repeats}")

    denom = float(total_repeats)
    rows = []
    for hi, chrom in enumerate(CHR_LIST):
        row = {
            "chr": chrom,
            "episcore": ep_counts[hi] / denom,
            "zscore": z_counts[hi] / denom,
            "ez@3": ez_counts[3.0][hi] / denom if 3.0 in ez_counts else np.nan,
            "ez@4.5": ez_counts[4.5][hi] / denom if 4.5 in ez_counts else np.nan,
        }
        rows.append(row)
    out_df = pd.DataFrame(rows)

    # Sanity: any-chr union should match sample-level ratios when available
    scores_path = ref_dir / "abnormality_signal_ratio.tsv"
    if scores_path.is_file():
        sample_row = pd.read_csv(scores_path, sep="\t")
        sample_row = sample_row[sample_row["sample"].astype(str) == sample]
        if len(sample_row):
            console.print(
                "sample-level (any chr) "
                f"ep={float(sample_row['episcore_signal_ratio'].iloc[0]):.4f} "
                f"z={float(sample_row['zscore_signal_ratio'].iloc[0]):.4f} "
                f"ez3={float(sample_row.get('ezscore_signal_ratio_3', sample_row['ezscore_signal_ratio']).iloc[0]):.4f} "
                f"ez45={float(sample_row['ezscore_signal_ratio'].iloc[0]):.4f}"
            )

    out_path = (
        Path(output_tsv)
        if output_tsv
        else ref_dir / f"{sample}_per_chr_signal_ratio.tsv"
    )
    out_df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    console.print(f"[green]OK[/green] Wrote {out_path}")
    console.print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
