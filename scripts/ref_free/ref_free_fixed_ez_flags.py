#!/usr/bin/env python3
"""Fixed-combo flags with a *fixed* ezscore reference set.

Design
------
* Ezscore refs are fixed from ``ezscore_ref_samples.txt`` (HCPT IDs truncated
  to 8 chars, matching ``load_ezscore_ref_samples``).
* Episcore/zscore refs: each repeat draws ``ref_n`` (default 40) from the
  remaining **dev Normal** pool (all 96 minus the resolved ez refs).
* Writes ``flags_{start}_{end}.npz`` for FP+FN density analysis.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
REF_EXPLORE_DIR = SCRIPT_DIR.parent / "reference_explore"
for p in (REF40_DIR, REF_EXPLORE_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from grid_coverage import assert_dense_coverage, assert_table_coverage  # noqa: E402
from grid_search_ref40 import (  # noqa: E402
    CHR_LIST,
    _build_dense,
    compute_episcore,
    compute_zscore,
)
from calc_zscore_episcore_ezscore import load_ezscore_ref_samples  # noqa: E402
from ref_free_ezscore import (  # noqa: E402
    DEFAULT_REF_N,
    _compute_ezscore,
    _flag_abnormal,
    _load_fixed_combo_arrays,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()

DEFAULT_BLACKLIST = (
    "PTAY0577P9S1",
    "PTAY0599P8S1",
    "PTAY0666P7S1",
    "PTAY0682P7S1",
    "PTAY0689P8H1",
)


def _draw_epiz_refs(
    pool_size: int,
    ref_n: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    if pool_size < ref_n:
        raise ValueError(f"pool_size={pool_size} < ref_n={ref_n}")
    return [
        rng.choice(pool_size, size=ref_n, replace=False).astype(np.int64)
        for _ in range(n_repeats)
    ]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-base", required=True, type=click.Path(file_okay=False))
@click.option(
    "--ezscore-ref-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Default: {input-dir}/ezscore_ref_samples.txt",
)
@click.option("--total-repeats", default=10_000, show_default=True, type=int)
@click.option("--repeat-start", default=0, show_default=True, type=int)
@click.option("--repeat-end", default=None, type=int)
@click.option("--ref-n", default=DEFAULT_REF_N, show_default=True, type=int,
              help="Episcore/zscore reference count drawn each repeat")
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--cutoff", default=3.0, show_default=True, type=float)
@click.option("--ez-cutoff", default=4.5, show_default=True, type=float)
@click.option("--ep-threshold", default=0.5, show_default=True, type=float)
@click.option("--ep-recall", default=0.65, show_default=True, type=float)
@click.option("--z-threshold", default=0.85, show_default=True, type=float)
@click.option("--z-recall", default=0.95, show_default=True, type=float)
@click.option(
    "--blacklist",
    default=",".join(DEFAULT_BLACKLIST),
    show_default=True,
    help="Excluded from eval/flag storage",
)
def main(
    input_dir: str,
    output_base: str,
    ezscore_ref_file: Optional[str],
    total_repeats: int,
    repeat_start: int,
    repeat_end: Optional[int],
    ref_n: int,
    seed: int,
    cutoff: float,
    ez_cutoff: float,
    ep_threshold: float,
    ep_recall: float,
    z_threshold: float,
    z_recall: float,
    blacklist: str,
) -> None:
    input_path = Path(input_dir)
    out_root = Path(output_base) / "fixed_ez_flags"
    out_root.mkdir(parents=True, exist_ok=True)
    if repeat_end is None:
        repeat_end = total_repeats
    if repeat_start < 0 or repeat_end > total_repeats or repeat_end <= repeat_start:
        raise click.ClickException(
            f"Repeat slice [{repeat_start}, {repeat_end}) invalid for total={total_repeats}"
        )
    n_shard = repeat_end - repeat_start
    bl = {s.strip() for s in blacklist.split(",") if s.strip()}
    ez_file = Path(ezscore_ref_file) if ezscore_ref_file else (
        input_path / "ezscore_ref_samples.txt"
    )

    console.rule("[bold blue]Fixed-ez refs + random epi/z refs (flags)")
    console.print(f"  repeats [{repeat_start}, {repeat_end}) / {total_repeats}")
    console.print(f"  ez file : {ez_file}")
    console.print(f"  epi/z n : {ref_n} from remaining dev Normal")
    console.print(f"  ez cut  : {ez_cutoff:g}")

    meta = pd.read_csv(input_path / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    meta["ff_before_mq"] = pd.to_numeric(meta["ff_before_mq"], errors="coerce")
    ep_df = pd.read_parquet(input_path / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_path / "zscore_grid_search.parquet")
    ep_samples = set(ep_df["sample"].astype(str).unique())
    z_samples = set(z_df["sample"].astype(str).unique())
    universe = sorted(set(meta["sample"]) & ep_samples & z_samples)
    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}

    meta_idx = meta.set_index("sample").reindex(universe)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    ff_arr = pd.to_numeric(meta_idx["ff_before_mq"], errors="coerce").to_numpy()
    sample_arr = np.asarray(universe, dtype=str)

    is_trisomy = np.array([bool(re.match(r"^T\d", s)) for s in label_arr])
    is_normal = label_arr == "Normal"
    is_dev_normal = (set_arr == "dev") & is_normal
    is_dev_trisomy = (set_arr == "dev") & is_trisomy
    is_test = set_arr == "test"
    not_blacklisted = ~np.isin(sample_arr, list(bl))

    ez_requested = load_ezscore_ref_samples(ez_file)
    ez_present = [s for s in ez_requested if s in sample_index]
    ez_missing = [s for s in ez_requested if s not in sample_index]
    if not ez_present:
        raise click.ClickException(f"No ezscore refs from {ez_file} found in universe")
    if ez_missing:
        console.print(
            f"[yellow]Warning[/yellow] {len(ez_missing)}/{len(ez_requested)} "
            f"ezscore refs missing (e.g. {ez_missing[:3]}); using {len(ez_present)}"
        )
    ez_ref_idx = np.array([sample_index[s] for s in ez_present], dtype=np.int64)

    # Remaining dev Normal pool for epi/z refs
    ez_set = set(ez_present)
    pool_mask = is_dev_normal & ~np.isin(sample_arr, list(ez_set))
    pool_idx = np.flatnonzero(pool_mask)
    if pool_idx.size < ref_n:
        raise click.ClickException(
            f"Remaining dev Normal pool {pool_idx.size} < ref_n={ref_n}"
        )

    eval_mask = (is_dev_trisomy | is_test) & not_blacklisted
    eval_idx = np.flatnonzero(eval_mask)
    if eval_idx.size == 0:
        raise click.ClickException("No eval samples after blacklist")

    ep_arrays, z_array = _load_fixed_combo_arrays(
        ep_df, z_df, ep_threshold, ep_recall, z_threshold, z_recall,
        sample_index, chr_index,
    )
    ep_dense = np.expand_dims(ep_arrays[0], 0)
    z_dense = np.expand_dims(z_array, 0)
    assert_table_coverage(ep_df, universe, "episcore", [(ep_threshold, ep_recall)])
    assert_table_coverage(z_df, universe, "zscore", [(z_threshold, z_recall)])
    assert_dense_coverage(ep_dense, universe, [(ep_threshold, ep_recall)], "episcore")
    assert_dense_coverage(z_dense, universe, [(z_threshold, z_recall)], "zscore")

    if repeat_start == 0:
        eval_info = pd.DataFrame(
            {
                "sample": [universe[i] for i in eval_idx],
                "set": set_arr[eval_idx],
                "label": label_arr[eval_idx],
                "ff_before_mq": ff_arr[eval_idx],
            }
        )
        eval_info.to_csv(out_root / "eval_samples.tsv", sep="\t", index=False)
        cfg = {
            "mode": "fixed_ez_flags",
            "total_repeats": total_repeats,
            "ref_n": ref_n,
            "ez_ref_n": len(ez_present),
            "ez_ref_requested": len(ez_requested),
            "ez_ref_samples": ez_present,
            "ez_ref_missing": ez_missing,
            "ezscore_ref_file": str(ez_file),
            "n_epiz_pool": int(pool_idx.size),
            "seed": seed,
            "cutoff": cutoff,
            "ez_cutoff": ez_cutoff,
            "ep_threshold": ep_threshold,
            "ep_recall": ep_recall,
            "z_threshold": z_threshold,
            "z_recall": z_recall,
            "n_eval": int(eval_idx.size),
            "blacklist": sorted(bl),
            "eval_sets": ["dev_trisomy", "test"],
        }
        (out_root / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (out_root / "ezscore_ref_samples_used.txt").write_text(
            "\n".join(ez_present) + "\n"
        )
        console.print(
            f"[green]OK[/green] ez_refs={len(ez_present)} "
            f"epiz_pool={pool_idx.size} eval={eval_idx.size}"
        )

    rng = np.random.default_rng(seed)
    epiz_draws = _draw_epiz_refs(pool_idx.size, ref_n, total_repeats, rng)

    n_eval = eval_idx.size
    flags_ep = np.zeros((n_shard, n_eval), dtype=np.uint8)
    flags_z = np.zeros((n_shard, n_eval), dtype=np.uint8)
    flags_ez = np.zeros((n_shard, n_eval), dtype=np.uint8)

    for local_i, repeat_index in enumerate(range(repeat_start, repeat_end)):
        ref_idx = pool_idx[epiz_draws[repeat_index]]
        episcore = compute_episcore(
            np.expand_dims(ep_arrays[0], 0),
            np.expand_dims(ep_arrays[1], 0),
            np.expand_dims(ep_arrays[2], 0),
            np.expand_dims(ep_arrays[3], 0),
            ref_idx,
        )[0]
        zscore = compute_zscore(np.expand_dims(z_array, 0), ref_idx)[0]
        ez = _compute_ezscore(episcore, zscore, ez_ref_idx)
        flags_ep[local_i] = _flag_abnormal(episcore, eval_idx, cutoff).astype(np.uint8)
        flags_z[local_i] = _flag_abnormal(zscore, eval_idx, cutoff).astype(np.uint8)
        flags_ez[local_i] = _flag_abnormal(ez, eval_idx, ez_cutoff).astype(np.uint8)
        if (local_i + 1) % 500 == 0 or local_i + 1 == n_shard:
            console.print(f"  completed {repeat_index + 1}/{repeat_end}")

    out = out_root / f"flags_{repeat_start}_{repeat_end}.npz"
    np.savez_compressed(
        out,
        flags_ep=flags_ep,
        flags_z=flags_z,
        flags_ez=flags_ez,
        repeat_start=np.asarray([repeat_start], dtype=np.int64),
        repeat_end=np.asarray([repeat_end], dtype=np.int64),
        ez_cutoff=np.asarray([ez_cutoff], dtype=np.float64),
    )
    console.print(f"[green]Done[/green] {n_shard} repeats -> {out}")


if __name__ == "__main__":
    main()
