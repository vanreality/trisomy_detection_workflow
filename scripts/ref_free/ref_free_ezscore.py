#!/usr/bin/env python3
"""
Reference-free episcore / zscore / ezscore abnormality signal sweep (40+40).

For each repeat:
    1. From the 96-sample dev Normal pool, draw two disjoint groups of
       ``ref_n`` (default 40): episcore/zscore refs and ezscore refs
       (16 pool samples unused each repeat).
    2. Compute episcore / zscore vs the first group.
    3. Compute ezscore = z-normalize(episcore + zscore) vs the second group.
    4. Flag eval samples when any chromosome exceeds the cutoff.

Episcore/zscore use ``--cutoff``. Ezscore is counted on a cutoff grid
(``--ez-cutoff-min`` .. ``--ez-cutoff-max`` step ``--ez-cutoff-step``) so
downstream interactive plots can slide the ezscore threshold.

Combo modes:
    fixed — one episcore + one zscore combo (paired for ezscore)
    all   — optional threshold/recall filters; ezscore pairs are the
            intersection of identical (thr, recall) keys when non-empty,
            otherwise the cartesian product of filtered ep × z combos
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
if str(REF40_DIR) not in sys.path:
    sys.path.insert(0, str(REF40_DIR))

from grid_coverage import (  # noqa: E402
    assert_dense_coverage,
    assert_table_coverage,
)
from grid_search_ref40 import (  # noqa: E402
    CHR_LIST,
    _build_dense,
    compute_episcore,
    compute_zscore,
)

from val_blacklist import VAL_BLACKLIST  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()

DEFAULT_CUTOFF = 3.0
DEFAULT_REF_N = 40
Combo = Tuple[float, float]
EzPair = Tuple[int, int]  # (ep_combo_index, z_combo_index)


def _combo_index(combos: List[Combo]) -> Dict[Combo, int]:
    return {c: i for i, c in enumerate(combos)}


def _ez_cutoff_grid(lo: float, hi: float, step: float) -> List[float]:
    if step <= 0 or hi < lo:
        raise click.ClickException("Invalid ez cutoff grid")
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 10) for i in range(n)]


def _ez_count_col(cutoff: float) -> str:
    return f"ezscore_abnormal_count_{cutoff:g}"


def _ez_ratio_col(cutoff: float) -> str:
    return f"ezscore_signal_ratio_{cutoff:g}"


def _filter_combo_df(
    df: pd.DataFrame,
    thr_min: Optional[float],
    thr_max: Optional[float],
    rec_min: Optional[float],
    rec_max: Optional[float],
) -> pd.DataFrame:
    out = df
    thr = out["threshold"].astype(float)
    rec = out["recall"].astype(float)
    if thr_min is not None:
        out = out.loc[thr >= thr_min]
        thr, rec = out["threshold"].astype(float), out["recall"].astype(float)
    if thr_max is not None:
        out = out.loc[thr <= thr_max]
        thr, rec = out["threshold"].astype(float), out["recall"].astype(float)
    if rec_min is not None:
        out = out.loc[rec >= rec_min]
        thr, rec = out["threshold"].astype(float), out["recall"].astype(float)
    if rec_max is not None:
        out = out.loc[rec <= rec_max]
    return out


def _require_score_coverage(
    ep_df: pd.DataFrame,
    z_df: pd.DataFrame,
    universe: List[str],
    ep_combos: List[Combo],
    z_combos: List[Combo],
    ep_values: np.ndarray,
    z_values: np.ndarray,
) -> None:
    try:
        assert_table_coverage(ep_df, universe, "episcore", ep_combos)
        assert_table_coverage(z_df, universe, "zscore", z_combos)
        assert_dense_coverage(ep_values, universe, ep_combos, "episcore")
        assert_dense_coverage(z_values, universe, z_combos, "zscore")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _generate_half_partitions(
    pool_size: int,
    half: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw two disjoint halves of size ``half`` from a larger pool.

    When ``pool_size > 2 * half``, the remaining samples are unused that repeat.

    Returns
    -------
    ref_draws, ez_draws : ndarray, shape (n_repeats, half), dtype int64
    """
    need = 2 * half
    if pool_size < need:
        raise ValueError(f"pool_size={pool_size} must be >= 2 * half={need}")
    ref_draws = np.empty((n_repeats, half), dtype=np.int64)
    ez_draws = np.empty((n_repeats, half), dtype=np.int64)
    for i in range(n_repeats):
        perm = rng.permutation(pool_size)
        ref_draws[i] = perm[:half]
        ez_draws[i] = perm[half:need]
    return ref_draws, ez_draws


def _accumulate_combo_flags(
    scores: np.ndarray,
    eval_idx: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    sub = scores[:, :, eval_idx]
    flags = (sub > cutoff).any(axis=1)
    return flags.sum(axis=0).astype(np.int64)


def _flag_abnormal(scores: np.ndarray, eval_idx: np.ndarray, cutoff: float) -> np.ndarray:
    sub = scores[:, eval_idx]
    return (sub > cutoff).any(axis=0).astype(np.int64)


def _flag_abnormal_multi(
    scores: np.ndarray,
    eval_idx: np.ndarray,
    cutoffs: Sequence[float],
) -> np.ndarray:
    """Return [n_cutoff, n_eval] int flags from max-chr scores."""
    sub = scores[:, eval_idx]
    with np.errstate(invalid="ignore"):
        max_chr = np.nanmax(sub, axis=0)
    return np.stack([(max_chr > c).astype(np.int64) for c in cutoffs], axis=0)


def _compute_ezscore(
    episcore: np.ndarray,
    zscore: np.ndarray,
    ez_ref_idx: np.ndarray,
) -> np.ndarray:
    combined = episcore + zscore
    n_chr, _n_sample = combined.shape
    ez = np.empty_like(combined)
    for hi in range(n_chr):
        ref_vals = combined[hi, ez_ref_idx]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(ref_vals)
            sd = np.nanstd(ref_vals, ddof=0)
        mu = mu if np.isfinite(mu) else 0.0
        sd_safe = sd if sd > 0 else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ez[hi] = (combined[hi] - mu) / sd_safe
    return ez


def _accumulate_ez_pairs_multi(
    episcore_all: np.ndarray,
    zscore_all: np.ndarray,
    eval_idx: np.ndarray,
    ez_ref_idx: np.ndarray,
    cutoffs: Sequence[float],
    pairs: Sequence[EzPair],
    *,
    per_pair_out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Accumulate ez flags. If ``per_pair_out`` is [n_pairs, n_cutoffs, n_eval], fill it."""
    counts = np.zeros((len(cutoffs), eval_idx.size), dtype=np.int64)
    for p_i, (ep_i, z_i) in enumerate(pairs):
        ez = _compute_ezscore(episcore_all[ep_i], zscore_all[z_i], ez_ref_idx)
        step = _flag_abnormal_multi(ez, eval_idx, cutoffs)
        counts += step
        if per_pair_out is not None:
            per_pair_out[p_i] += step.astype(np.int32, copy=False)
    return counts


def _build_ez_pairs(
    ep_combos: List[Combo],
    z_combos: List[Combo],
) -> Tuple[List[EzPair], str]:
    """Prefer identical (thr, recall) pairs; else cartesian product."""
    ep_index = _combo_index(ep_combos)
    z_index = _combo_index(z_combos)
    common = sorted(set(ep_combos) & set(z_combos))
    if common:
        pairs = [(ep_index[c], z_index[c]) for c in common]
        return pairs, "intersection"
    pairs = [(ei, zi) for ei in range(len(ep_combos)) for zi in range(len(z_combos))]
    return pairs, "cartesian"


def _load_fixed_combo_arrays(
    ep_df: pd.DataFrame,
    z_df: pd.DataFrame,
    ep_threshold: float,
    ep_recall: float,
    z_threshold: float,
    z_recall: float,
    sample_index: Dict[str, int],
    chr_index: Dict[str, int],
) -> Tuple[List[np.ndarray], np.ndarray]:
    ep_sub = ep_df[
        (ep_df["threshold"].astype(float) == ep_threshold)
        & (ep_df["recall"].astype(float) == ep_recall)
    ]
    z_sub = z_df[
        (z_df["threshold"].astype(float) == z_threshold)
        & (z_df["recall"].astype(float) == z_recall)
    ]
    if ep_sub.empty:
        raise click.ClickException(
            f"No episcore rows for threshold={ep_threshold}, recall={ep_recall}"
        )
    if z_sub.empty:
        raise click.ClickException(
            f"No zscore rows for threshold={z_threshold}, recall={z_recall}"
        )
    _, ep_arrays = _build_dense(
        ep_sub,
        ["hypo_z_intra", "hyper_z_intra", "hypo_cpgs_count", "hyper_cpgs_count"],
        sample_index,
        chr_index,
    )
    _, z_arrays = _build_dense(z_sub, ["percentage"], sample_index, chr_index)
    return [arr[0] for arr in ep_arrays], z_arrays[0][0]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-base", required=True, type=click.Path(file_okay=False))
@click.option("--total-repeats", default=10000, show_default=True, type=int)
@click.option("--repeat-start", default=0, show_default=True, type=int)
@click.option("--repeat-end", default=None, type=int)
@click.option("--ref-n", default=DEFAULT_REF_N, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--cutoff", default=DEFAULT_CUTOFF, show_default=True, type=float)
@click.option("--ez-cutoff-min", default=3.0, show_default=True, type=float)
@click.option("--ez-cutoff-max", default=4.5, show_default=True, type=float)
@click.option("--ez-cutoff-step", default=0.1, show_default=True, type=float)
@click.option("--min-ff", default=0.0, show_default=True, type=float)
@click.option("--combo-mode", default="all", type=click.Choice(["all", "fixed"]))
@click.option("--ep-threshold", default=None, type=float)
@click.option("--ep-recall", default=None, type=float)
@click.option("--z-threshold", default=None, type=float)
@click.option("--z-recall", default=None, type=float)
@click.option("--ep-threshold-min", default=None, type=float)
@click.option("--ep-threshold-max", default=None, type=float)
@click.option("--ep-recall-min", default=None, type=float)
@click.option("--ep-recall-max", default=None, type=float)
@click.option("--z-threshold-min", default=None, type=float)
@click.option("--z-threshold-max", default=None, type=float)
@click.option("--z-recall-min", default=None, type=float)
@click.option("--z-recall-max", default=None, type=float)
@click.option(
    "--store-pair-counts/--no-store-pair-counts",
    default=False,
    show_default=True,
    help="For combo-mode=all, also write per-ez-pair counts (compressed) for subset search",
)
@click.option(
    "--compress/--no-compress",
    default=True,
    show_default=True,
    help="Write slice counts as compressed .npz.npz instead of .tsv",
)
def main(
    input_dir: str,
    output_base: str,
    total_repeats: int,
    repeat_start: int,
    repeat_end: Optional[int],
    ref_n: int,
    seed: int,
    cutoff: float,
    ez_cutoff_min: float,
    ez_cutoff_max: float,
    ez_cutoff_step: float,
    min_ff: float,
    combo_mode: str,
    ep_threshold: Optional[float],
    ep_recall: Optional[float],
    z_threshold: Optional[float],
    z_recall: Optional[float],
    ep_threshold_min: Optional[float],
    ep_threshold_max: Optional[float],
    ep_recall_min: Optional[float],
    ep_recall_max: Optional[float],
    z_threshold_min: Optional[float],
    z_threshold_max: Optional[float],
    z_recall_min: Optional[float],
    z_recall_max: Optional[float],
    store_pair_counts: bool,
    compress: bool,
) -> None:
    """Run reference-free episcore/zscore/ezscore abnormality sweep."""
    ez_cutoffs = _ez_cutoff_grid(ez_cutoff_min, ez_cutoff_max, ez_cutoff_step)
    input_path = Path(input_dir)
    out_root = Path(output_base) / "ref_free_ezscore"
    out_root.mkdir(parents=True, exist_ok=True)

    if repeat_end is None:
        repeat_end = total_repeats
    if repeat_start < 0 or repeat_end > total_repeats or repeat_end <= repeat_start:
        raise click.ClickException(
            f"Repeat slice [{repeat_start}, {repeat_end}) must lie within [0, {total_repeats})"
        )

    use_fixed = combo_mode == "fixed"
    if use_fixed:
        missing = [
            name
            for name, val in (
                ("ep-threshold", ep_threshold),
                ("ep-recall", ep_recall),
                ("z-threshold", z_threshold),
                ("z-recall", z_recall),
            )
            if val is None
        ]
        if missing:
            raise click.ClickException(f"--combo-mode fixed requires: {', '.join(missing)}")

    console.rule("[bold blue]Reference-free episcore/zscore/ezscore sweep")
    console.print(f"  Input dir      : {input_path}")
    console.print(f"  Output root    : {out_root}")
    console.print(f"  Repeat range   : [{repeat_start}, {repeat_end}) of {total_repeats}")
    console.print(f"  ref split      : {ref_n} + {ref_n} (from 96-sample pool)")
    console.print(f"  combo-mode     : {combo_mode}")
    console.print(f"  ep/z cutoff    : {cutoff}")
    console.print(
        f"  ez cutoff grid : {ez_cutoffs[0]:g} .. {ez_cutoffs[-1]:g} "
        f"step {ez_cutoff_step:g} (n={len(ez_cutoffs)})"
    )

    meta = pd.read_csv(input_path / "meta.csv")
    for col in ("sample", "set", "label", "ff_before_mq"):
        if col not in meta.columns:
            raise click.ClickException(f"meta.csv missing column: {col}")
    meta = meta.drop_duplicates("sample", keep="first").copy()
    meta["sample"] = meta["sample"].astype(str)
    meta["ff_before_mq"] = pd.to_numeric(meta["ff_before_mq"], errors="coerce")

    console.print("[cyan]Loading parquets ...[/cyan]")
    ep_df = pd.read_parquet(input_path / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_path / "zscore_grid_search.parquet")

    if not use_fixed:
        ep_df = _filter_combo_df(
            ep_df, ep_threshold_min, ep_threshold_max, ep_recall_min, ep_recall_max
        )
        z_df = _filter_combo_df(
            z_df, z_threshold_min, z_threshold_max, z_recall_min, z_recall_max
        )
        if ep_df.empty or z_df.empty:
            raise click.ClickException("Combo filters removed all episcore or zscore rows")

    ep_samples = set(ep_df["sample"].astype(str).unique())
    z_samples = set(z_df["sample"].astype(str).unique())
    meta_samples = set(meta["sample"])
    ff_pass = set(meta.loc[meta["ff_before_mq"] > min_ff, "sample"].astype(str))
    universe = sorted(meta_samples & ep_samples & z_samples & ff_pass)
    if not universe:
        raise click.ClickException("No samples remain after filters")

    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}

    meta_idx = meta.set_index("sample").reindex(universe)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    ff_arr = pd.to_numeric(meta_idx["ff_before_mq"], errors="coerce").to_numpy()

    label_str = label_arr.astype(str)
    is_trisomy = np.array([bool(re.match(r"^T\d", s)) for s in label_str])
    is_normal = label_str == "Normal"
    is_dev_normal = (set_arr == "dev") & is_normal
    is_dev_trisomy = (set_arr == "dev") & is_trisomy
    is_test = set_arr == "test"
    # Independent validation samples tagged set=val (Normal or Trisomy only)
    sample_arr = np.asarray(universe, dtype=str)
    not_blacklisted = ~np.isin(sample_arr, list(VAL_BLACKLIST))
    is_val = (set_arr == "val") & (is_normal | is_trisomy) & not_blacklisted
    eval_mask = is_dev_trisomy | is_test | is_val
    ref_pool_idx = np.flatnonzero(is_dev_normal)
    eval_idx = np.flatnonzero(eval_mask)

    min_pool = 2 * ref_n
    if ref_pool_idx.size < min_pool:
        raise click.ClickException(
            f"Need at least {min_pool} dev Normal samples for a {ref_n}+{ref_n} "
            f"draw, found {ref_pool_idx.size}"
        )
    if eval_idx.size == 0:
        raise click.ClickException("No evaluation samples (dev trisomy + test)")

    if use_fixed:
        assert ep_threshold is not None and ep_recall is not None
        assert z_threshold is not None and z_recall is not None
        ep_arrays, z_array = _load_fixed_combo_arrays(
            ep_df, z_df, ep_threshold, ep_recall, z_threshold, z_recall,
            sample_index, chr_index,
        )
        ep_combos: List[Combo] = [(ep_threshold, ep_recall)]
        z_combos = [(z_threshold, z_recall)]
        ez_pairs: List[EzPair] = [(0, 0)]
        ez_pair_mode = "fixed"
        z_array_all = None
        ep_dense = np.expand_dims(ep_arrays[0], 0)
        z_dense = np.expand_dims(z_array, 0)
    else:
        ep_combos, ep_arrays = _build_dense(
            ep_df,
            ["hypo_z_intra", "hyper_z_intra", "hypo_cpgs_count", "hyper_cpgs_count"],
            sample_index,
            chr_index,
        )
        z_combos, z_arrays = _build_dense(z_df, ["percentage"], sample_index, chr_index)
        z_array_all = z_arrays[0]
        ez_pairs, ez_pair_mode = _build_ez_pairs(ep_combos, z_combos)
        ep_dense = ep_arrays[0]
        z_dense = z_array_all

    _require_score_coverage(
        ep_df, z_df, universe, ep_combos, z_combos, ep_dense, z_dense
    )
    console.print("[green]OK[/green] episcore/zscore parquet coverage complete")
    console.print(f"  universe samples : {len(universe)}")
    console.print(f"  dev Normal pool  : {ref_pool_idx.size}")
    console.print(
        f"  eval samples     : {eval_idx.size} "
        f"(dev trisomy={int(is_dev_trisomy.sum())}, test={int(is_test.sum())}, "
        f"val={int(is_val.sum())})"
    )
    console.print(f"  episcore combos  : {len(ep_combos)}")
    console.print(f"  zscore combos    : {len(z_combos)}")
    console.print(f"  ezscore pairs    : {len(ez_pairs)} ({ez_pair_mode})")

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
        # Fixed combo reports / plots default to ez=4.5; filtered grid keeps 3.0.
        primary_ez_cutoff = 4.5 if use_fixed else 3.0
        if primary_ez_cutoff not in ez_cutoffs:
            primary_ez_cutoff = ez_cutoffs[-1] if use_fixed else ez_cutoffs[0]
        run_config = {
            "combo_mode": combo_mode,
            "ref_n": ref_n,
            "ez_ref_n": ref_n,
            "normal_pool_size": int(ref_pool_idx.size),
            "cutoff": cutoff,
            "ez_cutoff_min": ez_cutoff_min,
            "ez_cutoff_max": ez_cutoff_max,
            "ez_cutoff_step": ez_cutoff_step,
            "ez_cutoffs": ez_cutoffs,
            "primary_ez_cutoff": primary_ez_cutoff,
            "ez_pair_mode": ez_pair_mode,
            "total_repeats": total_repeats,
            "seed": seed,
            "n_ep_combos": len(ep_combos),
            "n_z_combos": len(z_combos),
            "n_ez_combos": len(ez_pairs),
            "episcore_denominator": len(ep_combos) * total_repeats,
            "zscore_denominator": len(z_combos) * total_repeats,
            "ezscore_denominator": len(ez_pairs) * total_repeats,
            "ep_threshold_min": ep_threshold_min,
            "ep_threshold_max": ep_threshold_max,
            "ep_recall_min": ep_recall_min,
            "ep_recall_max": ep_recall_max,
            "z_threshold_min": z_threshold_min,
            "z_threshold_max": z_threshold_max,
            "z_recall_min": z_recall_min,
            "z_recall_max": z_recall_max,
            "store_pair_counts": bool(store_pair_counts and not use_fixed),
            "compress": compress,
            "n_val_samples": int(is_val.sum()),
        }
        if use_fixed:
            run_config.update(
                {
                    "ep_threshold": ep_threshold,
                    "ep_recall": ep_recall,
                    "z_threshold": z_threshold,
                    "z_recall": z_recall,
                }
            )
        else:
            run_config["ez_pairs"] = [[int(a), int(b)] for a, b in ez_pairs]
            run_config["ep_combos"] = [[float(a), float(b)] for a, b in ep_combos]
            run_config["z_combos"] = [[float(a), float(b)] for a, b in z_combos]
        (out_root / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        console.print(f"[green]OK[/green] Wrote {out_root / 'eval_samples.tsv'}")

    rng = np.random.default_rng(seed)
    ref_local_draws, ez_local_draws = _generate_half_partitions(
        pool_size=ref_pool_idx.size,
        half=ref_n,
        n_repeats=total_repeats,
        rng=rng,
    )

    n_eval = eval_idx.size
    ep_counts = np.zeros(n_eval, dtype=np.int64)
    z_counts = np.zeros(n_eval, dtype=np.int64)
    ez_counts = np.zeros((len(ez_cutoffs), n_eval), dtype=np.int64)
    do_pairs = bool(store_pair_counts and not use_fixed)
    pair_counts = (
        np.zeros((len(ez_pairs), len(ez_cutoffs), n_eval), dtype=np.int32)
        if do_pairs
        else None
    )

    for repeat_index in range(repeat_start, repeat_end):
        ref_idx = ref_pool_idx[ref_local_draws[repeat_index]]
        ez_ref_idx = ref_pool_idx[ez_local_draws[repeat_index]]

        if use_fixed:
            episcore = compute_episcore(
                np.expand_dims(ep_arrays[0], 0),
                np.expand_dims(ep_arrays[1], 0),
                np.expand_dims(ep_arrays[2], 0),
                np.expand_dims(ep_arrays[3], 0),
                ref_idx,
            )[0]
            zscore = compute_zscore(np.expand_dims(z_array, 0), ref_idx)[0]
            ep_step = _flag_abnormal(episcore, eval_idx, cutoff)
            z_step = _flag_abnormal(zscore, eval_idx, cutoff)
            ez = _compute_ezscore(episcore, zscore, ez_ref_idx)
            ez_step = _flag_abnormal_multi(ez, eval_idx, ez_cutoffs)
        else:
            assert z_array_all is not None
            episcore_all = compute_episcore(
                ep_arrays[0], ep_arrays[1], ep_arrays[2], ep_arrays[3], ref_idx
            )
            zscore_all = compute_zscore(z_array_all, ref_idx)
            ep_step = _accumulate_combo_flags(episcore_all, eval_idx, cutoff)
            z_step = _accumulate_combo_flags(zscore_all, eval_idx, cutoff)
            ez_step = _accumulate_ez_pairs_multi(
                episcore_all,
                zscore_all,
                eval_idx,
                ez_ref_idx,
                ez_cutoffs,
                ez_pairs,
                per_pair_out=pair_counts,
            )

        ep_counts += ep_step
        z_counts += z_step
        ez_counts += ez_step
        done = repeat_index - repeat_start + 1
        if done % 50 == 0 or done == repeat_end - repeat_start:
            console.print(f"  completed repeat {repeat_index + 1}/{repeat_end}")

    stem = f"abnormality_counts_{repeat_start}_{repeat_end}"
    if compress:
        payload = {
            "eval_pos": np.arange(n_eval, dtype=np.int64),
            "episcore_abnormal_count": ep_counts,
            "zscore_abnormal_count": z_counts,
            "ez_cutoffs": np.asarray(ez_cutoffs, dtype=np.float64),
            "ezscore_abnormal_count": ez_counts.astype(np.int64, copy=False),
        }
        if pair_counts is not None:
            payload["pair_abnormal_count"] = pair_counts
        slice_path = out_root / f"{stem}.npz"
        np.savez_compressed(slice_path, **payload)
    else:
        slice_data = {
            "eval_pos": np.arange(n_eval, dtype=np.int64),
            "episcore_abnormal_count": ep_counts,
            "zscore_abnormal_count": z_counts,
        }
        for i, c in enumerate(ez_cutoffs):
            slice_data[_ez_count_col(c)] = ez_counts[i]
        slice_path = out_root / f"{stem}.tsv"
        pd.DataFrame(slice_data).to_csv(slice_path, sep="\t", index=False)
    console.print(f"[green]Done[/green] {repeat_end - repeat_start} repeats -> {slice_path}")


if __name__ == "__main__":
    main()
