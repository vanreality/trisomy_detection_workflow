#!/usr/bin/env python3
"""Aggregate ref_free_ezscore slice outputs into per-sample signal ratios."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from separation import separation_for_cutoffs, separation_index
from val_blacklist import drop_blacklisted

console = Console()


def _ez_count_col(cutoff: float) -> str:
    return f"ezscore_abnormal_count_{cutoff:g}"


def _ez_ratio_col(cutoff: float) -> str:
    return f"ezscore_signal_ratio_{cutoff:g}"


def _primary_ez_cutoff(config: dict, ez_cutoffs: list[float]) -> float:
    """Fixed combo → 4.5; filtered/all → 3.0 (fallback to nearest available)."""
    if "primary_ez_cutoff" in config:
        primary = float(config["primary_ez_cutoff"])
    elif str(config.get("combo_mode", "")) == "fixed":
        primary = 4.5
    else:
        primary = 3.0
    if primary in ez_cutoffs:
        return primary
    # Prefer max for fixed (stricter), else first grid value.
    if str(config.get("combo_mode", "")) == "fixed":
        return ez_cutoffs[-1]
    return ez_cutoffs[0]


def _load_slices(out_root: Path, n_eval: int, ez_cutoffs: list[float]):
    ep_counts = np.zeros(n_eval, dtype=np.int64)
    z_counts = np.zeros(n_eval, dtype=np.int64)
    ez_counts = {c: np.zeros(n_eval, dtype=np.int64) for c in ez_cutoffs}
    pair_sum = None
    n_files = 0

    npz_files = sorted(out_root.glob("abnormality_counts_*.npz"))
    tsv_files = sorted(out_root.glob("abnormality_counts_*.tsv"))

    for path in npz_files:
        data = np.load(path)
        pos = data["eval_pos"].astype(np.int64)
        ep_counts[pos] += data["episcore_abnormal_count"].astype(np.int64)
        z_counts[pos] += data["zscore_abnormal_count"].astype(np.int64)
        ez_arr = data["ezscore_abnormal_count"]
        cut = data["ez_cutoffs"] if "ez_cutoffs" in data.files else np.asarray(ez_cutoffs)
        for i, c in enumerate(cut):
            c = float(c)
            if c in ez_counts:
                ez_counts[c][pos] += ez_arr[i].astype(np.int64)
        if "pair_abnormal_count" in data.files:
            pc = data["pair_abnormal_count"].astype(np.int64)
            if pair_sum is None:
                pair_sum = np.zeros_like(pc)
            # align on eval_pos (assume full 0..n_eval-1 slices)
            pair_sum += pc
        n_files += 1

    for path in tsv_files:
        df = pd.read_csv(path, sep="\t")
        pos = df["eval_pos"].to_numpy(dtype=np.int64)
        ep_counts[pos] += df["episcore_abnormal_count"].to_numpy(dtype=np.int64)
        z_counts[pos] += df["zscore_abnormal_count"].to_numpy(dtype=np.int64)
        for c in ez_cutoffs:
            col = _ez_count_col(c)
            if col not in df.columns:
                if "ezscore_abnormal_count" in df.columns and len(ez_cutoffs) == 1:
                    ez_counts[c][pos] += df["ezscore_abnormal_count"].to_numpy(dtype=np.int64)
                else:
                    raise click.ClickException(f"{path} missing column {col}")
            else:
                ez_counts[c][pos] += df[col].to_numpy(dtype=np.int64)
        n_files += 1

    if n_files == 0:
        raise click.ClickException(f"No abnormality_counts_* under {out_root}")
    return ep_counts, z_counts, ez_counts, pair_sum, n_files


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--output-base", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--total-repeats", default=None, type=int)
@click.option("--ff-min", default=0.01, show_default=True, type=float)
def main(output_base: str, total_repeats: int | None, ff_min: float) -> None:
    out_root = Path(output_base) / "ref_free_ezscore"
    eval_path = out_root / "eval_samples.tsv"
    config_path = out_root / "run_config.json"
    if not eval_path.is_file() or not config_path.is_file():
        raise click.ClickException(f"Missing outputs under {out_root}")

    eval_info = pd.read_csv(eval_path, sep="\t")
    config = json.loads(config_path.read_text())
    n_eval = len(eval_info)

    repeats = total_repeats if total_repeats is not None else int(config["total_repeats"])
    ep_denom = float(int(config["n_ep_combos"]) * repeats)
    z_denom = float(int(config["n_z_combos"]) * repeats)
    ez_denom = float(int(config["n_ez_combos"]) * repeats)

    ez_cutoffs = [float(x) for x in config.get("ez_cutoffs", [config.get("ez_cutoff", 3.0)])]

    ep_counts, z_counts, ez_counts, pair_sum, n_files = _load_slices(
        out_root, n_eval, ez_cutoffs
    )

    result = eval_info.copy()
    result["episcore_abnormal_count"] = ep_counts
    result["episcore_signal_ratio"] = ep_counts / ep_denom
    result["zscore_abnormal_count"] = z_counts
    result["zscore_signal_ratio"] = z_counts / z_denom
    for c in ez_cutoffs:
        result[_ez_count_col(c)] = ez_counts[c]
        result[_ez_ratio_col(c)] = ez_counts[c] / ez_denom
    primary_ez = _primary_ez_cutoff(config, ez_cutoffs)
    result["ezscore_abnormal_count"] = result[_ez_count_col(primary_ez)]
    result["ezscore_signal_ratio"] = result[_ez_ratio_col(primary_ez)]

    # Drop val blacklist before publishing ratios / separation
    result = drop_blacklisted(result)

    out_path = out_root / "abnormality_signal_ratio.tsv"
    result.to_csv(out_path, sep="\t", index=False, float_format="%.6f")

    # Split eval (dev/test) vs val for separation reporting
    is_val = result["set"].astype(str).eq("val")
    eval_df = result[~is_val].copy()
    val_df = result[is_val].copy()

    sep_eval = {}
    sep_val = {}
    for name, col in [
        ("episcore", "episcore_signal_ratio"),
        ("zscore", "zscore_signal_ratio"),
    ]:
        sep_eval[name] = separation_index(eval_df, col, ff_min=ff_min)
        if len(val_df):
            sep_val[name] = separation_index(val_df, col, ff_min=ff_min)
    sep_eval["ezscore"] = separation_for_cutoffs(eval_df, ez_cutoffs, ff_min=ff_min)
    if len(val_df):
        sep_val["ezscore"] = separation_for_cutoffs(val_df, ez_cutoffs, ff_min=ff_min)

    if pair_sum is not None:
        pair_path = out_root / "pair_abnormal_count.npz"
        np.savez_compressed(
            pair_path,
            pair_abnormal_count=pair_sum.astype(np.int64, copy=False),
            ez_cutoffs=np.asarray(ez_cutoffs, dtype=np.float64),
            n_repeats=np.asarray([repeats], dtype=np.int64),
        )
        console.print(f"[green]OK[/green] Wrote {pair_path}")

    summary = {
        "total_repeats": repeats,
        "n_ep_combos": int(config["n_ep_combos"]),
        "n_z_combos": int(config["n_z_combos"]),
        "n_ez_combos": int(config["n_ez_combos"]),
        "ez_cutoffs": ez_cutoffs,
        "primary_ez_cutoff": primary_ez,
        "combo_mode": config.get("combo_mode"),
        "episcore_denominator": ep_denom,
        "zscore_denominator": z_denom,
        "ezscore_denominator": ez_denom,
        "n_slice_files": n_files,
        "n_eval_samples": int((~is_val).sum()),
        "n_val_samples": int(is_val.sum()),
        "ff_min": ff_min,
        "separation_eval": sep_eval,
        "separation_val": sep_val,
        "has_pair_counts": pair_sum is not None,
    }
    summary_path = out_root / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    console.print(f"[green]OK[/green] Aggregated {n_files} slice files")
    console.print(f"  -> {out_path}")
    console.print(f"  -> {summary_path}")
    ez_pri = sep_eval.get("ezscore", {}).get(primary_ez, {})
    console.print(
        f"  eval sep@ez{primary_ez:g} AUC={ez_pri.get('sep', float('nan')):.4f} "
        f"(N={ez_pri.get('n_normal')}, T={ez_pri.get('n_trisomy')})"
    )


if __name__ == "__main__":
    main()
