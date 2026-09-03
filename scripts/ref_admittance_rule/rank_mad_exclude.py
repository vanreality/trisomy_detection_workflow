#!/usr/bin/env python3
"""Rank candidate refs by max(any-chr |percentage MAD-z|, |z_intra MAD-z|) and drop top N.

MAD-z is taken from ``sample_features.tsv`` (computed within that candidate
pool). Fetal fraction is not used. Writes admitted / dropped / size-matched
random lists plus a ranking table.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--features", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--n-drop", default=16, show_default=True, type=int)
@click.option("--random-seed", default=7, show_default=True, type=int)
@click.option("--label", default="", help="Optional cohort name stored in summary JSON.")
def main(
    features: str,
    output_dir: str,
    n_drop: int,
    random_seed: int,
    label: str,
) -> None:
    feat = pd.read_csv(features, sep="\t")
    need = {"sample", "max_abs_pct_madz", "max_abs_intra_madz"}
    missing = need - set(feat.columns)
    if missing:
        raise click.ClickException(f"{features} missing columns: {sorted(missing)}")
    feat = feat.copy()
    feat["sample"] = feat["sample"].astype(str)
    feat["mad_rank_score"] = feat[["max_abs_pct_madz", "max_abs_intra_madz"]].max(axis=1)
    n = len(feat)
    if n_drop < 0 or n_drop >= n:
        raise click.ClickException(f"n-drop={n_drop} invalid for n={n}")

    # Highest score dropped first; ties keep original pool order.
    drop_mask = feat["mad_rank_score"].rank(method="first", ascending=False) <= n_drop
    ranked = feat.assign(
        mad_rank=feat["mad_rank_score"].rank(method="first", ascending=False).astype(int),
        dropped=drop_mask,
    ).sort_values(["mad_rank", "sample"])
    admitted = feat.loc[~drop_mask, "sample"].tolist()
    dropped = feat.loc[drop_mask, "sample"].tolist()

    rng = np.random.default_rng(random_seed)
    rand_mask = np.ones(n, dtype=bool)
    rand_mask[rng.choice(n, size=n_drop, replace=False)] = False
    random_keep = feat.loc[rand_mask, "sample"].tolist()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranked[
        [
            "mad_rank",
            "sample",
            "mad_rank_score",
            "max_abs_pct_madz",
            "max_abs_intra_madz",
            "dropped",
        ]
        + (["ff_before_mq"] if "ff_before_mq" in ranked.columns else [])
    ].to_csv(out / "ranking.tsv", sep="\t", index=False, float_format="%.6f")
    _write_ids(out / "admitted_samples.txt", admitted)
    _write_ids(out / "dropped_samples.txt", dropped)
    _write_ids(out / "random_control_samples.txt", random_keep)
    summary = {
        "label": label,
        "features": str(Path(features).resolve()),
        "n_pool": n,
        "n_drop": n_drop,
        "n_admitted": len(admitted),
        "score": "max(max_abs_pct_madz, max_abs_intra_madz)",
        "uses_ff": False,
        "random_seed": random_seed,
        "dropped_samples": dropped,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print(
        f"{label or out.name}: drop {n_drop}/{n} by MAD-rank -> {out}"
    )
    console.print("  dropped: " + ",".join(dropped))


if __name__ == "__main__":
    main()
