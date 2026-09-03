#!/usr/bin/env python3
"""Randomly pick 96 test-set Normal / depth_qc=pass samples as test_ref_candidates."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from common import (
    DEFAULT_BLACKLIST,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUT_BASE,
    overlay_ff,
)

console = Console()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", default=str(DEFAULT_INPUT_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--n", default=96, show_default=True, type=int)
@click.option("--seed", default=13, show_default=True, type=int)
@click.option("--blacklist", default=",".join(DEFAULT_BLACKLIST), show_default=True)
def main(input_dir: str, output_dir: str | None, n: int, seed: int, blacklist: str) -> None:
    inp = Path(input_dir)
    out = Path(output_dir) if output_dir else DEFAULT_OUT_BASE / "ref_admittance_check"
    out.mkdir(parents=True, exist_ok=True)
    bl = {s.strip() for s in blacklist.split(",") if s.strip()}

    meta = overlay_ff(pd.read_csv(inp / "meta.csv").drop_duplicates("sample", keep="first"))
    meta["sample"] = meta["sample"].astype(str)
    ep = set(pd.read_parquet(inp / "episcore_grid_search.parquet")["sample"].astype(str))
    z = set(pd.read_parquet(inp / "zscore_grid_search.parquet")["sample"].astype(str))
    have = ep & z
    qc = meta["depth_qc"].astype(str).str.lower()
    cand = meta.loc[
        (meta["set"].astype(str) == "test")
        & (meta["label"].astype(str) == "Normal")
        & (qc == "pass")
        & meta["sample"].isin(have)
        & ~meta["sample"].isin(bl)
    ].copy()
    ids = cand["sample"].drop_duplicates().tolist()
    if len(ids) < n:
        raise click.ClickException(f"Need {n} test_ref_candidates, found {len(ids)}")
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(ids, size=n, replace=False).tolist())
    table = cand.loc[cand["sample"].isin(picked), ["sample", "set", "label", "depth_qc", "ff_before_mq"]]
    table = table.drop_duplicates("sample").sort_values("sample")
    dest = out / "test_ref_candidates.tsv"
    table.to_csv(dest, sep="\t", index=False)
    (out / "test_ref_candidates.txt").write_text("\n".join(picked) + "\n")
    leftover = sorted(set(ids) - set(picked))
    (out / "test_normal_not_selected.txt").write_text("\n".join(leftover) + "\n")
    cfg = {
        "n_requested": n,
        "n_eligible": len(ids),
        "n_picked": len(picked),
        "seed": seed,
        "blacklist": sorted(bl),
        "input_dir": str(inp),
        "eligible_rule": "set==test & label==Normal & depth_qc==pass & in episcore/zscore grids & not blacklist",
    }
    (out / "test_ref_candidates.json").write_text(json.dumps(cfg, indent=2) + "\n")
    console.print(f"eligible={len(ids)} picked={len(picked)} seed={seed} -> {dest}")
    console.print(f"  leftover test Normal={len(leftover)}")


if __name__ == "__main__":
    main()
