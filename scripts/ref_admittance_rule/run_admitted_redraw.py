#!/usr/bin/env python3
"""Prospective 40+40 redraw from an admitted (or random-N) pool.

Thin wrapper around ``score_repeats.py`` with a pool file. Prefer the SLURM
submit path for 10k–50k repeats:

    POOL_SAMPLES=.../admitted_samples.txt TAG=admitted SEED=7 TOTAL_REPEATS=20000 \\
      bash submit_score_repeats.sh

This CLI is for small local/smoke redraws.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--pool-samples", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tag", default="admitted", show_default=True)
@click.option("--total-repeats", default=20000, show_default=True, type=int)
@click.option("--seed", default=7, show_default=True, type=int)
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--input-dir", default=None, type=click.Path(exists=True, file_okay=False))
def main(
    pool_samples: str,
    tag: str,
    total_repeats: int,
    seed: int,
    output_dir: str | None,
    input_dir: str | None,
) -> None:
    argv = [
        "score_repeats.py",
        "--pool-samples",
        pool_samples,
        "--tag",
        tag,
        "--total-repeats",
        str(total_repeats),
        "--seed",
        str(seed),
    ]
    if output_dir:
        argv += ["--output-dir", output_dir]
    if input_dir:
        argv += ["--input-dir", input_dir]
    sys.argv = argv
    runpy.run_path(str(Path(__file__).resolve().parent / "score_repeats.py"), run_name="__main__")


if __name__ == "__main__":
    main()
