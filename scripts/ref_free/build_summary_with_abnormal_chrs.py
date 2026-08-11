#!/usr/bin/env python3
"""Enrich fixed-combo summary with abnormal chr lists at ez@3 / ez@4.5."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()


def _fmt_abnormal(per_chr: pd.DataFrame, col: str, thr: float = 0.0) -> str:
    hits: list[str] = []
    for _, r in per_chr.iterrows():
        v = float(r[col])
        if v > thr:
            s = f"{v:.4f}".rstrip("0").rstrip(".")
            hits.append(f"{r['chr']}_{s}")
    return ",".join(hits)


@click.command()
@click.option(
    "--result-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="fixed_combo result directory",
)
@click.option("--thr", default=0.0, show_default=True, type=float)
def main(result_dir: Path, thr: float) -> None:
    ref = result_dir / "ref_free_ezscore"
    check_path = result_dir.parent / "input_fixed" / "check_samples.tsv"
    scores = pd.read_csv(ref / "abnormality_signal_ratio.tsv", sep="\t")
    check = pd.read_csv(check_path, sep="\t")
    tab = scores.merge(check[["sample", "batch", "orig_sample"]], on="sample", how="left")

    rows = []
    for _, r in tab.iterrows():
        sid = str(r["sample"])
        pch_path = ref / f"{sid}_per_chr_signal_ratio.tsv"
        if not pch_path.is_file():
            raise click.ClickException(f"Missing per-chr table: {pch_path}")
        pch = pd.read_csv(pch_path, sep="\t")
        rows.append(
            {
                "sample": r["orig_sample"],
                "batch": r["batch"],
                "ff_before_mq": r["ff_before_mq"],
                "episcore": r["episcore_signal_ratio"],
                "zscore": r["zscore_signal_ratio"],
                "ez@3": r["ezscore_signal_ratio_3"],
                "ez@4.5": r["ezscore_signal_ratio"],
                "abnormal_chrs_ez@3": _fmt_abnormal(pch, "ez@3", thr=thr),
                "abnormal_chrs_ez@4.5": _fmt_abnormal(pch, "ez@4.5", thr=thr),
            }
        )
    out = pd.DataFrame(rows).sort_values(["sample", "batch"]).reset_index(drop=True)
    out_path = ref / "summary_table.tsv"
    out.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    console.print(out.to_string(index=False))
    console.print(f"[green]OK[/green] Wrote {out_path}")


if __name__ == "__main__":
    main()
