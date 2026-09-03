#!/usr/bin/env python3
"""Harvest NF extract_beta outputs into batch-QC per-unit episcore TSVs.

Looks for ``{nf_outdir}/extract_beta_value/{unit_id}_beta_value.tsv.gz``
(or nested publishDir layouts), runs ``beta_to_episcore.py`` with the mode
CpG list, and writes ``{ep_dir}/{unit_id}.episcore.tsv``.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()

BIN_EP = "/lustre1/cqyi/AIPT_2.0/workflow/episcore/bin/beta_to_episcore.py"
CHR_LIST = [f"chr{i}" for i in range(1, 23)]

CPG = {
    (0.5, 0.65): "/lustre1/cqyi/AIPT_2.0/workflow/episcore/assets/CpG_recall0.65.txt",
    (0.1, 0.61): (
        "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
        "20260525-grid_search_240k_panel_240k_model/recall_list_220k/220k_cpg_recall_0.61.txt"
    ),
}


def _find_beta(nf_outdir: Path, unit_id: str) -> Path | None:
    patterns = [
        f"**/extract_beta_value/{unit_id}_beta_value.tsv.gz",
        f"**/{unit_id}_beta_value.tsv.gz",
    ]
    for pat in patterns:
        hits = sorted(nf_outdir.glob(pat))
        if hits:
            return hits[0]
    return None


def _melt(sample: str, row: pd.Series, threshold: float, recall: float) -> pd.DataFrame:
    records = []
    for chr_name in CHR_LIST:
        records.append(
            {
                "sample": sample,
                "chr": chr_name,
                "threshold": threshold,
                "recall": recall,
                "hypo_z_intra": float(row[f"{chr_name}_hypo_z_intra"]),
                "hyper_z_intra": float(row[f"{chr_name}_hyper_z_intra"]),
                "hypo_cpgs_count": float(row[f"{chr_name}_hypo_cpgs_count"]),
                "hyper_cpgs_count": float(row[f"{chr_name}_hyper_cpgs_count"]),
            }
        )
    return pd.DataFrame.from_records(records)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--nf-outdir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--ep-dir", required=True, type=click.Path(file_okay=False))
@click.option("--threshold", required=True, type=float)
@click.option("--recall", required=True, type=float)
@click.option("--depth", default=30, type=int)
@click.option("--force", is_flag=True, default=False)
def main(
    units: str,
    nf_outdir: str,
    ep_dir: str,
    threshold: float,
    recall: float,
    depth: int,
    force: bool,
) -> None:
    udf = pd.read_csv(units)
    out = Path(ep_dir)
    out.mkdir(parents=True, exist_ok=True)
    nf = Path(nf_outdir)
    cpg = CPG.get((threshold, recall))
    if cpg is None:
        raise click.ClickException(f"No default CpG list for thr={threshold} recall={recall}")

    ok = miss = skip = 0
    for uid in udf["unit_id"].astype(str):
        dest = out / f"{uid}.episcore.tsv"
        if dest.is_file() and not force:
            skip += 1
            continue
        beta = _find_beta(nf, uid)
        if beta is None:
            miss += 1
            continue
        with tempfile.TemporaryDirectory(prefix="harvest_ep_") as tmp:
            prefix = str(Path(tmp) / uid)
            # Caller is expected to already be inside the common_tools container
            # (nested singularity is unavailable on compute nodes).
            subprocess.run(
                [
                    "python3",
                    BIN_EP,
                    "--beta-value",
                    str(beta),
                    "--output-prefix",
                    prefix,
                    "--depth",
                    str(depth),
                    "--cpg-list",
                    cpg,
                    "--chr-list",
                    "1-22",
                ],
                check=True,
            )
            wide = pd.read_csv(f"{prefix}_zscore.tsv", sep="\t")
            _melt(uid, wide.iloc[0], threshold, recall).to_csv(dest, sep="\t", index=False)
        ok += 1
        console.print(f"  [green]OK[/green] {uid} <- {beta}")
    console.print(f"done ok={ok} skip={skip} miss={miss}")


if __name__ == "__main__":
    main()
