#!/usr/bin/env python3
"""Compute Mode-A/B episcore long rows from production wide TSV or beta file.

Writes ``{out}/{unit_id}.episcore.tsv`` with columns:
  sample, chr, threshold, recall, hypo_z_intra, hyper_z_intra,
  hypo_cpgs_count, hyper_cpgs_count

Sources (first hit):
  1. ``ep_wide_path`` in units CSV (production beta_to_episcore / beta_to_zscore)
  2. ``beta_path`` + ``--cpg-list`` via ``beta_to_episcore.py`` (tmp wide → melt)

Note: production / beta files are only valid for the extract threshold they
were built at (early panel = 0.5). Mode B (thr=0.1) needs a new beta extract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()

CHR_LIST = [f"chr{i}" for i in range(1, 23)]
SIF = "/lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif"
BIN_EP = "/lustre1/cqyi/AIPT_2.0/workflow/episcore/bin/beta_to_episcore.py"
DEFAULT_CPG_065 = "/lustre1/cqyi/AIPT_2.0/workflow/episcore/assets/CpG_recall0.65.txt"
DEFAULT_CPG_061 = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260525-grid_search_240k_panel_240k_model/recall_list_220k/220k_cpg_recall_0.61.txt"
)


def _melt_wide(
    sample: str,
    row: pd.Series,
    threshold: float,
    recall: float,
) -> pd.DataFrame:
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


def _from_beta(
    unit_id: str,
    beta_path: Path,
    cpg_list: Path,
    depth: int,
    threshold: float,
    recall: float,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="batch_qc_ep_") as tmp:
        prefix = str(Path(tmp) / unit_id)
        py_args = [
            "python3",
            BIN_EP,
            "--beta-value",
            str(beta_path),
            "--output-prefix",
            prefix,
            "--depth",
            str(depth),
            "--cpg-list",
            str(cpg_list),
            "--chr-list",
            "1-22",
        ]
        # Avoid nested singularity when already inside the container (SLURM jobs).
        in_sif = bool(
            os.environ.get("SINGULARITY_NAME")
            or os.environ.get("APPTAINER_NAME")
            or Path("/.singularity.d").exists()
        )
        if in_sif or shutil.which("singularity") is None:
            cmd = py_args
        else:
            cmd = [
                "singularity",
                "exec",
                "-B",
                "/lustre1,/lustre2,/appsnew",
                SIF,
                *py_args,
            ]
        subprocess.run(cmd, check=True)
        wide_path = Path(f"{prefix}_zscore.tsv")
        if not wide_path.is_file():
            # beta_to_episcore may write without _zscore suffix depending on prefix
            candidates = list(Path(tmp).glob(f"{unit_id}*"))
            raise FileNotFoundError(f"No wide episcore for {unit_id}: {candidates}")
        wide = pd.read_csv(wide_path, sep="\t")
        return _melt_wide(unit_id, wide.iloc[0], threshold, recall)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--threshold", required=True, type=float)
@click.option("--recall", required=True, type=float)
@click.option("--cpg-list", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--depth", default=30, show_default=True, type=int)
@click.option("--prefer-wide", is_flag=True, default=True, help="Use production wide when present.")
@click.option("--unit-id", default=None)
@click.option("--index", default=None, type=int)
@click.option("--force", is_flag=True, default=False)
def main(
    units: str,
    output_dir: str,
    threshold: float,
    recall: float,
    cpg_list: str | None,
    depth: int,
    prefer_wide: bool,
    unit_id: str | None,
    index: int | None,
    force: bool,
) -> None:
    udf = pd.read_csv(units)
    if unit_id is not None:
        udf = udf[udf["unit_id"].astype(str) == unit_id]
    elif index is not None:
        udf = udf.iloc[[index]]
    if udf.empty:
        raise click.ClickException("No units")

    if cpg_list is None:
        cpg_list = DEFAULT_CPG_065 if abs(recall - 0.65) < 1e-9 else DEFAULT_CPG_061
    cpg_path = Path(cpg_list)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    console.print(f"units={len(udf)} thr={threshold} recall={recall}")

    ok = skip = miss = 0
    for _, r in udf.iterrows():
        uid = str(r["unit_id"])
        out_path = out_root / f"{uid}.episcore.tsv"
        if out_path.is_file() and not force:
            skip += 1
            continue

        long_df = None
        wide = str(r.get("ep_wide_path", "") or "")
        beta = str(r.get("beta_path", "") or "")

        # Only reuse production wide when combo matches Mode A (0.5 / 0.65)
        use_wide = (
            prefer_wide
            and wide
            and Path(wide).is_file()
            and abs(threshold - 0.5) < 1e-9
            and abs(recall - 0.65) < 1e-9
        )
        if use_wide:
            w = pd.read_csv(wide, sep="\t")
            long_df = _melt_wide(uid, w.iloc[0], threshold, recall)
        elif beta and Path(beta).is_file() and abs(threshold - 0.5) < 1e-9:
            # Production beta is extracted at thr=0.5
            long_df = _from_beta(uid, Path(beta), cpg_path, depth, threshold, recall)
        else:
            miss += 1
            console.print(
                f"[yellow]Need beta@thr={threshold}[/yellow] {uid} "
                f"(wide={bool(wide)} beta={bool(beta)})"
            )
            continue

        long_df.to_csv(out_path, sep="\t", index=False)
        ok += 1
        console.print(f"  [green]OK[/green] {uid}")

    console.print(f"done ok={ok} skip={skip} miss={miss}")


if __name__ == "__main__":
    main()
