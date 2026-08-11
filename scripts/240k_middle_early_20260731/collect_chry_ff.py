#!/usr/bin/env python3
"""Collect ff_before_mq + chrY/X ratios for early + middle samples."""

from __future__ import annotations

import click
import pandas as pd
import pysam
from rich.console import Console

import config as cfg

console = Console()


def _chr_ratios(bam_path: str) -> tuple[float | None, float | None]:
    try:
        idxstats = pysam.idxstats(bam_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]idxstats failed[/yellow] {bam_path}: {exc}")
        return None, None
    chrY_count = chrX_count = total = 0
    for line in idxstats.strip().split("\n"):
        if not line.strip():
            continue
        chrom, _length, count, _unmapped = line.split("\t")[:4]
        count = int(count)
        total += count
        if chrom in ("chrY", "Y"):
            chrY_count = count
        if chrom in ("chrX", "X"):
            chrX_count = count
    if total <= 0:
        return None, None
    return chrY_count / total, chrX_count / total


def _load_ff() -> pd.DataFrame:
    frames = []
    for path in (
        cfg.OLD_FF_20260416,
        cfg.OLD_FF_20260507,
        cfg.OLD_FF_20260720,
        cfg.EARLY_NF_OUT / "collect_reports" / "summary_report.tsv",
        cfg.MIDDLE_NF_OUT / "collect_reports" / "summary_report.tsv",
        cfg.PRIOR_CHRY_FF,  # fallback FF from prior table
    ):
        if not path.is_file():
            continue
        if path == cfg.PRIOR_CHRY_FF:
            df = pd.read_csv(path, sep="\t", usecols=["sample", "ff_before_mq"])
        else:
            df = pd.read_csv(path, sep="\t", usecols=["sample", "ff_before_mq"])
        frames.append(df)
    if not frames:
        raise click.ClickException("No summary_report / FF tables found")
    return pd.concat(frames, ignore_index=True).drop_duplicates("sample", keep="last")


def _bam_table() -> pd.DataFrame:
    frames = [
        pd.read_csv(cfg.OLD_SAMPLESHEET_20260416),
        pd.read_csv(cfg.OLD_SAMPLESHEET_20260507),
        pd.read_csv(cfg.OLD_SAMPLESHEET_20260720),
        pd.read_csv(cfg.EARLY_MQRES),
        pd.read_csv(cfg.MIDDLE_MQRES),
    ]
    ss = pd.concat(frames, ignore_index=True)
    return ss.groupby("sample", as_index=False).agg(clean_bam=("clean_bam", "first"))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    labels = pd.read_csv(cfg.COHORT_LABELS)
    ff = _load_ff()
    bams = _bam_table()
    df = labels.merge(ff, on="sample", how="left").merge(bams, on="sample", how="left")

    ratios = []
    for bam in df["clean_bam"]:
        if pd.isna(bam):
            ratios.append((None, None))
            continue
        ratios.append(_chr_ratios(str(bam)))
    df["chrY_ratio"] = [r[0] for r in ratios]
    df["chrX_ratio"] = [r[1] for r in ratios]

    cfg.CHRY_FF_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.CHRY_FF_TSV, sep="\t", index=False)
    console.print(f"[green]Wrote[/green] {cfg.CHRY_FF_TSV}")
    console.print(
        f"  samples={len(df)}  with_ff={df['ff_before_mq'].notna().sum()}  "
        f"with_chrY={df['chrY_ratio'].notna().sum()}"
    )
    miss = df.loc[df["ff_before_mq"].isna(), ["sample", "dataset", "cohort"]]
    if len(miss):
        console.print(
            f"[yellow]Missing FF ({len(miss)}):[/yellow] "
            f"{', '.join(miss['sample'].astype(str).head(20).tolist())}"
            + ("..." if len(miss) > 20 else "")
        )


if __name__ == "__main__":
    main()
