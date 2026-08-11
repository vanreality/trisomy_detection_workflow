#!/usr/bin/env python3
"""Merge the 4 missing ezscore-ref Normals into a fixed-combo input dir.

Sources (already run elsewhere):
  * meta: ``20260607-ref_40/meta.csv`` (dev Normal)
  * episcore wide: early ``beta_to_zscore/{sample}_zscore.tsv``
    (matches main parquet @ ep 0.5/0.65)
  * zscore percentage: ``20260730-stable_ref40/percentage.csv`` cutoff=0.85
    (percent → fraction; tagged as threshold=0.85, recall=0.95)

Writes a working input dir for ``ref_free_fixed_ez_flags`` with all 25
ezscore_ref_samples.txt IDs resolvable (HCPT truncated to 8 chars).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()

CHR_LIST = [f"chr{i}" for i in range(1, 23)]

DEFAULT_MAIN = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng"
)
DEFAULT_META_SRC = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260607-ref_40"
)
DEFAULT_PCT = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260730-stable_ref40/percentage.csv"
)
DEFAULT_EP_ROOT = (
    "/lustre1/cqyi/syfan/snp_nipt/results/beta_trisomy_detection/"
    "20260123_early/beta_to_zscore"
)

MISSING_DEFAULT = (
    "PTAY0103P,PTAY0630P7S1,PTAY0652P7H1,PTAY1223P7S1"
)


def _melt_wide_episcore_row(
    sample: str, row: pd.Series, threshold: float, recall: float
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--main-input", default=DEFAULT_MAIN, type=click.Path(exists=True, file_okay=False))
@click.option("--meta-src", default=DEFAULT_META_SRC, type=click.Path(exists=True, file_okay=False))
@click.option("--percentage-tsv", default=DEFAULT_PCT, type=click.Path(exists=True, dir_okay=False))
@click.option("--ep-wide-root", default=DEFAULT_EP_ROOT, type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--missing-samples", default=MISSING_DEFAULT, show_default=True)
@click.option("--ep-threshold", default=0.5, show_default=True, type=float)
@click.option("--ep-recall", default=0.65, show_default=True, type=float)
@click.option("--z-threshold", default=0.85, show_default=True, type=float)
@click.option("--z-recall", default=0.95, show_default=True, type=float)
@click.option("--z-pct-cutoff", default=0.85, show_default=True, type=float,
              help="cutoff column value in percentage.csv")
def main(
    main_input: str,
    meta_src: str,
    percentage_tsv: str,
    ep_wide_root: str,
    output_dir: str,
    missing_samples: str,
    ep_threshold: float,
    ep_recall: float,
    z_threshold: float,
    z_recall: float,
    z_pct_cutoff: float,
) -> None:
    main_path = Path(main_input)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing = [s.strip() for s in missing_samples.split(",") if s.strip()]
    ep_root = Path(ep_wide_root)

    main_meta = pd.read_csv(main_path / "meta.csv").drop_duplicates("sample", keep="first")
    main_meta["sample"] = main_meta["sample"].astype(str)
    already = set(main_meta["sample"]) & set(missing)
    if already:
        console.print(f"[yellow]Already in main (skip): {sorted(already)}[/yellow]")
    need = [s for s in missing if s not in already]
    if not need:
        raise click.ClickException("Nothing to add — all missing samples already in main")

    src_meta = pd.read_csv(Path(meta_src) / "meta.csv").drop_duplicates("sample", keep="first")
    src_meta["sample"] = src_meta["sample"].astype(str)
    add_meta = src_meta[src_meta["sample"].isin(need)].copy()
    if len(add_meta) != len(need):
        found = set(add_meta["sample"])
        raise click.ClickException(f"Meta missing for {sorted(set(need) - found)}")
    add_meta["set"] = "dev"
    # keep label Normal as in source
    console.print(f"Adding meta for {len(add_meta)} samples from {meta_src}")

    # --- episcore from early wide zscore TSVs ---
    ep_frames = []
    for sample in need:
        wide_path = ep_root / f"{sample}_zscore.tsv"
        if not wide_path.is_file():
            raise click.ClickException(f"Missing episcore wide TSV: {wide_path}")
        wide = pd.read_csv(wide_path, sep="\t")
        ep_frames.append(
            _melt_wide_episcore_row(sample, wide.iloc[0], ep_threshold, ep_recall)
        )
    add_ep = pd.concat(ep_frames, ignore_index=True)
    console.print(f"Episcore rows: {len(add_ep)} ({len(need)} × 22 chr)")

    # --- zscore percentage (percent → fraction) ---
    pct = pd.read_csv(percentage_tsv, sep="\t")
    pct["sample"] = pct["sample"].astype(str)
    sub = pct[(pct["sample"].isin(need)) & (pct["cutoff"].astype(float) == z_pct_cutoff)].copy()
    if sub["sample"].nunique() != len(need):
        raise click.ClickException(
            f"percentage.csv incomplete for cutoff={z_pct_cutoff}: "
            f"{sorted(set(need) - set(sub['sample']))}"
        )
    add_z = pd.DataFrame(
        {
            "sample": sub["sample"],
            "chr": sub["chr"].astype(str),
            "threshold": z_threshold,
            "recall": z_recall,
            "percentage": pd.to_numeric(sub["percentage"], errors="coerce") / 100.0,
        }
    )
    # ensure chr ordering completeness
    for sample in need:
        n = (add_z["sample"] == sample).sum()
        if n != 22:
            raise click.ClickException(f"{sample}: expected 22 chr rows, got {n}")
    console.print(f"Zscore rows: {len(add_z)} (percentage/100 @ cutoff={z_pct_cutoff})")

    main_ep = pd.read_parquet(main_path / "episcore_grid_search.parquet")
    main_z = pd.read_parquet(main_path / "zscore_grid_search.parquet")

    merged_meta = pd.concat([main_meta, add_meta], ignore_index=True, sort=False)
    merged_ep = pd.concat([main_ep, add_ep], ignore_index=True)
    merged_z = pd.concat([main_z, add_z], ignore_index=True)

    merged_meta.to_csv(out / "meta.csv", index=False)
    merged_ep.to_parquet(out / "episcore_grid_search.parquet", index=False, compression="snappy")
    merged_z.to_parquet(out / "zscore_grid_search.parquet", index=False, compression="snappy")

    # copy ezscore ref list (+ optional pool list)
    shutil.copy2(main_path / "ezscore_ref_samples.txt", out / "ezscore_ref_samples.txt")
    if (main_path / "ref_pool_samples.txt").is_file():
        shutil.copy2(main_path / "ref_pool_samples.txt", out / "ref_pool_samples.txt")

    add_meta[["sample", "set", "label", "ff_before_mq"]].to_csv(
        out / "added_ez_ref_samples.tsv", sep="\t", index=False
    )
    summary = (
        f"added={need}\n"
        f"ep_wide_root={ep_root}\n"
        f"percentage={percentage_tsv} cutoff={z_pct_cutoff} -> thr={z_threshold}/rec={z_recall}\n"
        f"meta_n={len(merged_meta)} ep_rows={len(merged_ep)} z_rows={len(merged_z)}\n"
    )
    (out / "prepare_fixed_ez25_summary.txt").write_text(summary)
    console.print(f"[green]OK[/green] Wrote {out}")
    console.print(summary)


if __name__ == "__main__":
    main()
