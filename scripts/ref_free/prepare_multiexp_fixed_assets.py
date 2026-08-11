#!/usr/bin/env python3
"""Build fixed-combo input for quick ref_free sample checks.

Reuses production beta_to_episcore wide TSVs + zscore CSVs (skips FF / nextflow),
merges check samples with the 96-sample dev Normal ref pool from the main
20260621 assets.

Sample list is inferred from ``--mqres``:
  - ``sample`` column = id used in ref_free
  - batch from ``.../output/<BATCH>/...`` in deconv_res / clean_bam
  - production lookup uses ``orig_sample`` (strip trailing ``_<batch-date>`` if present)
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()

CHR_LIST = [f"chr{i}" for i in range(1, 23)]

DEFAULT_MAIN = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng"
)
DEFAULT_MQRES = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260803-ref_free_multiexp_sample_check/mqres.csv"
)

PROD_ROOT = Path(
    "/lustre1/cqyi/myli/bert/DNA_5mC_analysis_pipeline/output"
)
_BATCH_RE = re.compile(r"/(?P<batch>20\d{6}-XML)/")


def _prod_base(batch: str, orig: str) -> Path:
    return (
        PROD_ROOT
        / batch
        / "bwameth_results"
        / "zscore_downstream"
        / "beta_zscore"
        / orig
    )


def _infer_batch(path: str) -> str:
    m = _BATCH_RE.search(str(path).replace("\\", "/"))
    if not m:
        raise click.ClickException(f"Cannot infer batch (20YYMMDD-XML) from path: {path}")
    return m.group("batch")


def _orig_sample(sample: str, batch: str) -> str:
    """Strip trailing _YYYYMMDD when it matches the batch date."""
    date = batch.split("-", 1)[0]
    suffix = f"_{date}"
    if sample.endswith(suffix):
        return sample[: -len(suffix)]
    return sample


def _samples_from_mqres(mq: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return [(sample_id, batch, orig_sample), ...] unique by sample_id."""
    rows = []
    seen = set()
    for _, r in mq.iterrows():
        sample = str(r["sample"])
        if sample in seen:
            continue
        seen.add(sample)
        path = r["deconv_res"] if "deconv_res" in r and pd.notna(r["deconv_res"]) else r["clean_bam"]
        batch = _infer_batch(str(path))
        orig = _orig_sample(sample, batch)
        rows.append((sample, batch, orig))
    return rows


def _melt_wide_episcore_row(
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


def _wide_episcore_path(batch: str, orig: str) -> Path:
    """Prefer beta_to_episcore; older runs may only have beta_to_zscore."""
    base = _prod_base(batch, orig)
    for sub in ("beta_to_episcore", "beta_to_zscore"):
        path = base / sub / f"{orig}_zscore.tsv"
        if path.is_file():
            return path
    return base / "beta_to_episcore" / f"{orig}_zscore.tsv"


def _load_ff(batch: str, orig: str) -> float:
    base = _prod_base(batch, orig)
    for rel in (
        f"estimate_ff_higher_precision/{orig}_ff.tsv",
        f"snp_to_ff/{orig}_ff.tsv",
        f"generate_report/{orig}_report.tsv",
    ):
        ff_path = base / rel
        if ff_path.is_file():
            df = pd.read_csv(ff_path, sep="\t")
            if "ff_before_mq" in df.columns:
                return float(pd.to_numeric(df["ff_before_mq"], errors="coerce").iloc[0])
    console.print(f"[yellow]No FF for {orig}@{batch}; using NaN[/yellow]")
    return float("nan")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--main-input", default=DEFAULT_MAIN, type=click.Path(exists=True, file_okay=False))
@click.option("--mqres", default=DEFAULT_MQRES, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--ep-threshold", default=0.5, show_default=True, type=float)
@click.option("--ep-recall", default=0.65, show_default=True, type=float)
@click.option("--z-threshold", default=0.85, show_default=True, type=float)
@click.option("--z-recall", default=0.95, show_default=True, type=float)
def main(
    main_input: str,
    mqres: str,
    output_dir: str,
    ep_threshold: float,
    ep_recall: float,
    z_threshold: float,
    z_recall: float,
) -> None:
    main_path = Path(main_input)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mq = pd.read_csv(mqres)
    if "sample" not in mq.columns:
        raise click.ClickException("mqres missing sample column")
    check_samples = _samples_from_mqres(mq)
    if not check_samples:
        raise click.ClickException(f"No samples in mqres: {mqres}")
    console.print(f"Check samples from mqres: {len(check_samples)}")

    # --- slim main: keep only 96-sample dev Normal ref pool ---
    main_meta = pd.read_csv(main_path / "meta.csv").drop_duplicates("sample", keep="first")
    main_meta["sample"] = main_meta["sample"].astype(str)
    ref_meta = main_meta[
        (main_meta["set"].astype(str) == "dev")
        & (main_meta["label"].astype(str) == "Normal")
    ].copy()
    if len(ref_meta) < 96:
        raise click.ClickException(f"Expected >=96 dev Normal, found {len(ref_meta)}")
    ref_samples = set(ref_meta["sample"])

    main_ep = pd.read_parquet(main_path / "episcore_grid_search.parquet")
    main_z = pd.read_parquet(main_path / "zscore_grid_search.parquet")
    main_ep = main_ep[
        main_ep["sample"].astype(str).isin(ref_samples)
        & (main_ep["threshold"] == ep_threshold)
        & (main_ep["recall"] == ep_recall)
    ]
    main_z = main_z[
        main_z["sample"].astype(str).isin(ref_samples)
        & (main_z["threshold"] == z_threshold)
        & (main_z["recall"] == z_recall)
    ]
    console.print(
        f"Ref pool: n={len(ref_meta)} ep_rows={len(main_ep)} z_rows={len(main_z)}"
    )

    # --- check samples from production artifacts ---
    ep_rows = []
    z_rows = []
    meta_rows = []
    for new_id, batch, orig in check_samples:
        wide_path = _wide_episcore_path(batch, orig)
        z_glob = list(
            (
                PROD_ROOT
                / batch
                / "bwameth_results"
                / "zscore_downstream"
                / "zscore_data.CpG_recall0.95"
                / orig
            ).glob(f"{orig}.{z_threshold:g}.*.zscore.csv")
        )
        if not wide_path.is_file():
            raise click.ClickException(f"Missing episcore wide: {wide_path}")
        if not z_glob:
            raise click.ClickException(f"Missing zscore CSV for {new_id} ({orig}@{batch})")

        wide = pd.read_csv(wide_path, sep="\t")
        ep_rows.append(
            _melt_wide_episcore_row(new_id, wide.iloc[0], ep_threshold, ep_recall)
        )
        console.print(f"    wide source: {wide_path.parent.name}/{wide_path.name}")

        zdf = pd.read_csv(z_glob[0])
        sub = zdf[["chr", "percentage"]].copy()
        sub["sample"] = new_id
        sub["threshold"] = z_threshold
        sub["recall"] = z_recall
        z_rows.append(sub[["sample", "chr", "threshold", "recall", "percentage"]])

        ff = _load_ff(batch, orig)
        meta_rows.append(
            {
                "sample": new_id,
                "set": "test",
                "label": "Unknown",
                "ff_before_mq": ff,
                "batch": batch,
                "orig_sample": orig,
            }
        )
        console.print(
            f"  {new_id}: batch={batch} orig={orig} "
            f"ff={ff if pd.isna(ff) else f'{ff:.4f}'} z={z_glob[0].name}"
        )

    check_meta = pd.DataFrame(meta_rows)
    check_ep = pd.concat(ep_rows, ignore_index=True)
    check_z = pd.concat(z_rows, ignore_index=True)

    for col in ("set", "label", "ff_before_mq"):
        if col not in ref_meta.columns:
            raise click.ClickException(f"main meta missing {col}")
    extra_cols = [c for c in check_meta.columns if c not in ref_meta.columns]
    for c in extra_cols:
        ref_meta[c] = pd.NA

    merged_meta = pd.concat([ref_meta, check_meta], ignore_index=True, sort=False)
    merged_ep = pd.concat([main_ep, check_ep], ignore_index=True)
    merged_z = pd.concat([main_z, check_z], ignore_index=True)

    merged_meta.to_csv(out / "meta.csv", index=False)
    merged_ep.to_parquet(out / "episcore_grid_search.parquet", index=False, compression="snappy")
    merged_z.to_parquet(out / "zscore_grid_search.parquet", index=False, compression="snappy")
    check_meta.to_csv(out / "check_samples.tsv", sep="\t", index=False)
    (out / "prepare_summary.txt").write_text(
        f"ref_n={len(ref_meta)}\n"
        f"check_n={len(check_meta)}\n"
        f"ep_threshold={ep_threshold}\nep_recall={ep_recall}\n"
        f"z_threshold={z_threshold}\nz_recall={z_recall}\n"
        f"mqres={mqres}\n"
    )
    console.print(f"[green]OK[/green] Wrote {out}")
    console.print(
        f"  meta={len(merged_meta)} ep={len(merged_ep)} z={len(merged_z)} "
        f"check={list(check_meta['sample'])}"
    )


if __name__ == "__main__":
    main()
