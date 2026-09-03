#!/usr/bin/env python3
"""Assemble slim ref_free input for one fixed combo (batch QC).

Merges:
  * 96-sample dev Normal ref pool from ``--main-input`` (filtered to the combo)
  * Check units from per-unit ``*.episcore.tsv`` + ``*.percentage.tsv``

Writes ``meta.csv``, ``episcore_grid_search.parquet``, ``zscore_grid_search.parquet``,
``check_samples.tsv``, ``prepare_summary.txt`` under ``--output-dir``.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()

DEFAULT_MAIN = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng"
)
CHR_LIST = [f"chr{i}" for i in range(1, 23)]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--main-input", default=DEFAULT_MAIN, type=click.Path(exists=True, file_okay=False))
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--ep-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--z-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--ep-threshold", required=True, type=float)
@click.option("--ep-recall", required=True, type=float)
@click.option("--z-threshold", required=True, type=float)
@click.option("--z-recall", required=True, type=float)
@click.option(
    "--require-complete/--allow-partial",
    default=True,
    help="Fail if any unit is missing ep or z (default: require complete).",
)
def main(
    main_input: str,
    units: str,
    ep_dir: str,
    z_dir: str,
    output_dir: str,
    ep_threshold: float,
    ep_recall: float,
    z_threshold: float,
    z_recall: float,
    require_complete: bool,
) -> None:
    main_path = Path(main_input)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ep_root = Path(ep_dir)
    z_root = Path(z_dir)

    udf = pd.read_csv(units)
    udf["unit_id"] = udf["unit_id"].astype(str)

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
        f"Ref pool n={len(ref_meta)} ep_rows={len(main_ep)} z_rows={len(main_z)}"
    )

    ep_rows = []
    z_rows = []
    meta_rows = []
    missing = []
    for _, r in udf.iterrows():
        uid = str(r["unit_id"])
        ep_path = ep_root / f"{uid}.episcore.tsv"
        z_path = z_root / f"{uid}.percentage.tsv"
        if not ep_path.is_file() or not z_path.is_file():
            missing.append(uid)
            continue

        ep = pd.read_csv(ep_path, sep="\t")
        ep["sample"] = uid
        ep["threshold"] = ep_threshold
        ep["recall"] = ep_recall
        need_ep = [
            "sample",
            "chr",
            "threshold",
            "recall",
            "hypo_z_intra",
            "hyper_z_intra",
            "hypo_cpgs_count",
            "hyper_cpgs_count",
        ]
        ep_rows.append(ep[need_ep])

        zdf = pd.read_csv(z_path, sep="\t")
        if "percentage" not in zdf.columns:
            raise click.ClickException(f"{z_path} missing percentage")
        zdf = zdf[zdf["chr"].isin(CHR_LIST)].copy()
        zdf["sample"] = uid
        zdf["threshold"] = z_threshold
        zdf["recall"] = z_recall
        z_rows.append(zdf[["sample", "chr", "threshold", "recall", "percentage"]])

        ff = r.get("ff_before_mq")
        set_name = str(r.get("set", "test") or "test")
        # Keep real cohort set for viz; ref_free eval includes test/buffer/emergency
        meta_rows.append(
            {
                "sample": uid,
                "set": set_name,
                "label": str(r.get("label", "Unknown") or "Unknown"),
                "ff_before_mq": float(ff) if pd.notna(ff) else float("nan"),
                "purity": float(r["purity"])
                if "purity" in r and pd.notna(r.get("purity"))
                else float("nan"),
                "orig_sample": str(r["sample"]),
                "batch_key": str(r["batch_key"]),
                "n_batches": int(r.get("n_batches", 1) or 1),
                "preferred_batch_key": str(r.get("preferred_batch_key", "") or ""),
                "is_preferred_batch": bool(r.get("is_preferred_batch", False)),
                "pred_label": str(r.get("pred_label", "") or ""),
                "depth_qc": str(r.get("depth_qc", "") or ""),
            }
        )

    if missing:
        msg = f"Missing ep/z for {len(missing)}/{len(udf)} units (e.g. {missing[:5]})"
        if require_complete:
            raise click.ClickException(msg)
        console.print(f"[yellow]{msg}[/yellow]")

    if not meta_rows:
        raise click.ClickException("No complete check units to merge")

    check_meta = pd.DataFrame(meta_rows)
    check_ep = pd.concat(ep_rows, ignore_index=True)
    check_z = pd.concat(z_rows, ignore_index=True)

    for c in (
        "orig_sample",
        "batch_key",
        "n_batches",
        "preferred_batch_key",
        "is_preferred_batch",
        "pred_label",
        "purity",
        "depth_qc",
    ):
        if c not in ref_meta.columns:
            ref_meta[c] = pd.NA

    merged_meta = pd.concat([ref_meta, check_meta], ignore_index=True, sort=False)
    merged_ep = pd.concat([main_ep, check_ep], ignore_index=True)
    merged_z = pd.concat([main_z, check_z], ignore_index=True)

    merged_meta.to_csv(out / "meta.csv", index=False)
    merged_ep.to_parquet(out / "episcore_grid_search.parquet", index=False, compression="snappy")
    merged_z.to_parquet(out / "zscore_grid_search.parquet", index=False, compression="snappy")
    check_meta.to_csv(out / "check_samples.tsv", sep="\t", index=False)
    (out / "missing_units.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    (out / "prepare_summary.txt").write_text(
        f"ref_n={len(ref_meta)}\n"
        f"check_n={len(check_meta)}\n"
        f"missing_n={len(missing)}\n"
        f"ep_threshold={ep_threshold}\nep_recall={ep_recall}\n"
        f"z_threshold={z_threshold}\nz_recall={z_recall}\n"
        f"units={units}\n"
    )
    console.print(
        f"[green]OK[/green] {out} meta={len(merged_meta)} "
        f"check={len(check_meta)} missing={len(missing)}"
    )


if __name__ == "__main__":
    main()
