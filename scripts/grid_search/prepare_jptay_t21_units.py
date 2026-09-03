#!/usr/bin/env python3
"""Build JPTAY T21 units, combo lists, and resolved reference sample names.

Reads the meta / mqres samplesheets plus the previous grid-search parquets
(``episcore_grid_search.parquet``, ``zscore_grid_search.parquet``). Does not
recompute reference scores: it only records which (threshold, recall) combos
already exist so later steps can reuse those mu/sigma values.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console

from t21_combo_common import (
    DEFAULT_CPG_DIR,
    DEFAULT_EARLY_REF_MATRIX,
    DEFAULT_EZ_REF,
    DEFAULT_GRID_INPUT,
    DEFAULT_META,
    DEFAULT_MQRES,
    DEFAULT_SAMPLES,
    EXTRA_MQRES_DEFAULT,
    PRODUCTION_EP,
    PRODUCTION_Z,
    batch_key,
    early_ref_from_matrix,
    find_production_beta,
    load_sample_list,
    parse_sample_batch,
    pick_deconv_row,
    resolve_names,
    write_combo_csv,
)

console = Console()


def _load_mqres(paths: list[Path], keep_samples: set[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.is_file():
            console.print(f"[yellow]skip missing mqres[/yellow] {path}")
            continue
        df = pd.read_csv(path)
        if "deconv_res" not in df.columns or "clean_bam" not in df.columns:
            raise click.ClickException(f"{path} missing deconv_res/clean_bam")
        rows = []
        for _, r in df.iterrows():
            try:
                sample, batch = parse_sample_batch(
                    r["sample"], r.get("mqres_batch", r.get("qc_batch"))
                )
            except ValueError:
                continue
            if sample not in keep_samples:
                continue
            rows.append(
                {
                    "sample": sample,
                    "batch": batch,
                    "batch_key": batch_key(batch),
                    "clean_bam": r["clean_bam"],
                    "deconv_res": r["deconv_res"],
                    "source_mqres": str(path),
                }
            )
        if rows:
            frames.append(pd.DataFrame(rows))
            console.print(f"  {path.name}: {len(rows)} matching rows")
    if not frames:
        raise click.ClickException("No mqres rows matched the sample list")
    return pd.concat(frames, ignore_index=True)


def _parquet_combos(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["threshold", "recall"])
    combos = (
        df[["threshold", "recall"]]
        .drop_duplicates()
        .astype({"threshold": float, "recall": float})
        .sort_values(["threshold", "recall"])
        .reset_index(drop=True)
    )
    return combos


def _parquet_samples(path: Path) -> list[str]:
    df = pd.read_parquet(path, columns=["sample"])
    return df["sample"].astype(str).drop_duplicates().tolist()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--mqres", default=str(DEFAULT_MQRES), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option(
    "--samples",
    default=",".join(DEFAULT_SAMPLES),
    show_default=True,
    help="Comma-separated sample IDs.",
)
@click.option("--grid-input", default=str(DEFAULT_GRID_INPUT), type=click.Path(exists=True, file_okay=False))
@click.option("--early-ref-matrix", default=str(DEFAULT_EARLY_REF_MATRIX), type=click.Path(exists=True, dir_okay=False))
@click.option("--ez-ref-file", default=str(DEFAULT_EZ_REF), type=click.Path(exists=True, dir_okay=False))
@click.option("--cpg-dir", default=str(DEFAULT_CPG_DIR), type=click.Path(exists=True, file_okay=False))
@click.option(
    "--extra-mqres",
    default=",".join(str(p) for p in EXTRA_MQRES_DEFAULT),
    help="Optional extra mqres CSVs (re-run batches). Empty to disable.",
)
def main(
    meta: str,
    mqres: str,
    output_dir: str,
    samples: str,
    grid_input: str,
    early_ref_matrix: str,
    ez_ref_file: str,
    cpg_dir: str,
    extra_mqres: str,
) -> None:
    keep = {s.strip() for s in samples.split(",") if s.strip()}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    console.rule("[bold blue]Prepare JPTAY T21 units")

    mqres_paths = [Path(mqres)]
    if extra_mqres.strip():
        mqres_paths.extend(Path(p.strip()) for p in extra_mqres.split(",") if p.strip())
    raw = _load_mqres(mqres_paths, keep)

    picked = []
    for (_, _), g in raw.groupby(["sample", "batch"], sort=False):
        picked.append(pick_deconv_row(g))
    units = pd.DataFrame(picked).reset_index(drop=True)
    units["unit_id"] = units["sample"] + "__" + units["batch_key"]

    meta_df = pd.read_csv(meta)
    meta_df["sample"] = meta_df["sample"].astype(str)
    meta_by = meta_df.drop_duplicates("sample", keep="first").set_index("sample")

    rows = []
    for _, r in units.iterrows():
        sample = str(r["sample"])
        m = meta_by.loc[sample] if sample in meta_by.index else None
        beta = find_production_beta(sample, str(r["clean_bam"]))
        rows.append(
            {
                "unit_id": r["unit_id"],
                "sample": sample,
                "batch": r["batch"],
                "batch_key": r["batch_key"],
                "clean_bam": r["clean_bam"],
                "deconv_res": r["deconv_res"],
                "label": "T21",
                "meta_label": (str(m["label"]) if m is not None else ""),
                "pred_label": (str(m["pred_label"]) if m is not None else ""),
                "ff_before_mq": (
                    float(m["ff_before_mq"])
                    if m is not None and pd.notna(m.get("ff_before_mq"))
                    else float("nan")
                ),
                "available_batches": (
                    str(m["available_batches"]) if m is not None else ""
                ),
                "beta_path": str(beta) if beta else "",
                "has_beta": beta is not None,
                "source_mqres": r["source_mqres"],
            }
        )
    unit_df = pd.DataFrame(rows).sort_values(["sample", "batch"]).reset_index(drop=True)
    unit_df.to_csv(out / "unit_samplesheet.csv", index=False)
    nf = unit_df[["unit_id", "clean_bam", "deconv_res"]].rename(columns={"unit_id": "sample"})
    nf.to_csv(out / "nf_split_bam_samplesheet.csv", index=False)
    (out / "n_units.txt").write_text(f"{len(unit_df)}\n")

    grid = Path(grid_input)
    ep_pq = grid / "episcore_grid_search.parquet"
    z_pq = grid / "zscore_grid_search.parquet"
    if not ep_pq.is_file() or not z_pq.is_file():
        raise click.ClickException(f"Missing grid-search parquets under {grid}")

    console.print("[cyan]Reading combo lists from previous grid-search parquets[/cyan]")
    ep_combos = _parquet_combos(ep_pq)
    z_combos = _parquet_combos(z_pq)
    write_combo_csv(out / "epi_combos.csv", list(ep_combos.itertuples(index=False, name=None)))
    write_combo_csv(out / "z_combos.csv", list(z_combos.itertuples(index=False, name=None)))

    ep_thr = sorted(ep_combos["threshold"].unique().tolist())
    z_thr = sorted(z_combos["threshold"].unique().tolist())
    (out / "epi_thresholds.txt").write_text("\n".join(f"{t:g}" for t in ep_thr) + "\n")
    (out / "z_thresholds.txt").write_text("\n".join(f"{t:g}" for t in z_thr) + "\n")

    wanted_early = early_ref_from_matrix(Path(early_ref_matrix))
    wanted_ez = load_sample_list(Path(ez_ref_file))
    ep_names = _parquet_samples(ep_pq)
    z_names = _parquet_samples(z_pq)
    shared = set(ep_names) & set(z_names)

    early_ep, miss_early_ep = resolve_names(wanted_early, shared)
    early_z, miss_early_z = resolve_names(wanted_early, shared)
    ez_ep, miss_ez_ep = resolve_names(wanted_ez, shared)
    ez_z, miss_ez_z = resolve_names(wanted_ez, shared)

    missing = {
        "early_ref_episcore": miss_early_ep,
        "early_ref_zscore": miss_early_z,
        "ez_ref_episcore": miss_ez_ep,
        "ez_ref_zscore": miss_ez_z,
    }
    bad = {k: v for k, v in missing.items() if v}
    if bad:
        raise click.ClickException(
            "Reference samples missing from grid-search parquets: "
            + ", ".join(f"{k}={v}" for k, v in bad.items())
        )
    if early_ep != early_z or ez_ep != ez_z:
        raise click.ClickException("Resolved early_ref / ez_ref names differ between episcore and zscore")

    Path(out / "early_ref_samples.txt").write_text("\n".join(early_ep) + "\n")
    Path(out / "ez_ref_samples.txt").write_text("\n".join(ez_ep) + "\n")
    Path(out / "early_ref_requested.txt").write_text("\n".join(wanted_early) + "\n")
    Path(out / "ez_ref_requested.txt").write_text("\n".join(wanted_ez) + "\n")

    prod_ep = (float(PRODUCTION_EP[0]), float(PRODUCTION_EP[1]))
    prod_z = (float(PRODUCTION_Z[0]), float(PRODUCTION_Z[1]))
    ep_set = {(float(t), float(r)) for t, r in ep_combos.itertuples(index=False, name=None)}
    z_set = {(float(t), float(r)) for t, r in z_combos.itertuples(index=False, name=None)}
    if prod_ep not in ep_set:
        raise click.ClickException(f"Production epi combo {prod_ep} not in parquet")
    if prod_z not in z_set:
        raise click.ClickException(f"Production z combo {prod_z} not in parquet")

    summary = (
        f"units={len(unit_df)}\n"
        f"samples={','.join(sorted(unit_df['sample'].unique()))}\n"
        f"batches={','.join(unit_df['unit_id'])}\n"
        f"epi_combos={len(ep_combos)} thresholds={ep_thr}\n"
        f"z_combos={len(z_combos)} thresholds={z_thr}\n"
        f"early_ref={len(early_ep)}\n"
        f"ez_ref={len(ez_ep)}\n"
        f"production_ep={prod_ep}\n"
        f"production_z={prod_z}\n"
        f"grid_input={grid}\n"
        f"cpg_dir={cpg_dir}\n"
        f"has_production_beta={int(unit_df['has_beta'].sum())}/{len(unit_df)}\n"
    )
    (out / "prepare_summary.txt").write_text(summary)
    console.print(summary)
    console.print(f"[green]OK[/green] Wrote {out}")
    console.print(
        "[yellow]Note[/yellow] Production beta files (thr=0.5) are often filtered "
        "to recall 0.65. Episcore grid uses Nextflow EXTRACT_BETA with the "
        "grid_search full CpG panel so every recall is valid."
    )


if __name__ == "__main__":
    main()
