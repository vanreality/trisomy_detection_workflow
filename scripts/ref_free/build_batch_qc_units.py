#!/usr/bin/env python3
"""Expand mqres into one unit per (sample, batch_key) for batch-quality ref_free.

SE vs non-SE rows for the same batch are collapsed: prefer ``selected==True``,
then non-SE (``is_single_end==False``).

Writes under ``--output-dir``:
  - ``unit_samplesheet.csv`` — one row per unit (id = ``{sample}__{batch_key}``)
  - ``nf_split_bam_samplesheet.csv`` — Nextflow split_bam / grid_search input
  - ``score_inventory.csv`` — existing production artifact paths
  - ``build_summary.txt``
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()

DEFAULT_MQRES = (
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/mqres_samplesheet.csv"
)
DEFAULT_META = (
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv"
)


def _pick_row(g: pd.DataFrame) -> pd.Series:
    g = g.copy()
    if "selected" in g.columns and g["selected"].any():
        g = g[g["selected"] == True]  # noqa: E712
    if "is_single_end" in g.columns and (~g["is_single_end"]).any():
        g = g[~g["is_single_end"]]
    return g.iloc[0]


def _bam_root(clean_bam: str) -> Path | None:
    p = Path(str(clean_bam))
    if "bwameth_results" not in p.parts:
        return None
    i = p.parts.index("bwameth_results")
    return Path(*p.parts[: i + 1])


def _resolve_artifacts(sample: str, batch_key: str, clean_bam: str, meta_row: pd.Series | None):
    root = _bam_root(clean_bam)
    ep = z085 = beta = None
    if meta_row is not None:
        sbk = str(meta_row.get("score_batch_key", "") or "")
        ep_file = meta_row.get("episcore_file")
        if sbk == batch_key and pd.notna(ep_file) and Path(str(ep_file)).is_file():
            ep = Path(str(ep_file))
    if root is not None:
        if ep is None:
            for sub in ("beta_to_episcore", "beta_to_zscore"):
                p = (
                    root
                    / "zscore_downstream"
                    / "beta_zscore"
                    / sample
                    / sub
                    / f"{sample}_zscore.tsv"
                )
                if p.is_file():
                    ep = p
                    break
        bp = (
            root
            / "zscore_downstream"
            / "beta_zscore"
            / sample
            / "extract_beta_value"
            / f"{sample}_beta_value.tsv.gz"
        )
        if bp.is_file():
            beta = bp
        zdir = root / "zscore_downstream" / "zscore_data.CpG_recall0.95" / sample
        if zdir.is_dir():
            hits = sorted(zdir.glob(f"{sample}.0.85.*.zscore.csv"))
            if hits:
                z085 = hits[0]
    return ep, z085, beta, root


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--mqres", default=DEFAULT_MQRES, type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=DEFAULT_META, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option(
    "--multi-batch-only",
    is_flag=True,
    default=False,
    help="Keep only samples with n_batches > 1.",
)
def main(mqres: str, meta: str, output_dir: str, multi_batch_only: bool) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mq = pd.read_csv(mqres)
    meta_df = pd.read_csv(meta)
    meta_df["sample"] = meta_df["sample"].astype(str)
    meta_by = meta_df.drop_duplicates("sample", keep="first").set_index("sample")

    required = {"sample", "batch_key", "clean_bam", "deconv_res"}
    missing = required - set(mq.columns)
    if missing:
        raise click.ClickException(f"mqres missing columns: {sorted(missing)}")

    picked = []
    for (_, _), g in mq.groupby(["sample", "batch_key"], sort=False):
        picked.append(_pick_row(g))
    units = pd.DataFrame(picked).reset_index(drop=True)
    units["sample"] = units["sample"].astype(str)
    units["batch_key"] = units["batch_key"].astype(str)
    units["unit_id"] = units["sample"] + "__" + units["batch_key"]

    if multi_batch_only:
        multi = set(meta_df.loc[meta_df["n_batches"] > 1, "sample"].astype(str))
        units = units[units["sample"].isin(multi)].copy()

    rows = []
    inv_rows = []
    for _, r in units.iterrows():
        sample = str(r["sample"])
        batch_key = str(r["batch_key"])
        unit_id = str(r["unit_id"])
        m = meta_by.loc[sample] if sample in meta_by.index else None
        ep, z085, beta, root = _resolve_artifacts(
            sample, batch_key, str(r["clean_bam"]), m
        )
        n_batches = int(m["n_batches"]) if m is not None and pd.notna(m.get("n_batches")) else 1
        preferred = str(m["preferred_batch_key"]) if m is not None else ""
        label = str(m["label"]) if m is not None and pd.notna(m.get("label")) else "Unknown"
        pred = str(m["pred_label"]) if m is not None and pd.notna(m.get("pred_label")) else ""
        ff = (
            float(m["ff_before_mq"])
            if m is not None and pd.notna(m.get("ff_before_mq"))
            else float("nan")
        )
        # Prefer meta FF only when this unit is the scored batch
        if m is not None and str(m.get("score_batch_key", "") or "") != batch_key:
            ff = float("nan")

        rows.append(
            {
                "unit_id": unit_id,
                "sample": sample,
                "batch_key": batch_key,
                "clean_bam": r["clean_bam"],
                "deconv_res": r["deconv_res"],
                "is_single_end": bool(r.get("is_single_end", False)),
                "selected": bool(r.get("selected", False)),
                "n_batches": n_batches,
                "preferred_batch_key": preferred,
                "is_preferred_batch": preferred == batch_key,
                "label": label,
                "pred_label": pred,
                "ff_before_mq": ff,
                "has_ep_wide": ep is not None,
                "has_zscore_csv_0_85": z085 is not None,
                "has_beta": beta is not None,
                "ep_wide_path": str(ep) if ep else "",
                "zscore_csv_0_85_path": str(z085) if z085 else "",
                "beta_path": str(beta) if beta else "",
                "bam_root": str(root) if root else "",
            }
        )
        inv_rows.append(rows[-1])

    unit_df = pd.DataFrame(rows)
    unit_df.to_csv(out / "unit_samplesheet.csv", index=False)

    nf = unit_df[["unit_id", "clean_bam", "deconv_res"]].rename(columns={"unit_id": "sample"})
    nf.to_csv(out / "nf_split_bam_samplesheet.csv", index=False)

    inv = pd.DataFrame(inv_rows)
    inv.to_csv(out / "score_inventory.csv", index=False)

    n = len(unit_df)
    n_multi = int((unit_df["n_batches"] > 1).sum())
    summary = (
        f"units={n}\n"
        f"unique_samples={unit_df['sample'].nunique()}\n"
        f"multi_batch_units={n_multi}\n"
        f"has_ep_wide={int(unit_df['has_ep_wide'].sum())}\n"
        f"has_zscore_csv_0_85={int(unit_df['has_zscore_csv_0_85'].sum())}\n"
        f"has_beta={int(unit_df['has_beta'].sum())}\n"
        f"mqres={mqres}\n"
        f"meta={meta}\n"
        f"multi_batch_only={multi_batch_only}\n"
    )
    (out / "build_summary.txt").write_text(summary)
    console.print(summary)
    console.print(f"[green]OK[/green] Wrote {out}")


if __name__ == "__main__":
    main()
