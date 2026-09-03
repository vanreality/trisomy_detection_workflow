#!/usr/bin/env python3
"""Fill remaining NA percentage / z_intra cells in modeA each-batch parquet.

Convention (matches the original 935-row matrix):
  percentage before_mq : deconv thr=0.0  recall=0.95
  percentage after_mq  : deconv thr=0.85 recall=0.95
  z_intra after_mq     : beta@0.5 + CpG recall 0.65
  z_intra before_mq    : production wide when present, else the same after_mq
                         episcore (before==after on every already-filled row)

If a score file is missing, writes job CSVs so percentages can be recomputed
and episcore can be harvested from beta or EXTRACT_BETA @0.5.

Hyper-z that is NA because hyper_cpg_count==0 is set to 0 (no CpGs to score).
FF columns are never changed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "samplesheet_summary"))
from build_intermediate_matrices import (  # noqa: E402
    DEFAULT_OUTDIR,
    ep_tsv_to_wide,
    pct_tsv_to_wide,
)
from fill_placeholder_intermediate_modeA import (  # noqa: E402
    NF05_BETA_DIRS,
    _find_beta,
    _pe_deconv,
    _prod_episcore_wide,
    _score_paths,
)

console = Console()
Z_KINDS = ("hypo_z_intra", "hyper_z_intra", "hypo_cpg_count", "hyper_cpg_count")


def _chr_na_report(df: pd.DataFrame) -> dict:
    out = {}
    for label, suffix in (
        ("pct_before", "_percentage_before_mq"),
        ("pct_after", "_percentage_after_mq"),
        ("hypo_z_before", "_hypo_z_intra_before_mq"),
        ("hypo_z_after", "_hypo_z_intra_after_mq"),
        ("hyper_z_before", "_hyper_z_intra_before_mq"),
        ("hyper_z_after", "_hyper_z_intra_after_mq"),
    ):
        cols = [c for c in df.columns if c.endswith(suffix)]
        out[label] = int(df[cols[0]].isna().sum()) if cols else -1
    return out


def _copy_after_to_before(df: pd.DataFrame) -> int:
    """Fill before_mq z/cpg from after_mq when after is present."""
    n = 0
    for kind in Z_KINDS:
        for i in range(1, 23):
            b = f"chr{i}_{kind}_before_mq"
            a = f"chr{i}_{kind}_after_mq"
            if b not in df.columns or a not in df.columns:
                continue
            hit = df[b].isna() & df[a].notna()
            n += int(hit.sum())
            df.loc[hit, b] = df.loc[hit, a]
    return n


def _zero_hyper_z_when_no_cpg(df: pd.DataFrame) -> int:
    n = 0
    for when in ("before_mq", "after_mq"):
        for i in range(1, 23):
            z = f"chr{i}_hyper_z_intra_{when}"
            c = f"chr{i}_hyper_cpg_count_{when}"
            if z not in df.columns or c not in df.columns:
                continue
            hit = df[z].isna() & (df[c] == 0)
            n += int(hit.sum())
            df.loc[hit, z] = 0.0
    return n


def _apply_wide_blocks(df: pd.DataFrame, idx, block: dict) -> int:
    n = 0
    for col, val in block.items():
        if col not in df.columns:
            continue
        if pd.isna(df.at[idx, col]) and pd.notna(val):
            df.at[idx, col] = val
            n += 1
    return n


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--outdir", default=str(DEFAULT_OUTDIR), type=click.Path(file_okay=False))
@click.option(
    "--cmd",
    type=click.Choice(["apply", "jobs", "status"]),
    default="apply",
    show_default=True,
)
def main(outdir: str, cmd: str) -> None:
    out = Path(outdir)
    parquet = out / "intermediate_each_batch_modeA.parquet"
    cache = out / "intermediate_cache"
    jobs = cache / "jobs"
    cache.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(parents=True, exist_ok=True)

    ib = pd.read_parquet(parquet)
    console.print(f"parquet rows={len(ib)} missing={_chr_na_report(ib)}")
    if cmd == "status":
        return

    mq = pd.read_csv(out / "mqres_samplesheet.csv")
    units = _pe_deconv(mq)
    paths = _score_paths(cache)
    u_by = units.set_index(["sample", "dataset"], drop=False)

    # Harvest after_mq holes from on-disk BQC / production / beta-derived tsvs
    n_after = 0
    need_pct = []
    need_ep = []
    need_nf = []
    for idx, row in ib.iterrows():
        key = (row["sample"], row["dataset"])
        if key not in u_by.index:
            continue
        u = u_by.loc[key]
        if isinstance(u, pd.DataFrame):
            u = u.iloc[0]
        uid = str(u["unit_id"])
        root = Path(u["bam_root"]) if pd.notna(u.get("bam_root")) and u["bam_root"] else None

        if pd.isna(row.get("chr1_percentage_after_mq")):
            block = pct_tsv_to_wide(paths["after_pct"] / f"{uid}.percentage.tsv", "after_mq")
            wrote = _apply_wide_blocks(ib, idx, block)
            n_after += wrote
            if wrote == 0:
                need_pct.append({**u.to_dict(), "which": "after"})
        if pd.isna(row.get("chr1_percentage_before_mq")):
            block = pct_tsv_to_wide(paths["before_pct"] / f"{uid}.percentage.tsv", "before_mq")
            wrote = _apply_wide_blocks(ib, idx, block)
            n_after += wrote
            if wrote == 0:
                need_pct.append({**u.to_dict(), "which": "before"})
        if pd.isna(row.get("chr1_hypo_z_intra_after_mq")):
            block = ep_tsv_to_wide(paths["after_ep"] / f"{uid}.episcore.tsv", "after_mq")
            wrote = _apply_wide_blocks(ib, idx, block)
            n_after += wrote
            if wrote == 0:
                rec = u.to_dict()
                rec["ep_wide_path"] = str(_prod_episcore_wide(row["sample"], root) or "")
                rec["beta_path"] = str(_find_beta(row["sample"], uid, root) or "")
                if rec["ep_wide_path"] or rec["beta_path"]:
                    need_ep.append(rec)
                else:
                    need_nf.append(rec)

    n_before = _copy_after_to_before(ib)
    n_zero = _zero_hyper_z_when_no_cpg(ib)
    console.print(
        f"harvested extra cells={n_after}  "
        f"copied after→before z/cpg cells={n_before}  "
        f"zeroed 0-CpG hyper_z cells={n_zero}"
    )

    pd.DataFrame(need_pct).to_csv(jobs / "missing_pct_compute.csv", index=False)
    pd.DataFrame(need_ep).to_csv(jobs / "missing_ep_from_beta.csv", index=False)
    nf = pd.DataFrame(
        {
            "sample": [r["unit_id"] for r in need_nf],
            "clean_bam": [r["clean_bam"] for r in need_nf],
            "deconv_res": [r["deconv_res"] for r in need_nf],
        }
    ) if need_nf else pd.DataFrame(columns=["sample", "clean_bam", "deconv_res"])
    nf.to_csv(jobs / "missing_ep_nf_extract.csv", index=False)
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_need_pct_compute": len(need_pct),
        "n_need_ep_from_beta": len(need_ep),
        "n_need_nf_extract": len(need_nf),
        "copied_after_to_before_cells": n_before,
        "zeroed_hyper_z_cells": n_zero,
        "missing_after_apply": _chr_na_report(ib),
    }
    (jobs / "fill_missing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print(json.dumps(summary, indent=2))

    if cmd == "jobs":
        console.print(f"Wrote job CSVs under {jobs} (parquet not written)")
        return

    still = _chr_na_report(ib)
    leftover = [k for k, v in still.items() if v]
    if leftover and (need_pct or need_ep or need_nf):
        console.print(
            f"[yellow]Still missing {leftover}. Run compute jobs from {jobs} "
            f"then re-run --cmd apply.[/yellow]"
        )
    ib.to_parquet(parquet, index=False)
    console.print(f"Wrote {parquet} missing={still}")


if __name__ == "__main__":
    main()
