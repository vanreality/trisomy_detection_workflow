#!/usr/bin/env python3
"""Prepare units + job manifests to fill intermediate matrix gaps.

Writes under ``--outdir/intermediate_cache/jobs/``:
  - units_enriched.csv
  - contamination_audit.md
  - missing_after_pct_{modeA,modeB}.csv
  - missing_before_pct_{modeA,modeB}.csv   (all units; thr=0)
  - missing_after_ep_A_from_prod.csv       (wide/beta@0.5 reusable)
  - missing_after_ep_nf_A.csv / _B.csv     (need NF EXTRACT_BETA)
  - nf_extract_missing_A.csv / _B.csv      (Nextflow samplesheet)
  - multi_merge_units.csv                  (one row per multi-batch sample)
  - job_summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "samplesheet_summary"))
from build_intermediate_matrices import (  # noqa: E402
    DEFAULT_BQC,
    DEFAULT_META,
    DEFAULT_MQRES,
    DEFAULT_OUTDIR,
    MODES,
    build_units,
    find_ep_wide,
    yyyymmdd,
)

NOISY_GLOBAL = {"20260203", "20260528", "20260623"}
NOISY_PRED = {"20260310"}  # only dropped for specific noisy-pred samples


def _find_beta(sample: str, root: Path | None) -> Path | None:
    if root is None:
        return None
    p = (
        root
        / "zscore_downstream"
        / "beta_zscore"
        / sample
        / "extract_beta_value"
        / f"{sample}_beta_value.tsv.gz"
    )
    return p if p.is_file() else None


def _enrich(units: pd.DataFrame, bqc: Path) -> pd.DataFrame:
    rows = []
    for r in units.itertuples(index=False):
        root = Path(r.bam_root) if r.bam_root else None
        ep_wide = find_ep_wide(r.sample, root)
        beta = _find_beta(r.sample, root)
        d = r._asdict()
        d["ep_wide_path"] = str(ep_wide) if ep_wide else ""
        d["beta_path"] = str(beta) if beta else ""
        for key, cfg in MODES.items():
            scores = Path(cfg["scores"])
            uid = r.unit_id
            d[f"has_after_pct_{key}"] = (
                scores / "percentage" / f"{uid}.percentage.tsv"
            ).is_file()
            d[f"has_after_ep_{key}"] = (
                scores / "episcore" / f"{uid}.episcore.tsv"
            ).is_file()
        rows.append(d)
    return pd.DataFrame(rows)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--outdir", default=str(DEFAULT_OUTDIR), type=click.Path(file_okay=False))
@click.option("--mqres", default=str(DEFAULT_MQRES), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--bqc", default=str(DEFAULT_BQC), type=click.Path(file_okay=False))
def main(outdir: str, mqres: str, meta: str, bqc: str) -> None:
    out = Path(outdir)
    cache = out / "intermediate_cache"
    jobs = cache / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    bqc_p = Path(bqc)

    mq = pd.read_csv(mqres)
    meta_df = pd.read_csv(meta)
    units = build_units(mq)
    units = _enrich(units, bqc_p)
    units.to_csv(cache / "units.csv", index=False)
    units.to_csv(jobs / "units_enriched.csv", index=False)

    # --- contamination audit ---
    old_path = bqc_p / "units" / "unit_samplesheet.csv"
    lines = [
        "# Intermediate reuse / contamination audit",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Verdict",
        "",
        "- **each_batch after_mq scores**: safe to reuse. BQC files are per-unit "
        "(one deconv → one percentage/episcore). Units dropped with noisy batches "
        "are simply unused; remaining units were never mixed with noisy deconv.",
        "- **merged products**: none on disk yet under `intermediate_cache/merged_*`. "
        "New merges will use **current** mqres only (noisy multi-batch rows already gone).",
        "- **before_mq production wide**: keyed by each unit's own `bam_root`; "
        "not a cross-batch merge. No forced recompute for remaining units.",
        "- **unit_id fix**: always `{sample}__{mqres_batch}-XML` (no BAM-path date).",
        "",
    ]
    noisy_remain = units[units["batch"].astype(str).isin(NOISY_GLOBAL | NOISY_PRED)]
    lines += [
        f"Current units still on globally-noisy / noisy-pred dates: **{len(noisy_remain)}** "
        "(single-batch keepers or non-dropped 20260310 multi-batch — intentional).",
        "",
    ]
    if old_path.is_file():
        old = pd.read_csv(old_path)
        old["batch_ymd"] = old["unit_id"].astype(str).str.extract(r"__(\d{8})")[0]
        cur_set = set(zip(units["sample"], units["batch"].astype(str)))
        old_set = set(zip(old["sample"].astype(str), old["batch_ymd"].astype(str)))
        dropped = sorted(old_set - cur_set)
        lines += [
            f"Old BQC cohort units dropped vs current mqres: **{len(dropped)}** "
            f"(noisy / blacklist).",
            "",
            "| sample | dropped_batch |",
            "|--------|---------------|",
        ]
        for s, b in dropped[:40]:
            lines.append(f"| {s} | {b} |")
        if len(dropped) > 40:
            lines.append(f"| … | +{len(dropped) - 40} more |")
        lines.append("")
    (jobs / "contamination_audit.md").write_text("\n".join(lines) + "\n")
    (out / "contamination_audit.md").write_text("\n".join(lines) + "\n")

    summary = {"n_units": len(units), "n_samples": int(units["sample"].nunique())}

    # --- percentage manifests ---
    for key, cfg in MODES.items():
        label = cfg["label"]
        miss_after = units[~units[f"has_after_pct_{key}"]].copy()
        miss_after.to_csv(jobs / f"missing_after_pct_{label}.csv", index=False)
        # before thr=0: all units (cache may already have some)
        before_dir = cache / f"percentage_thr0_{label}"
        before_dir.mkdir(parents=True, exist_ok=True)
        need_before = []
        for r in units.itertuples(index=False):
            p = before_dir / f"{r.unit_id}.percentage.tsv"
            if not p.is_file():
                need_before.append(r._asdict())
        pd.DataFrame(need_before).to_csv(
            jobs / f"missing_before_pct_{label}.csv", index=False
        )
        summary[f"missing_after_pct_{label}"] = len(miss_after)
        summary[f"missing_before_pct_{label}"] = len(need_before)

    # --- episcore A from production ---
    miss_a = units[~units["has_after_ep_A"]].copy()
    from_prod = miss_a[
        (miss_a["ep_wide_path"].astype(str).str.len() > 0)
        | (miss_a["beta_path"].astype(str).str.len() > 0)
    ].copy()
    from_prod.to_csv(jobs / "missing_after_ep_A_from_prod.csv", index=False)
    need_nf_a = miss_a.drop(from_prod.index)
    need_nf_a.to_csv(jobs / "missing_after_ep_nf_A.csv", index=False)
    # Mode B: production beta is thr=0.5 — always need NF@0.1 for missing
    miss_b = units[~units["has_after_ep_B"]].copy()
    miss_b.to_csv(jobs / "missing_after_ep_nf_B.csv", index=False)

    def _nf_sheet(df: pd.DataFrame, path: Path) -> None:
        nf = pd.DataFrame(
            {
                "sample": df["unit_id"].astype(str),
                "clean_bam": df["clean_bam"].astype(str),
                "deconv_res": df["deconv_res"].astype(str),
            }
        )
        nf.to_csv(path, index=False)

    _nf_sheet(need_nf_a, jobs / "nf_extract_missing_A.csv")
    _nf_sheet(miss_b, jobs / "nf_extract_missing_B.csv")
    # union for logging
    nf_union = pd.concat([need_nf_a, miss_b]).drop_duplicates("unit_id")
    _nf_sheet(nf_union, jobs / "nf_extract_missing_union.csv")

    summary["missing_after_ep_A"] = int((~units["has_after_ep_A"]).sum())
    summary["missing_after_ep_A_from_prod"] = len(from_prod)
    summary["missing_after_ep_nf_A"] = len(need_nf_a)
    summary["missing_after_ep_nf_B"] = len(miss_b)

    # --- multi-batch merge ---
    multi_rows = []
    for sample, ug in units.groupby("sample", sort=False):
        if len(ug) < 2:
            continue
        multi_rows.append(
            {
                "sample": sample,
                "n_batches": len(ug),
                "batches": ",".join(sorted(ug["batch"].astype(str))),
                "deconv_list": " ".join(ug["deconv_res"].astype(str).tolist()),
                "unit_ids": ",".join(ug["unit_id"].astype(str).tolist()),
            }
        )
    multi = pd.DataFrame(multi_rows)
    multi.to_csv(jobs / "multi_merge_units.csv", index=False)
    summary["n_multi_samples"] = len(multi)

    (jobs / "job_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote manifests under {jobs}")


if __name__ == "__main__":
    main()
