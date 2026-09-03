#!/usr/bin/env python3
"""Fill Mode-B (thr=0.1 / recall=0.61) unit episcores from existing assets.

Sources (in order):
  1. Already-present ``{ep_dir}/{unit_id}.episcore.tsv``
  2. Long episcore rows in ref_free / grid_search parquets (sample-level → unit_id)
  3. Sample- or unit-named ``*_beta_value.tsv.gz`` under known thr=0.1 trees
     (converted via ``beta_to_episcore.py``)

Writes a fill report and an NF gap samplesheet for units still missing.
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
CPG_061 = (
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260525-grid_search_240k_panel_240k_model/recall_list_220k/220k_cpg_recall_0.61.txt"
)
CHR_LIST = [f"chr{i}" for i in range(1, 23)]

DEFAULT_PARQUETS = [
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260721-ref_free/input_with_val_filtered/episcore_grid_search.parquet",
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260721-ref_free/input_with_val/episcore_grid_search.parquet",
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size/fixed_ez25/input_with_missing4/episcore_grid_search.parquet",
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng/episcore_grid_search.parquet",
]

DEFAULT_BETA_ROOTS = [
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc/nf_extract_thr0.1",
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260721-ref_free/val_filtered_grid/episcore/nf_extract_thr0.1",
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260721-ref_free/val_filtered_grid/episcore/analyze_thr0.1",
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260525-grid_search_240k_panel_240k_model/threshold_0.1",
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260508-grid_search/threshold_0.1",
]


def _index_betas(roots: list[str]) -> tuple[dict[str, Path], dict[str, Path]]:
    by_unit: dict[str, Path] = {}
    by_sample: dict[str, Path] = {}
    for root in roots:
        d = Path(root)
        if not d.is_dir():
            continue
        for p in d.rglob("*_beta_value.tsv.gz"):
            stem = p.name.replace("_beta_value.tsv.gz", "")
            if "__" in stem:
                by_unit.setdefault(stem, p)
            else:
                by_sample.setdefault(stem, p)
    return by_unit, by_sample


def _load_parquet_ep(paths: list[str], threshold: float, recall: float) -> pd.DataFrame:
    frames = []
    cols = [
        "sample",
        "chr",
        "threshold",
        "recall",
        "hypo_z_intra",
        "hyper_z_intra",
        "hypo_cpgs_count",
        "hyper_cpgs_count",
    ]
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        ep = pd.read_parquet(path, columns=cols)
        sub = ep[(ep.threshold == threshold) & (ep.recall == recall)]
        console.print(f"  parquet {path.parent.name}: samples={sub['sample'].nunique()}")
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["sample", "chr"], keep="first")


def _beta_to_unit_ep(beta: Path, unit_id: str, dest: Path, threshold: float, recall: float, depth: int) -> None:
    with tempfile.TemporaryDirectory(prefix="fill_ep_") as tmp:
        prefix = str(Path(tmp) / unit_id)
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
                CPG_061,
                "--chr-list",
                "1-22",
            ],
            check=True,
        )
        wide = pd.read_csv(f"{prefix}_zscore.tsv", sep="\t")
        row = wide.iloc[0]
        records = []
        for chr_name in CHR_LIST:
            records.append(
                {
                    "sample": unit_id,
                    "chr": chr_name,
                    "threshold": threshold,
                    "recall": recall,
                    "hypo_z_intra": float(row[f"{chr_name}_hypo_z_intra"]),
                    "hyper_z_intra": float(row[f"{chr_name}_hyper_z_intra"]),
                    "hypo_cpgs_count": float(row[f"{chr_name}_hypo_cpgs_count"]),
                    "hyper_cpgs_count": float(row[f"{chr_name}_hyper_cpgs_count"]),
                }
            )
        pd.DataFrame(records).to_csv(dest, sep="\t", index=False)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--ep-dir", required=True, type=click.Path(file_okay=False))
@click.option("--threshold", default=0.1, show_default=True, type=float)
@click.option("--recall", default=0.61, show_default=True, type=float)
@click.option("--depth", default=30, type=int)
@click.option("--gap-samplesheet", required=True, type=click.Path(dir_okay=False))
@click.option("--report", required=True, type=click.Path(dir_okay=False))
def main(
    units: str,
    ep_dir: str,
    threshold: float,
    recall: float,
    depth: int,
    gap_samplesheet: str,
    report: str,
) -> None:
    udf = pd.read_csv(units)
    out = Path(ep_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print("Indexing thr=0.1 betas ...")
    by_unit, by_sample = _index_betas(DEFAULT_BETA_ROOTS)
    console.print(f"  unit betas={len(by_unit)} sample betas={len(by_sample)}")
    console.print("Loading parquet episcores ...")
    pq = _load_parquet_ep(DEFAULT_PARQUETS, threshold, recall)
    pq_by_sample = {s: g for s, g in pq.groupby(pq["sample"].astype(str))} if len(pq) else {}

    rows = []
    for _, r in udf.iterrows():
        uid = str(r["unit_id"])
        sample = str(r["sample"])
        dest = out / f"{uid}.episcore.tsv"
        if dest.is_file():
            rows.append({"unit_id": uid, "source": "existing", "detail": ""})
            continue
        if sample in pq_by_sample:
            g = pq_by_sample[sample].copy()
            g["sample"] = uid
            g = g.set_index("chr").reindex(CHR_LIST).reset_index()
            if g[["hypo_z_intra", "hyper_z_intra"]].isna().any().any():
                rows.append({"unit_id": uid, "source": "parquet_incomplete", "detail": sample})
            else:
                g["threshold"] = threshold
                g["recall"] = recall
                g.to_csv(dest, sep="\t", index=False)
                rows.append({"unit_id": uid, "source": "parquet", "detail": sample})
                continue
        beta = by_unit.get(uid) or by_sample.get(sample)
        if beta is not None:
            try:
                _beta_to_unit_ep(beta, uid, dest, threshold, recall, depth)
                rows.append({"unit_id": uid, "source": "beta", "detail": str(beta)})
            except Exception as exc:  # noqa: BLE001
                rows.append({"unit_id": uid, "source": "beta_fail", "detail": f"{beta}: {exc}"})
            continue
        rows.append({"unit_id": uid, "source": "nf", "detail": ""})

    plan = pd.DataFrame(rows)
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(report, index=False)
    console.print("sources:", plan["source"].value_counts().to_dict())

    nf_uids = set(plan.loc[plan["source"].isin(["nf", "beta_fail", "parquet_incomplete"]), "unit_id"])
    gap = udf[udf["unit_id"].astype(str).isin(nf_uids)][["unit_id", "clean_bam", "deconv_res"]].copy()
    gap = gap.rename(columns={"unit_id": "sample"})
    Path(gap_samplesheet).parent.mkdir(parents=True, exist_ok=True)
    gap.to_csv(gap_samplesheet, index=False)
    console.print(f"NF gap samplesheet n={len(gap)} -> {gap_samplesheet}")
    console.print(f"report -> {report}")


if __name__ == "__main__":
    main()
