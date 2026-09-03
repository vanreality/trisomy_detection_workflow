#!/usr/bin/env python3
"""Fill placeholder rows in ``intermediate_each_batch_modeA.parquet``.

Placeholder = every chromosome feature is NaN (status-0 units appended without
scores). This writes the same feature columns the original 935-row matrix used
(percentage + hypo/hyper z_intra + CpG counts + FF). Ezscore / ref17+25 is a
downstream consumer of these features and is not run here.

Pins (same as the existing modeA score tree):

  episcore  threshold=0.5  recall=0.65
  percentage after_mq  threshold=0.85  recall=0.95
  percentage before_mq threshold=0.0   recall=0.95

Sources, in order:
  FF            production ``*_ff.tsv``, else existing parquet / meta
  after_mq ep   BQC ``*.episcore.tsv``, else ``beta_to_episcore`` on production
                or BQC NF beta@0.5
  after_mq %    BQC ``*.percentage.tsv`` (compute if missing)
  before_mq %   ``intermediate_cache/percentage_thr0_modeA`` (compute if missing)
  before_mq ep  production wide ``*_zscore.tsv`` when present (same proxy as
                the original 935-row matrix)

Writes manifests under ``intermediate_cache/jobs/`` and patches the parquet
in place (non-placeholder rows are not rewritten).
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
    DEFAULT_BQC,
    DEFAULT_OUTDIR,
    MODES,
    bam_root,
    empty_chr_block,
    ep_tsv_to_wide,
    ep_wide_tsv_to_wide,
    find_ep_wide,
    find_ff,
    pct_tsv_to_wide,
    yyyymmdd,
)

console = Console()
MODE_A = MODES["A"]
CPG065 = ROOT / "assets" / "CpG_recall0.65.txt"
NF05_BETA_DIRS = [
    DEFAULT_BQC / "nf_extract_thr0.5" / "extract_beta_value",
    Path(
        "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
        "20260814-intermediate_nf/nf_extract_thr0.5/extract_beta_value"
    ),
    Path(
        "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
        "20260828-placeholder_nf/nf_extract_thr0.5/extract_beta_value"
    ),
]


def _chr_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("chr")]


def placeholder_mask(df: pd.DataFrame) -> pd.Series:
    cols = _chr_cols(df)
    return df[cols].notna().sum(axis=1) == 0


def _pe_deconv(mqres: pd.DataFrame) -> pd.DataFrame:
    df = mqres.copy()
    if "dataset" not in df.columns:
        raise click.ClickException("mqres needs a dataset column")
    df["is_se"] = df["deconv_res"].astype(str).str.contains(
        "single_end", case=False, na=False
    )
    df["_prio"] = df["deconv_res"].map(
        lambda p: (1 if "single_end" in str(p).lower() else 0)
        + (0 if str(p).endswith(".parquet") else 1)
    )
    rows = []
    for (sample, dataset), g in df.groupby(["sample", "dataset"], sort=False):
        g2 = g[~g["is_se"]] if (~g["is_se"]).any() else g
        r = g2.sort_values("_prio").iloc[0]
        date = yyyymmdd(dataset)
        bk = f"{date}-XML" if date else str(dataset)
        root = bam_root(r["clean_bam"])
        rows.append(
            {
                "sample": sample,
                "dataset": dataset,
                "batch": date,
                "batch_key": bk,
                "unit_id": f"{sample}__{bk}",
                "clean_bam": r["clean_bam"],
                "deconv_res": r["deconv_res"],
                "bam_root": str(root) if root else None,
            }
        )
    return pd.DataFrame(rows)


def _find_beta(sample: str, unit_id: str, root: Path | None) -> Path | None:
    if root is not None:
        prod = (
            root
            / "zscore_downstream"
            / "beta_zscore"
            / sample
            / "extract_beta_value"
            / f"{sample}_beta_value.tsv.gz"
        )
        if prod.is_file():
            return prod
    for d in NF05_BETA_DIRS:
        hit = d / f"{unit_id}_beta_value.tsv.gz"
        if hit.is_file():
            return hit
        if d.parent.is_dir():
            found = sorted(d.parent.glob(f"**/{unit_id}_beta_value.tsv.gz"))
            if found:
                return found[0]
    return None


def _prod_episcore_wide(sample: str, root: Path | None) -> Path | None:
    """True methylation episcore wide TSV (not read-count zscore)."""
    if root is None:
        return None
    p = (
        root
        / "zscore_downstream"
        / "beta_zscore"
        / sample
        / "beta_to_episcore"
        / f"{sample}_zscore.tsv"
    )
    return p if p.is_file() else None


def enrich_units(ph: pd.DataFrame, mq_units: pd.DataFrame) -> pd.DataFrame:
    u = ph.merge(mq_units, on=["sample", "dataset"], how="left", suffixes=("", "_u"))
    rows = []
    for r in u.itertuples(index=False):
        sample = str(r.sample)
        uid = str(r.unit_id)
        root = Path(r.bam_root) if pd.notna(r.bam_root) else None
        ff_b, ff_a, _ = find_ff(sample, root)
        if ff_b is None:
            ff_b = r.ff_before_mq if pd.notna(r.ff_before_mq) else None
            ff_a = r.ff_after_mq if pd.notna(r.ff_after_mq) else None
        beta = _find_beta(sample, uid, root)
        ep_true = _prod_episcore_wide(sample, root)
        ep_any = find_ep_wide(sample, root)
        d = {
            "sample": sample,
            "dataset": r.dataset,
            "unit_id": uid,
            "batch": r.batch,
            "batch_key": r.batch_key,
            "clean_bam": r.clean_bam,
            "deconv_res": r.deconv_res,
            "bam_root": str(root) if root else "",
            "ff_before_mq": ff_b,
            "ff_after_mq": ff_a,
            "ep_wide_path": str(ep_true) if ep_true is not None else "",
            "before_ep_wide_path": str(ep_any) if ep_any is not None else "",
            "beta_path": str(beta) if beta is not None else "",
            "deconv_exists": bool(pd.notna(r.deconv_res) and Path(str(r.deconv_res)).is_file()),
            "bam_exists": bool(pd.notna(r.clean_bam) and Path(str(r.clean_bam)).is_file()),
        }
        rows.append(d)
    return pd.DataFrame(rows)


def _score_paths(cache: Path) -> dict[str, Path]:
    scores = Path(MODE_A["scores"])
    return {
        "after_pct": scores / "percentage",
        "after_ep": scores / "episcore",
        "before_pct": cache / "percentage_thr0_modeA",
    }


def file_flags(units: pd.DataFrame, cache: Path) -> pd.DataFrame:
    paths = _score_paths(cache)
    out = units.copy()
    out["has_after_pct"] = [
        (paths["after_pct"] / f"{u}.percentage.tsv").is_file() for u in out["unit_id"]
    ]
    out["has_after_ep"] = [
        (paths["after_ep"] / f"{u}.episcore.tsv").is_file() for u in out["unit_id"]
    ]
    out["has_before_pct"] = [
        (paths["before_pct"] / f"{u}.percentage.tsv").is_file() for u in out["unit_id"]
    ]
    out["has_before_ep"] = out["before_ep_wide_path"].astype(str).str.len() > 0
    out["can_after_ep"] = (out["ep_wide_path"].astype(str).str.len() > 0) | (
        out["beta_path"].astype(str).str.len() > 0
    )
    out["need_nf"] = (~out["has_after_ep"]) & (~out["can_after_ep"])
    return out


def write_jobs(units: pd.DataFrame, jobs: Path) -> dict:
    jobs.mkdir(parents=True, exist_ok=True)
    units.to_csv(jobs / "placeholder_units.csv", index=False)

    need_pct = units[units["deconv_exists"]].copy()
    need_pct.to_csv(jobs / "placeholder_pct.csv", index=False)

    need_ep = units[(~units["has_after_ep"]) & units["can_after_ep"]].copy()
    need_ep.to_csv(jobs / "placeholder_ep.csv", index=False)

    need_nf = units[units["need_nf"]].copy()
    need_nf.to_csv(jobs / "placeholder_need_nf.csv", index=False)
    nf = pd.DataFrame(
        {
            "sample": need_nf["unit_id"].astype(str),
            "clean_bam": need_nf["clean_bam"].astype(str),
            "deconv_res": need_nf["deconv_res"].astype(str),
        }
    )
    nf.to_csv(jobs / "placeholder_nf_extract.csv", index=False)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_placeholder": int(len(units)),
        "n_need_before_pct": int((~units["has_before_pct"]).sum()),
        "n_need_after_pct": int((~units["has_after_pct"]).sum()),
        "n_need_after_ep_from_beta": int(len(need_ep)),
        "n_need_nf_extract": int(len(need_nf)),
        "n_have_ff": int(units["ff_before_mq"].notna().sum()),
        "modeA": {
            "ep_thr": MODE_A["ep_thr"],
            "ep_recall": MODE_A["ep_recall"],
            "pct_thr": MODE_A["pct_thr"],
            "pct_recall": MODE_A["pct_recall"],
            "before_pct_thr": 0.0,
        },
        "need_nf_units": need_nf["unit_id"].astype(str).tolist(),
    }
    (jobs / "placeholder_job_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _row_from_files(r: pd.Series, cache: Path) -> dict:
    uid = str(r["unit_id"])
    paths = _score_paths(cache)
    after_pct = pct_tsv_to_wide(paths["after_pct"] / f"{uid}.percentage.tsv", "after_mq")
    before_pct = pct_tsv_to_wide(
        paths["before_pct"] / f"{uid}.percentage.tsv", "before_mq"
    )
    after_ep = ep_tsv_to_wide(paths["after_ep"] / f"{uid}.episcore.tsv", "after_mq")
    wide = str(r.get("before_ep_wide_path") or "")
    if wide and Path(wide).is_file():
        before_ep = ep_wide_tsv_to_wide(Path(wide), "before_mq")
    else:
        before_ep = {}
        for kind in ("hypo_z_intra", "hyper_z_intra", "hypo_cpg_count", "hyper_cpg_count"):
            before_ep.update(empty_chr_block(f"{kind}_before_mq"))
    return {
        "sample": r["sample"],
        "dataset": r["dataset"],
        "ff_before_mq": r["ff_before_mq"],
        "ff_after_mq": r["ff_after_mq"],
        **before_pct,
        **after_pct,
        **before_ep,
        **after_ep,
    }


def assemble(parquet: Path, units: pd.DataFrame, cache: Path) -> dict:
    ib = pd.read_parquet(parquet)
    chr_cols = _chr_cols(ib)
    ph_idx = ib.index[placeholder_mask(ib)]
    by_key = {(str(r.sample), str(r.dataset)): r for r in units.itertuples(index=False)}

    n_patch = n_skip = 0
    still = []
    for idx in ph_idx:
        sample = str(ib.at[idx, "sample"])
        dataset = str(ib.at[idx, "dataset"])
        rec = by_key.get((sample, dataset))
        if rec is None:
            still.append({"sample": sample, "dataset": dataset, "reason": "no_unit"})
            n_skip += 1
            continue
        filled = _row_from_files(pd.Series(rec._asdict()), cache)
        after_ok = pd.notna(filled.get("chr1_percentage_after_mq")) and pd.notna(
            filled.get("chr1_hypo_z_intra_after_mq")
        )
        before_ok = pd.notna(filled.get("chr1_percentage_before_mq"))
        if not (after_ok and before_ok):
            still.append(
                {
                    "sample": sample,
                    "dataset": dataset,
                    "unit_id": rec.unit_id,
                    "after_pct": bool(pd.notna(filled.get("chr1_percentage_after_mq"))),
                    "after_ep": bool(pd.notna(filled.get("chr1_hypo_z_intra_after_mq"))),
                    "before_pct": bool(pd.notna(filled.get("chr1_percentage_before_mq"))),
                    "before_ep": bool(pd.notna(filled.get("chr1_hypo_z_intra_before_mq"))),
                    "ff": bool(pd.notna(filled.get("ff_before_mq"))),
                }
            )
            # still write whatever we have
        for col, val in filled.items():
            if col in ib.columns and val is not None:
                ib.at[idx, col] = val
        n_patch += 1

    ib.to_parquet(parquet, index=False)
    n_left = int(placeholder_mask(ib).sum())
    report = {
        "patched_rows": n_patch,
        "still_empty": n_left,
        "n_placeholder_before": int(len(ph_idx)),
        "incomplete": still,
    }
    return report


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--outdir", default=str(DEFAULT_OUTDIR), type=click.Path(file_okay=False))
@click.option(
    "--cmd",
    type=click.Choice(["prepare", "assemble", "status"]),
    default="prepare",
    show_default=True,
)
def main(outdir: str, cmd: str) -> None:
    out = Path(outdir)
    parquet = out / "intermediate_each_batch_modeA.parquet"
    mqres = pd.read_csv(out / "mqres_samplesheet.csv")
    cache = out / "intermediate_cache"
    jobs = cache / "jobs"
    cache.mkdir(parents=True, exist_ok=True)

    ib = pd.read_parquet(parquet)
    ph = ib.loc[placeholder_mask(ib), ["sample", "ff_before_mq", "ff_after_mq", "dataset"]].copy()
    console.print(f"parquet={parquet} rows={len(ib)} placeholders={len(ph)}")
    if ph.empty:
        console.print("[green]No placeholder rows[/green]")
        return

    mq_units = _pe_deconv(mqres)
    units = enrich_units(ph, mq_units)
    units = file_flags(units, cache)

    if cmd == "prepare":
        summary = write_jobs(units, jobs)
        console.print(json.dumps(summary, indent=2))
        console.print(f"Wrote manifests under {jobs}")
        return

    if cmd == "status":
        paths = _score_paths(cache)
        console.print(
            f"after_pct {int(units.has_after_pct.sum())}/{len(units)} "
            f"after_ep {int(units.has_after_ep.sum())}/{len(units)} "
            f"before_pct {int(units.has_before_pct.sum())}/{len(units)} "
            f"need_nf {int(units.need_nf.sum())}"
        )
        miss = units[~units["has_after_ep"] | ~units["has_after_pct"] | ~units["has_before_pct"]]
        if len(miss):
            show = miss[
                [
                    "sample",
                    "dataset",
                    "unit_id",
                    "has_after_pct",
                    "has_after_ep",
                    "has_before_pct",
                    "can_after_ep",
                    "need_nf",
                ]
            ]
            console.print(show.to_string(index=False))
        console.print(f"score dirs: {paths}")
        return

    # assemble
    units_path = jobs / "placeholder_units.csv"
    if units_path.is_file():
        units = file_flags(pd.read_csv(units_path), cache)
    report = assemble(parquet, units, cache)
    (jobs / "placeholder_assemble_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    console.print(
        f"patched={report['patched_rows']} still_empty={report['still_empty']} "
        f"incomplete={len(report['incomplete'])}"
    )
    if report["incomplete"]:
        console.print(pd.DataFrame(report["incomplete"]).to_string(index=False))


if __name__ == "__main__":
    main()
