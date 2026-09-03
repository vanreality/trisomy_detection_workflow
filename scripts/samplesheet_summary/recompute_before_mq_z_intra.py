#!/usr/bin/env python3
"""Recompute ``chr*_hypo/hyper_z_intra_before_mq`` in DB table ``中游数据``.

After-MQ z_intra (current DB values) uses ``target_meth_count`` /
``target_unmeth_count``.  Before-MQ must use ``raw_meth_count`` /
``raw_unmeth_count`` from the same extract_beta_value file.

Mode A pins: CpG recall 0.65, depth 30 on ``raw_total_count``, chr1–22.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, "/lustre1/cqyi/AIPT_2.0/scripts/tools")
from beta_to_episcore import (  # noqa: E402
    calculate_chr_level_beta,
    calculate_s_intra,
    read_beta,
)
from db_helper import AIPTDatabase, BIOINFO_PATH_TABLE  # noqa: E402

console = Console()

TABLE = "中游数据"
CHR_LIST = [f"chr{i}" for i in range(1, 23)]
Z_BEFORE_COLS = [
    f"chr{i}_{kind}_before_mq"
    for kind in ("hypo_z_intra", "hyper_z_intra")
    for i in range(1, 23)
]
CPG_LIST = ROOT / "assets" / "CpG_recall0.65.txt"
DEPTH = 30
PRIMARY = Path(
    "/appsnew/home/myli/lustre1/bert/DNA_5mC_analysis_pipeline/output"
)
SECONDARY = Path(
    "/lustre1/cqyi/syfan/snp_nipt/results/beta_trisomy_detection/"
    "20260403_summary/beta_to_zscore"
)
BROAD_B = Path("/lustre1/cqyi/syfan/snp_nipt/results/beta_trisomy_detection")
DATA_RUN = Path(
    "/lustre1/cqyi/yfan/workflow/NIPT/00.data/target_data/data_run/output"
)
NF_BETA_DIRS = [
    Path(
        "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
        "20260814-intermediate_nf/nf_extract_thr0.5/extract_beta_value"
    ),
    Path(
        "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
        "20260828-placeholder_nf/nf_extract_thr0.5/extract_beta_value"
    ),
    Path(
        "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
        "20260811-ref_free_batch_qc/nf_extract_thr0.5/extract_beta_value"
    ),
]
DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")
DATE_RE = re.compile(r"(\d{8})")
PTAY_RE = re.compile(r"^(?:J)?PTAY", re.I)
USECOLS_RAW = [
    "chr",
    "start",
    "end",
    "raw_meth_count",
    "raw_unmeth_count",
    "raw_total_count",
    "meandiff",
]
USECOLS_TARGET = [
    "chr",
    "start",
    "end",
    "target_meth_count",
    "target_unmeth_count",
    "raw_total_count",
    "meandiff",
]


def _pk_col(df: pl.DataFrame) -> str:
    hits = [c for c in df.columns if str(c).endswith("_id") and "中游数据" in c]
    if not hits:
        raise click.ClickException(f"No 中游数据 PK in {df.columns[:8]}")
    return hits[0]


def p_to_b_sample(sample: str) -> str:
    last_p = sample.rfind("P")
    if last_p != -1:
        return sample[:last_p] + "B" + sample[last_p + 1 :]
    return sample


def sample_aliases(sample: str) -> list[str]:
    s = str(sample).strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(s)
    if PTAY_RE.match(s):
        add(p_to_b_sample(s))
    for suf in ("_rep1", "_rep2", "_1", "_2"):
        add(f"{s}{suf}")
        if PTAY_RE.match(s):
            add(f"{p_to_b_sample(s)}{suf}")
    return out


def dataset_aliases(dataset: str) -> list[str]:
    d = str(dataset).strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(d)
    if "," in d:
        return out
    if d.endswith(".igtc"):
        add(d[: -len(".igtc")])
        add(d.replace("-XML.igtc", "-XML_igtc"))
    if d.endswith("_igtc"):
        add(d[: -len("_igtc")])
        add(d.replace("-XML_igtc", "-XML.igtc"))
    m = DATE_RE.search(d)
    if m:
        date = m.group(1)
        for suf in ("-XML", "-XML_igtc", "-XML.igtc", "-240k"):
            add(date + suf)
    return out


def split_merged_dates(dataset: str) -> list[str]:
    d = str(dataset).strip()
    if "," not in d:
        return []
    return [p.strip() for p in d.split(",") if p.strip()]


def beta_under_root(root: Path, dataset: str, sample: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    for alias in dataset_aliases(dataset):
        for disk in sample_aliases(sample):
            p = (
                root
                / alias
                / "bwameth_results"
                / "zscore_downstream"
                / "beta_zscore"
                / disk
                / "extract_beta_value"
                / f"{disk}_beta_value.tsv.gz"
            )
            if p.is_file():
                return p
    return None


def beta_from_bam_root(clean_bam: str, sample: str) -> Optional[Path]:
    p = Path(str(clean_bam))
    if "bwameth_results" not in p.parts:
        return None
    i = p.parts.index("bwameth_results")
    bam_root = Path(*p.parts[: i + 1])
    for disk in sample_aliases(sample):
        hit = (
            bam_root
            / "zscore_downstream"
            / "beta_zscore"
            / disk
            / "extract_beta_value"
            / f"{disk}_beta_value.tsv.gz"
        )
        if hit.is_file():
            return hit
    return None


def beta_from_nf(sample: str, dataset: str) -> Optional[Path]:
    if "," in str(dataset):
        return None
    names: list[str] = []
    for disk in sample_aliases(sample):
        for alias in dataset_aliases(dataset):
            names.append(f"{disk}__{alias}_beta_value.tsv.gz")
        for date in DATE_RE.findall(dataset):
            names.append(f"{disk}__{date}-XML_beta_value.tsv.gz")
    for d in NF_BETA_DIRS:
        if not d.is_dir():
            continue
        for name in names:
            hit = d / name
            if hit.is_file():
                return hit
    return None


def beta_from_secondary(sample: str) -> Optional[Path]:
    for name in (
        f"{sample}_beta_value.tsv.gz",
        f"{sample}_beta_value.tsv",
    ):
        hit = SECONDARY / name
        if hit.is_file():
            return hit
    return None


def scan_tree_for_sample(root: Path, sample: str, max_hits: int = 8) -> list[Path]:
    if not root.is_dir():
        return []
    hits: list[Path] = []
    disks = sample_aliases(sample)
    try:
        kids = list(root.iterdir())
    except OSError:
        return []
    for child in kids:
        if not child.is_dir():
            continue
        for disk in disks:
            name = f"{disk}_beta_value.tsv.gz"
            p = (
                child
                / "bwameth_results"
                / "zscore_downstream"
                / "beta_zscore"
                / disk
                / "extract_beta_value"
                / name
            )
            if p.is_file():
                hits.append(p)
                if len(hits) >= max_hits:
                    return hits
            p2 = child / "extract_beta_value" / name
            if p2.is_file():
                hits.append(p2)
                if len(hits) >= max_hits:
                    return hits
    return hits


def pe_path_map(paths: pl.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    df = paths.with_columns(
        pl.col("sample").cast(pl.Utf8).str.strip_chars(),
        pl.col("dataset").cast(pl.Utf8).str.strip_chars(),
        pl.col("deconv_res").cast(pl.Utf8),
        pl.col("clean_bam").cast(pl.Utf8),
    )
    out: dict[tuple[str, str], dict[str, str]] = {}
    for rec in df.iter_rows(named=True):
        sample, dataset = rec["sample"], rec["dataset"]
        deconv = rec.get("deconv_res") or ""
        bam = rec.get("clean_bam") or ""
        if not sample or not dataset:
            continue
        is_se = "single_end" in str(deconv).lower()
        key = (sample, dataset)
        cur = out.get(key)
        if cur is None or (cur.get("is_se") and not is_se):
            out[key] = {
                "clean_bam": bam,
                "deconv_res": deconv,
                "is_se": is_se,
            }
    return out


def locate_one(
    sample: str,
    dataset: str,
    path_lookup: dict[tuple[str, str], dict[str, str]],
    *,
    allow_scan: bool = True,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "sample": sample,
        "dataset": dataset,
        "beta_path": "",
        "source": "",
        "merged_parts": "",
    }
    dates = split_merged_dates(dataset)
    if dates:
        parts: list[str] = []
        for date in dates:
            sub = locate_one(sample, date, path_lookup, allow_scan=False)
            if sub["beta_path"] and "|" not in str(sub["beta_path"]):
                parts.append(sub["beta_path"])
                continue
            for s, ds in path_lookup:
                if s != sample or not str(ds).startswith(date):
                    continue
                sub2 = locate_one(sample, ds, path_lookup, allow_scan=False)
                if sub2["beta_path"] and "|" not in str(sub2["beta_path"]):
                    parts.append(sub2["beta_path"])
                    break
        if parts:
            uniq = list(dict.fromkeys(parts))
            rec["beta_path"] = "|".join(uniq)
            rec["source"] = "merged_parts" if len(uniq) > 1 else "merged_one_part"
            rec["merged_parts"] = rec["beta_path"]
            return rec

    hit = beta_under_root(PRIMARY, dataset, sample)
    if hit:
        rec["beta_path"] = str(hit)
        rec["source"] = "primary"
        return rec
    hit = beta_from_secondary(sample)
    if hit:
        rec["beta_path"] = str(hit)
        rec["source"] = "secondary"
        return rec

    info = path_lookup.get((sample, dataset))
    if info and info.get("clean_bam"):
        hit = beta_from_bam_root(info["clean_bam"], sample)
        if hit:
            rec["beta_path"] = str(hit)
            rec["source"] = "bam_root"
            return rec

    hit = beta_under_root(DATA_RUN, dataset, sample)
    if hit:
        rec["beta_path"] = str(hit)
        rec["source"] = "data_run"
        return rec

    hit = beta_from_nf(sample, dataset)
    if hit:
        rec["beta_path"] = str(hit)
        rec["source"] = "nf_extract"
        return rec

    if allow_scan:
        for root, tag in ((PRIMARY, "primary_scan"), (DATA_RUN, "data_run_scan")):
            found = scan_tree_for_sample(root, sample)
            if found:
                rec["beta_path"] = str(found[0])
                rec["source"] = tag
                return rec
        found = scan_tree_for_sample(BROAD_B, sample)
        if found:
            rec["beta_path"] = str(found[0])
            rec["source"] = "beta_trisomy_scan"
            return rec
    return rec


_CPG_FILTER: Optional[list[str]] = None


def cpg_filter() -> list[str]:
    global _CPG_FILTER
    if _CPG_FILTER is None:
        df = pd.read_csv(CPG_LIST, sep="\t", usecols=["chr", "start", "end"])
        _CPG_FILTER = (
            df["chr"].astype(str)
            + ":"
            + df["start"].astype(str)
            + "-"
            + df["end"].astype(str)
        ).tolist()
    return _CPG_FILTER


def _z_from_frames(
    hypo: pd.DataFrame,
    hyper: pd.DataFrame,
    meth_col: str,
    unmeth_col: str,
) -> dict[str, float]:
    hypo_b, hyper_b, hypo_c, hyper_c = calculate_chr_level_beta(
        hypo, hyper, CHR_LIST, meth_col, unmeth_col
    )
    hypo_z, hyper_z, _ = calculate_s_intra(
        hypo_b, hyper_b, hypo_c, hyper_c, CHR_LIST
    )
    out: dict[str, float] = {}
    for i, chr_name in enumerate(CHR_LIST):
        hz = hypo_z[i]
        xz = hyper_z[i]
        if hypo_c[chr_name] == 0:
            hz = np.nan
        if hyper_c[chr_name] == 0:
            xz = 0.0
        out[f"{chr_name}_hypo_z_intra"] = float(round(hz, 6)) if hz == hz else np.nan
        out[f"{chr_name}_hyper_z_intra"] = float(round(xz, 6)) if xz == xz else np.nan
    return out


def compute_from_beta_paths(
    beta_paths: list[str],
    meth_col: str,
    unmeth_col: str,
    usecols: list[str],
) -> dict[str, float]:
    hypos = []
    hypers = []
    for p in beta_paths:
        hypo, hyper, _ = read_beta(
            p,
            usecols=usecols,
            filter_depth=None if len(beta_paths) > 1 else DEPTH,
            depth_col="raw_total_count",
            cpg_filter=cpg_filter(),
            chr_list=CHR_LIST,
        )
        hypos.append(hypo)
        hypers.append(hyper)
    if len(hypos) == 1:
        return _z_from_frames(hypos[0], hypers[0], meth_col, unmeth_col)
    hypo = pd.concat(hypos, ignore_index=True)
    hyper = pd.concat(hypers, ignore_index=True)
    keys = ["chr", "start", "end"]
    hypo = hypo.groupby(keys, as_index=False).agg({meth_col: "sum", unmeth_col: "sum"})
    hyper = hyper.groupby(keys, as_index=False).agg({meth_col: "sum", unmeth_col: "sum"})
    hypo["_tot"] = hypo[meth_col] + hypo[unmeth_col]
    hyper["_tot"] = hyper[meth_col] + hyper[unmeth_col]
    hypo = hypo[hypo["_tot"] > DEPTH].drop(columns=["_tot"])
    hyper = hyper[hyper["_tot"] > DEPTH].drop(columns=["_tot"])
    return _z_from_frames(hypo, hyper, meth_col, unmeth_col)


def _compute_worker(args: tuple) -> dict[str, Any]:
    sample, dataset, beta_path, source = args
    rec = {
        "sample": sample,
        "dataset": dataset,
        "source": source,
        "beta_path": beta_path,
        "ok": False,
        "error": "",
    }
    try:
        paths = [p for p in str(beta_path).split("|") if p]
        z = compute_from_beta_paths(
            paths, "raw_meth_count", "raw_unmeth_count", USECOLS_RAW
        )
        for i in range(1, 23):
            rec[f"chr{i}_hypo_z_intra_before_mq"] = z[f"chr{i}_hypo_z_intra"]
            rec[f"chr{i}_hyper_z_intra_before_mq"] = z[f"chr{i}_hyper_z_intra"]
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)
    return rec


def fetch_mid() -> tuple[pl.DataFrame, str]:
    with AIPTDatabase() as db:
        mid = db.fetch_table(TABLE)
        paths = db.fetch_table(BIOINFO_PATH_TABLE)
    return mid, paths


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--cmd",
    type=click.Choice(["locate", "compute", "upload", "all"]),
    default="all",
    show_default=True,
)
@click.option("--outdir", default=str(DEFAULT_OUTDIR), type=click.Path(file_okay=False))
@click.option("--workers", default=16, show_default=True, type=int)
@click.option("--validate-n", default=12, show_default=True, type=int)
@click.option("--dry-run", is_flag=True, default=False)
def main(cmd: str, outdir: str, workers: int, validate_n: int, dry_run: bool) -> None:
    out = Path(outdir)
    cache = out / "before_mq_z_intra"
    cache.mkdir(parents=True, exist_ok=True)
    locate_path = cache / "beta_locate.csv"
    scores_path = cache / "before_mq_scores.parquet"
    jobs_path = cache / "nf_extract_missing.csv"
    summary_path = cache / "recompute_summary.json"

    console.print("Fetching 中游数据 + 生信分析文件路径 ...")
    with AIPTDatabase() as db:
        mid = db.fetch_table(TABLE)
        paths = db.fetch_table(BIOINFO_PATH_TABLE)
    pk = _pk_col(mid)
    lookup = pe_path_map(paths)
    units = (
        mid.select([pk, "sample", "dataset"])
        .with_columns(
            pl.col("sample").cast(pl.Utf8).str.strip_chars(),
            pl.col("dataset").cast(pl.Utf8).str.strip_chars(),
        )
        .to_pandas()
    )
    console.print(f"combos={len(units)}")

    if cmd in {"locate", "compute", "all"}:
        rows = []
        for r in units.to_dict("records"):
            rec = locate_one(str(r["sample"]), str(r["dataset"]), lookup)
            rec[pk] = int(r[pk])
            rows.append(rec)
        loc = pd.DataFrame(rows)
        loc.to_csv(locate_path, index=False)
        src = loc["source"].fillna("").replace("", "missing").value_counts().to_dict()
        n_miss = int((loc["beta_path"].fillna("") == "").sum())
        console.print(f"locate sources={src} missing={n_miss} wrote {locate_path}")

        missing = loc[loc["beta_path"].fillna("") == ""].copy()
        job_rows = []
        for r in missing.itertuples(index=False):
            info = lookup.get((r.sample, r.dataset))
            if info is None and "," in str(r.dataset):
                # first constituent with paths
                for date in split_merged_dates(r.dataset):
                    for (s, ds), inf in lookup.items():
                        if s == r.sample and str(ds).startswith(date):
                            info = inf
                            break
                    if info:
                        break
            if not info:
                continue
            bam, deconv = info.get("clean_bam"), info.get("deconv_res")
            if bam and deconv and Path(bam).is_file() and Path(deconv).is_file():
                job_rows.append(
                    {
                        "sample": f"{r.sample}__{r.dataset}",
                        "clean_bam": bam,
                        "deconv_res": deconv,
                        "orig_sample": r.sample,
                        "dataset": r.dataset,
                    }
                )
        pd.DataFrame(job_rows).to_csv(jobs_path, index=False)
        console.print(f"nf extract candidates={len(job_rows)} wrote {jobs_path}")

    loc = pd.read_csv(locate_path)
    have = loc[loc["beta_path"].fillna("") != ""].copy()

    if cmd in {"compute", "all"} and validate_n > 0:
        console.print(f"Validating after_mq vs target counts on {validate_n} combos ...")
        mid_pd = mid.to_pandas()
        ok = 0
        checked = 0
        max_abs = 0.0
        for r in have.itertuples(index=False):
            if checked >= validate_n:
                break
            if "|" in str(r.beta_path):
                continue
            db_row = mid_pd[
                (mid_pd["sample"].astype(str) == str(r.sample))
                & (mid_pd["dataset"].astype(str) == str(r.dataset))
            ]
            if db_row.empty:
                continue
            try:
                z = compute_from_beta_paths(
                    [r.beta_path],
                    "target_meth_count",
                    "target_unmeth_count",
                    USECOLS_TARGET,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"  skip {r.sample}/{r.dataset}: {exc}")
                continue
            diffs = []
            db0 = db_row.iloc[0]
            for i in range(1, 23):
                a = float(db0[f"chr{i}_hypo_z_intra_after_mq"])
                b = z[f"chr{i}_hypo_z_intra"]
                if a == a and b == b:
                    diffs.append(abs(a - b))
            if not diffs:
                continue
            checked += 1
            peak = max(diffs)
            max_abs = max(max_abs, peak)
            if peak < 1e-4:
                ok += 1
            else:
                console.print(
                    f"  mismatch {r.sample}/{r.dataset} max|d|={peak:.4g} source={r.source}"
                )
        console.print(f"after_mq validate ok={ok}/{checked} max_abs={max_abs:.4g}")

    if cmd in {"compute", "all"}:
        tasks = [
            (str(r.sample), str(r.dataset), str(r.beta_path), str(r.source))
            for r in have.itertuples(index=False)
        ]
        console.print(f"Computing before_mq for {len(tasks)} combos workers={workers}")
        recs: list[dict[str, Any]] = []
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(_compute_worker, t) for t in tasks]
            for fut in as_completed(futs):
                recs.append(fut.result())
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    console.print(f"  computed {done}/{len(tasks)}")
        scores = pd.DataFrame(recs)
        scores.to_parquet(scores_path, index=False)
        n_ok = int(scores["ok"].sum()) if "ok" in scores.columns else 0
        n_err = len(scores) - n_ok
        console.print(f"compute ok={n_ok} err={n_err} wrote {scores_path}")
        if n_err:
            print(scores.loc[~scores["ok"], ["sample", "dataset", "error"]].head(20))

    if cmd in {"upload", "all"}:
        scores = pd.read_parquet(scores_path)
        ok = scores[scores["ok"]].copy()
        loc_ok = loc.merge(ok, on=["sample", "dataset"], how="inner", suffixes=("", "_s"))
        mid_pd = mid.to_pandas()
        patch = mid_pd[[pk, "sample", "dataset"]].merge(
            ok[["sample", "dataset", *Z_BEFORE_COLS]],
            on=["sample", "dataset"],
            how="inner",
        )
        # compare before vs after on a few stats
        merged = mid_pd.merge(
            ok[["sample", "dataset", *Z_BEFORE_COLS]],
            on=["sample", "dataset"],
            how="left",
            suffixes=("", "_new"),
        )
        n_changed = 0
        n_same = 0
        abs_d = []
        for i in range(1, 23):
            old = merged[f"chr{i}_hypo_z_intra_before_mq"]
            new = merged[f"chr{i}_hypo_z_intra_before_mq_new"]
            after = merged[f"chr{i}_hypo_z_intra_after_mq"]
            both = new.notna()
            d = (old - new).abs()
            n_changed += int((both & (d > 1e-8)).sum())
            n_same += int((both & (d <= 1e-8)).sum())
            aa = (after - new).abs()
            abs_d.extend(aa[both & after.notna() & new.notna()].tolist())
        console.print(
            f"hypo cells changed vs old before={n_changed} same={n_same} "
            f"median |after-new|={float(np.nanmedian(abs_d)) if abs_d else float('nan'):.4g}"
        )

        n_upload = len(patch)
        n_skip = len(units) - n_upload
        summary = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "n_combos": int(len(units)),
            "n_located": int((loc["beta_path"].fillna("") != "").sum()),
            "n_missing": int((loc["beta_path"].fillna("") == "").sum()),
            "locate_sources": loc["source"].fillna("").replace("", "missing").value_counts().to_dict(),
            "n_computed_ok": int(ok.shape[0]),
            "n_upload_rows": int(n_upload),
            "n_still_missing": int(n_skip),
            "n_nf_extract_jobs": 0,
            "dry_run": bool(dry_run),
        }
        missing_units = loc.loc[loc["beta_path"].fillna("") == "", ["sample", "dataset"]]
        summary["missing_combos"] = missing_units.to_dict(orient="records")
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        console.print(json.dumps({k: v for k, v in summary.items() if k != "missing_combos"}, indent=2))

        if dry_run:
            console.print("[yellow]dry-run: skip DB write[/yellow]")
            return

        patch_pl = pl.from_pandas(patch[[pk, *Z_BEFORE_COLS]])
        console.print(f"Uploading {n_upload} rows × {len(Z_BEFORE_COLS)} z_intra before_mq cols ...")
        with AIPTDatabase() as db:
            db.update_table(TABLE, patch_pl)
        console.print(f"summary wrote {summary_path}")


if __name__ == "__main__":
    main()
