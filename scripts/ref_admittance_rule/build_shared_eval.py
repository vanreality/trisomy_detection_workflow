#!/usr/bin/env python3
"""Build a shared eval list for MAD-rank 40+40 (same eval for 96-dev and 96-test).

Eval = original 20260621 (dev trisomy + test, ff≥1%, not blacklist) minus both
96-pools, plus extra ``set=test & label=Normal & depth_qc=pass`` preferred units
from 20260811-ref_free_batch_qc mode A that are not already in 20260621.

Writes eval_samples.txt and extra grids (new IDs only) for score_repeats.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

from common import (
    DEFAULT_BLACKLIST,
    DEFAULT_FF_MIN,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUT_BASE,
    overlay_ff,
    parse_sample_list,
)

console = Console()

BQC_FIXED = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260811-ref_free_batch_qc"
    "/mode_A_ep0.5_0.65_z0.85_0.95/input_fixed"
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--main-input", default=str(DEFAULT_INPUT_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--bqc-input", default=str(BQC_FIXED), type=click.Path(exists=True, file_okay=False))
@click.option("--dev-pool", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--test-pool", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--ff-min", default=DEFAULT_FF_MIN, show_default=True, type=float)
@click.option("--blacklist", default=",".join(DEFAULT_BLACKLIST), show_default=True)
def main(
    main_input: str,
    bqc_input: str,
    dev_pool: str,
    test_pool: str,
    output_dir: str,
    ff_min: float,
    blacklist: str,
) -> None:
    out = Path(output_dir)
    extra_dir = out / "extra_eval"
    extra_dir.mkdir(parents=True, exist_ok=True)
    bl = {s.strip() for s in blacklist.split(",") if s.strip()}
    pools = set(parse_sample_list(Path(dev_pool))) | set(parse_sample_list(Path(test_pool)))

    main_path = Path(main_input)
    meta = overlay_ff(pd.read_csv(main_path / "meta.csv").drop_duplicates("sample", keep="first"))
    meta["sample"] = meta["sample"].astype(str)
    ep = set(pd.read_parquet(main_path / "episcore_grid_search.parquet")["sample"].astype(str))
    z = set(pd.read_parquet(main_path / "zscore_grid_search.parquet")["sample"].astype(str))
    have = ep & z
    meta = meta.loc[meta["sample"].isin(have)].copy()
    ff = pd.to_numeric(meta["ff_before_mq"], errors="coerce")
    qc = meta["depth_qc"].astype(str).str.lower() if "depth_qc" in meta.columns else pd.Series("pass", index=meta.index)
    is_t = meta["label"].astype(str).str.match(r"^T\d")
    is_n = meta["label"].astype(str).eq("Normal")
    base = (
        ((meta["set"].astype(str) == "dev") & is_t)
        | ((meta["set"].astype(str) == "test") & (is_n | is_t))
    ) & (ff >= ff_min) & ~meta["sample"].isin(bl) & ~meta["sample"].isin(pools)
    # Extra constraint for leftover test Normals: depth_qc=pass when the column exists.
    leftover_n = (meta["set"].astype(str) == "test") & is_n
    base = base & (~leftover_n | (qc == "pass"))
    base_ids = meta.loc[base, "sample"].tolist()
    main_ids = set(meta["sample"])

    chk = pd.read_csv(Path(bqc_input) / "check_samples.tsv", sep="\t")
    chk["orig_sample"] = chk["orig_sample"].astype(str)
    chk["sample"] = chk["sample"].astype(str)
    qc2 = chk["depth_qc"].astype(str).str.lower()
    ff2 = pd.to_numeric(chk["ff_before_mq"], errors="coerce")
    pref = chk["is_preferred_batch"].astype(str).isin(["True", "true", "1"])
    extra_mask = (
        (chk["set"].astype(str) == "test")
        & (chk["label"].astype(str) == "Normal")
        & (qc2 == "pass")
        & pref
        & (ff2 >= ff_min)
        & ~chk["orig_sample"].isin(pools)
        & ~chk["orig_sample"].isin(bl)
        & ~chk["orig_sample"].isin(main_ids)
    )
    extra = chk.loc[extra_mask].drop_duplicates("orig_sample", keep="first").copy()

    extra_meta_rows = extra.rename(columns={"sample": "unit_id", "orig_sample": "sample"})
    extra_ids = extra_meta_rows["sample"].tolist()
    unit_to_orig = dict(zip(extra["sample"], extra["orig_sample"]))

    ep_df = pd.read_parquet(Path(bqc_input) / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(Path(bqc_input) / "zscore_grid_search.parquet")
    ep_df["sample"] = ep_df["sample"].astype(str)
    z_df["sample"] = z_df["sample"].astype(str)
    extra_ep = ep_df.loc[ep_df["sample"].isin(unit_to_orig)].copy()
    extra_z = z_df.loc[z_df["sample"].isin(unit_to_orig)].copy()
    extra_ep["sample"] = extra_ep["sample"].map(unit_to_orig)
    extra_z["sample"] = extra_z["sample"].map(unit_to_orig)
    extra_meta_rows.to_csv(extra_dir / "meta.csv", index=False)
    extra_ep.to_parquet(extra_dir / "episcore_grid_search.parquet", index=False)
    extra_z.to_parquet(extra_dir / "zscore_grid_search.parquet", index=False)

    eval_ids = sorted(set(base_ids) | set(extra_ids))
    rows = []
    for sid in eval_ids:
        if sid in extra_ids:
            r = extra_meta_rows.loc[extra_meta_rows["sample"] == sid].iloc[0]
            rows.append(
                {
                    "sample": sid,
                    "source": "20260811_bqc",
                    "set": r["set"],
                    "label": r["label"],
                    "ff_before_mq": r["ff_before_mq"],
                    "depth_qc": r.get("depth_qc", "pass"),
                }
            )
        else:
            r = meta.loc[meta["sample"] == sid].iloc[0]
            rows.append(
                {
                    "sample": sid,
                    "source": "20260621",
                    "set": r["set"],
                    "label": r["label"],
                    "ff_before_mq": r["ff_before_mq"],
                    "depth_qc": r["depth_qc"] if "depth_qc" in meta.columns else "",
                }
            )
    table = pd.DataFrame(rows)
    is_t_eval = table["label"].astype(str).str.match(r"^T\d")
    (out / "eval_samples.txt").write_text("\n".join(eval_ids) + "\n")
    table.to_csv(out / "eval_samples.tsv", sep="\t", index=False)
    summary = {
        "n_eval": len(eval_ids),
        "n_normal": int((~is_t_eval).sum()),
        "n_trisomy": int(is_t_eval.sum()),
        "n_from_20260621": int((table["source"] == "20260621").sum()),
        "n_from_20260811": int((table["source"] == "20260811_bqc").sum()),
        "n_pool_blocked": len(pools),
        "ff_min": ff_min,
        "extra_input_dir": str(extra_dir),
        "rule": (
            "shared eval = (20260621 dev-trisomy + test N/T, ff>=ff_min, not blacklist, "
            "not in either 96-pool, test Normal also depth_qc=pass) UNION "
            "(20260811 preferred test Normal depth_qc=pass ff>=ff_min, orig not in 20260621)"
        ),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print(
        f"eval n={summary['n_eval']} N={summary['n_normal']} T={summary['n_trisomy']} "
        f"bqc_extra={summary['n_from_20260811']} -> {out}"
    )


if __name__ == "__main__":
    main()
