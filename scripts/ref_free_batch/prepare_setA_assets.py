#!/usr/bin/env python3
"""Build slim ref_free input for Set A pool-size exploration.

Universe:
  * ``ref_pool``     — 96 dev Normal from main input (always the candidate core)
  * ``filler_pool``  — test Normal from main input (nested fillers when pool>96)
  * ``eval``         — meta-filtered Set A, excluding PTAY1351P8S1 and samples
    whose orig_sample is in the 96-ref pool

Eval scores: batch-QC unit ep/z when present; otherwise main parquet at the
mode combo (sample id rewritten to unit_id so it does not collide with refs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_OUT,
    DROP_SAMPLES,
    MAIN_INPUT,
    MODES,
)

console = Console()
CHR_LIST = [f"chr{i}" for i in range(1, 23)]


def _need_ep() -> list[str]:
    return [
        "sample",
        "chr",
        "threshold",
        "recall",
        "hypo_z_intra",
        "hyper_z_intra",
        "hypo_cpgs_count",
        "hyper_cpgs_count",
    ]


def _read_unit_ep(path: Path, uid: str, ep_thr: float, ep_rec: float) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    ep = pd.read_csv(path, sep="\t")
    ep["sample"] = uid
    ep["threshold"] = ep_thr
    ep["recall"] = ep_rec
    return ep[_need_ep()]


def _read_unit_z(path: Path, uid: str, z_thr: float, z_rec: float) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    zdf = pd.read_csv(path, sep="\t")
    zdf = zdf[zdf["chr"].isin(CHR_LIST)].copy()
    zdf["sample"] = uid
    zdf["threshold"] = z_thr
    zdf["recall"] = z_rec
    return zdf[["sample", "chr", "threshold", "recall", "percentage"]]


def _from_main(
    main_df: pd.DataFrame,
    orig: str,
    uid: str,
    thr: float,
    rec: float,
    cols: list[str],
) -> pd.DataFrame | None:
    sub = main_df[
        (main_df["sample"].astype(str) == orig)
        & (main_df["threshold"] == thr)
        & (main_df["recall"] == rec)
    ].copy()
    if sub.empty:
        return None
    sub["sample"] = uid
    return sub[cols]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--mode", required=True, type=click.Choice(sorted(MODES)))
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--cohort-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--main-input",
    default=str(MAIN_INPUT),
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def main(mode: str, output_dir: Path | None, cohort_dir: Path | None, main_input: Path) -> None:
    cfg = MODES[mode]
    out = output_dir or (DEFAULT_OUT / mode / "input")
    out.mkdir(parents=True, exist_ok=True)
    cohort = cohort_dir or (DEFAULT_OUT / "cohort")

    ep_thr, ep_rec = cfg["ep_threshold"], cfg["ep_recall"]
    z_thr, z_rec = cfg["z_threshold"], cfg["z_recall"]
    ep_root = cfg["bqc_dir"] / "scores" / "episcore"
    z_root = cfg["bqc_dir"] / "scores" / "percentage"

    set_a = pd.read_csv(cohort / "set_A.csv")
    set_a["sample"] = set_a["sample"].astype(str)
    set_a["unit_id"] = set_a["unit_id"].astype(str)

    main_meta = pd.read_csv(main_input / "meta.csv").drop_duplicates("sample", keep="first")
    main_meta["sample"] = main_meta["sample"].astype(str)
    ref_meta = main_meta[
        (main_meta["set"].astype(str) == "dev")
        & (main_meta["label"].astype(str) == "Normal")
    ].copy()
    filler_meta = main_meta[
        (main_meta["set"].astype(str) == "test")
        & (main_meta["label"].astype(str) == "Normal")
    ].copy()
    if len(ref_meta) < 96:
        raise click.ClickException(f"Expected >=96 dev Normal, found {len(ref_meta)}")

    ref_names = set(ref_meta["sample"])
    drop = {str(s) for s in DROP_SAMPLES}
    eval_units = set_a[~set_a["sample"].isin(ref_names) & ~set_a["sample"].isin(drop)].copy()

    main_ep = pd.read_parquet(main_input / "episcore_grid_search.parquet")
    main_z = pd.read_parquet(main_input / "zscore_grid_search.parquet")

    ep_rows = []
    z_rows = []
    meta_rows = []
    missing = []
    score_src = []
    for _, r in eval_units.iterrows():
        orig = str(r["sample"])
        uid = str(r["unit_id"])
        if not uid:
            missing.append(orig)
            continue
        ep = _read_unit_ep(ep_root / f"{uid}.episcore.tsv", uid, ep_thr, ep_rec)
        zdf = _read_unit_z(z_root / f"{uid}.percentage.tsv", uid, z_thr, z_rec)
        src = "unit"
        if ep is None or zdf is None:
            ep = _from_main(main_ep, orig, uid, ep_thr, ep_rec, _need_ep())
            zdf = _from_main(
                main_z, orig, uid, z_thr, z_rec,
                ["sample", "chr", "threshold", "recall", "percentage"],
            )
            src = "main_parquet"
        if ep is None or zdf is None:
            missing.append(orig)
            continue
        ep_rows.append(ep)
        z_rows.append(zdf)
        ff = r.get("ff_before_mq")
        pur = r.get("purity")
        meta_rows.append(
            {
                "sample": uid,
                "orig_sample": orig,
                "role": "eval",
                "set": str(r.get("set", "") or ""),
                "label": str(r.get("label", "") or ""),
                "ff_before_mq": float(ff) if pd.notna(ff) else float("nan"),
                "purity": float(pur) if pd.notna(pur) else float("nan"),
                "batch_key": str(r.get("score_batch_key_used") or r.get("batch_key") or ""),
                "unit_id": uid,
                "score_source": src,
            }
        )
        score_src.append(src)

    if not meta_rows:
        raise click.ClickException("No complete Set A eval units")

    eval_meta = pd.DataFrame(meta_rows)
    eval_ep = pd.concat(ep_rows, ignore_index=True)
    eval_z = pd.concat(z_rows, ignore_index=True)

    pool_names = set(ref_meta["sample"]) | set(filler_meta["sample"])
    pool_ep = main_ep[
        main_ep["sample"].astype(str).isin(pool_names)
        & (main_ep["threshold"] == ep_thr)
        & (main_ep["recall"] == ep_rec)
    ][_need_ep()].copy()
    pool_z = main_z[
        main_z["sample"].astype(str).isin(pool_names)
        & (main_z["threshold"] == z_thr)
        & (main_z["recall"] == z_rec)
    ][["sample", "chr", "threshold", "recall", "percentage"]].copy()

    def _tag(df: pd.DataFrame, role: str) -> pd.DataFrame:
        out_m = df.copy()
        out_m["orig_sample"] = out_m["sample"].astype(str)
        out_m["role"] = role
        out_m["unit_id"] = ""
        out_m["score_source"] = "main_parquet"
        if "batch_key" not in out_m.columns:
            out_m["batch_key"] = ""
        keep = [
            "sample",
            "orig_sample",
            "role",
            "set",
            "label",
            "ff_before_mq",
            "purity",
            "batch_key",
            "unit_id",
            "score_source",
        ]
        for c in keep:
            if c not in out_m.columns:
                out_m[c] = pd.NA
        out_m["ff_before_mq"] = pd.to_numeric(out_m["ff_before_mq"], errors="coerce")
        out_m["purity"] = pd.to_numeric(out_m["purity"], errors="coerce")
        return out_m[keep]

    merged_meta = pd.concat(
        [_tag(ref_meta, "ref_pool"), _tag(filler_meta, "filler_pool"), eval_meta],
        ignore_index=True,
        sort=False,
    )
    merged_ep = pd.concat([pool_ep, eval_ep], ignore_index=True)
    merged_z = pd.concat([pool_z, eval_z], ignore_index=True)

    merged_meta.to_csv(out / "meta.csv", index=False)
    merged_ep.to_parquet(out / "episcore_grid_search.parquet", index=False, compression="snappy")
    merged_z.to_parquet(out / "zscore_grid_search.parquet", index=False, compression="snappy")
    eval_meta.to_csv(out / "eval_samples.tsv", sep="\t", index=False)
    (out / "missing_units.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    n_src = pd.Series(score_src).value_counts().to_dict()
    summary = (
        f"mode={mode}\n"
        f"ep={ep_thr}/{ep_rec} z={z_thr}/{z_rec}\n"
        f"ref_pool={len(ref_meta)}\n"
        f"filler_pool={len(filler_meta)}\n"
        f"setA={len(set_a)}\n"
        f"setA_in_ref_pool={int(set_a['sample'].isin(ref_names).sum())}\n"
        f"dropped={','.join(DROP_SAMPLES)}\n"
        f"eval={len(eval_meta)}\n"
        f"eval_score_source={n_src}\n"
        f"missing={len(missing)} {missing}\n"
    )
    (out / "prepare_summary.txt").write_text(summary)
    console.print(summary)
    console.print(f"[green]OK[/green] {out} meta={len(merged_meta)}")


if __name__ == "__main__":
    main()
