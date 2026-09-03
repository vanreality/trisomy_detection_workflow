#!/usr/bin/env python3
"""Build Set A–D cohorts and a union unit samplesheet for batch-QC viz.

Set A: label Normal|T*, set∈{dev,test}, depth_qc=pass, preferred unit only
Set B: set=buffer, depth_qc=pass, preferred unit only
Set C: set=emergency, depth_qc=pass, preferred unit only
Set D: all multi-batch units with depth_qc=pass (every batch)

Writes under ``--output-dir``:
  set_A.csv … set_D.csv, viz_units.csv, cohort_summary.txt
"""

from __future__ import annotations

import re
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


def _is_t(label: object) -> bool:
    return bool(re.match(r"^T\d", str(label)))


def _bam_root(clean_bam: str) -> Path | None:
    p = Path(str(clean_bam))
    if "bwameth_results" not in p.parts:
        return None
    i = p.parts.index("bwameth_results")
    return Path(*p.parts[: i + 1])


def _resolve_artifacts(sample: str, batch_key: str, clean_bam: str, meta_row: pd.Series):
    root = _bam_root(clean_bam)
    ep = z085 = beta = None
    if str(meta_row.get("score_batch_key", "") or "") == batch_key:
        ep_file = meta_row.get("episcore_file")
        if pd.notna(ep_file) and Path(str(ep_file)).is_file():
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


def _expand_units(mq: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.drop_duplicates("sample", keep="first").set_index("sample")
    picked = []
    for (_, _), g in mq.groupby(["sample", "batch_key"], sort=False):
        picked.append(_pick_row(g))
    units = pd.DataFrame(picked).reset_index(drop=True)
    units["sample"] = units["sample"].astype(str)
    units["batch_key"] = units["batch_key"].astype(str)
    units["unit_id"] = units["sample"] + "__" + units["batch_key"]

    rows = []
    for _, r in units.iterrows():
        s = str(r["sample"])
        if s not in meta.index:
            continue
        m = meta.loc[s]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        ep, z085, beta, root = _resolve_artifacts(
            s, str(r["batch_key"]), str(r["clean_bam"]), m
        )
        ff = m.get("ff_before_mq")
        # meta FF is for preferred/score batch; keep only when batch matches
        if str(m.get("preferred_batch_key", "") or "") != str(r["batch_key"]):
            if str(m.get("score_batch_key", "") or "") != str(r["batch_key"]):
                ff = float("nan")
        rows.append(
            {
                "unit_id": r["unit_id"],
                "sample": s,
                "batch_key": str(r["batch_key"]),
                "clean_bam": r["clean_bam"],
                "deconv_res": r["deconv_res"],
                "is_single_end": bool(r.get("is_single_end", False)),
                "selected": bool(r.get("selected", False)),
                "n_batches": int(m.get("n_batches") or 1),
                "preferred_batch_key": str(m.get("preferred_batch_key") or ""),
                "is_preferred_batch": str(m.get("preferred_batch_key") or "")
                == str(r["batch_key"]),
                "label": str(m.get("label") or "Unknown"),
                "pred_label": str(m.get("pred_label") or ""),
                "set": str(m.get("set") or ""),
                "depth_qc": str(m.get("depth_qc") or ""),
                "purity": m.get("purity"),
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
    return pd.DataFrame(rows)


def _preferred(units: pd.DataFrame) -> pd.DataFrame:
    single = units["n_batches"] <= 1
    pref = units["is_preferred_batch"]
    # single-batch: keep one row per sample (already one batch)
    return units[single | pref].copy()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--mqres", default=DEFAULT_MQRES, type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=DEFAULT_META, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
def main(mqres: str, meta: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mq = pd.read_csv(mqres)
    meta_df = pd.read_csv(meta)
    meta_df["sample"] = meta_df["sample"].astype(str)
    units = _expand_units(mq, meta_df)
    units["purity"] = pd.to_numeric(units["purity"], errors="coerce")
    units["ff_before_mq"] = pd.to_numeric(units["ff_before_mq"], errors="coerce")

    pass_qc = units["depth_qc"].astype(str).eq("pass")
    pref = _preferred(units)
    pref_pass = pref[pref["depth_qc"].astype(str).eq("pass")].copy()

    lab_ok = pref_pass["label"].astype(str).eq("Normal") | pref_pass["label"].map(_is_t)
    set_a = pref_pass[lab_ok & pref_pass["set"].isin(["dev", "test"])].copy()
    set_b = pref_pass[pref_pass["set"].eq("buffer")].copy()
    set_c = pref_pass[pref_pass["set"].eq("emergency")].copy()
    set_d = units[pass_qc & (units["n_batches"] > 1)].copy()

    for name, df in [("A", set_a), ("B", set_b), ("C", set_c), ("D", set_d)]:
        df.to_csv(out / f"set_{name}.csv", index=False)
        console.print(f"Set {name}: {len(df)} rows, {df['sample'].nunique()} samples")

    viz = pd.concat([set_a, set_b, set_c, set_d], ignore_index=True)
    viz = viz.drop_duplicates("unit_id").sort_values(["sample", "batch_key"])
    viz.to_csv(out / "viz_units.csv", index=False)
    # Nextflow-style
    viz[["unit_id", "clean_bam", "deconv_res"]].rename(columns={"unit_id": "sample"}).to_csv(
        out / "nf_split_bam_samplesheet.csv", index=False
    )

    summary = (
        f"set_A={len(set_a)}\nset_B={len(set_b)}\nset_C={len(set_c)}\nset_D={len(set_d)}\n"
        f"viz_units={len(viz)}\n"
        f"viz_has_ep_wide={int(viz['has_ep_wide'].sum())}\n"
        f"viz_has_beta={int(viz['has_beta'].sum())}\n"
    )
    (out / "cohort_summary.txt").write_text(summary)
    console.print(summary)
    console.print(f"[green]OK[/green] Wrote {out}")


if __name__ == "__main__":
    main()
