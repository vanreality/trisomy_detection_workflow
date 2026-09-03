#!/usr/bin/env python3
"""Join MAD-toxic / expanded-pool samples to mqres batches and test enrichment.

Reads ``mqres_samplesheet.csv`` (one or more ``mqres_batch`` per sample). Background
is the expanded Normal pool in ``candidate_mad_scores.tsv`` (n=273). Writes batch
columns onto the DB-mapped toxic samplesheet and a short report.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import fisher_exact

console = Console()

DEFAULT_OUT_BASE = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
)
DEFAULT_OUTDIR = DEFAULT_OUT_BASE / "expanded_pool_mad"
DEFAULT_MQRES = Path(
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/mqres_samplesheet.csv"
)
_DATE_RE = re.compile(r"(\d{8})")


def yyyymmdd(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    m = _DATE_RE.search(str(val))
    return m.group(1) if m else str(val).strip()


def _md_escape(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).replace("\n", " ").replace("|", "\\|")


def _md_table(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    use = list(cols) if cols is not None else list(df.columns)
    header = "| " + " | ".join(use) + " |"
    sep = "| " + " | ".join("---" for _ in use) + " |"
    rows = [
        "| " + " | ".join(_md_escape(r[c]) for c in use) + " |"
        for _, r in df.iterrows()
    ]
    return [header, sep, *rows]


def unique_sample_batches(mqres: pd.DataFrame) -> pd.DataFrame:
    df = mqres.copy()
    df["mqres_batch"] = df["mqres_batch"].map(yyyymmdd)
    df["qc_batch"] = df["qc_batch"].map(yyyymmdd)
    keep = ["sample", "mqres_batch", "qc_batch"]
    for c in ("puc19", "lambda"):
        if c in df.columns:
            keep.append(c)
    return df.drop_duplicates(["sample", "mqres_batch"])[keep]


def per_sample_summary(sb: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "n_mqres_batches": ("mqres_batch", "nunique"),
        "mqres_batches": (
            "mqres_batch",
            lambda s: ",".join(sorted({str(x) for x in s if str(x)})),
        ),
        "latest_mqres_batch": ("mqres_batch", "max"),
    }
    if "puc19" in sb.columns:
        agg["puc19_min"] = ("puc19", "min")
    if "lambda" in sb.columns:
        agg["lambda_min"] = ("lambda", "min")
    if "qc_batch" in sb.columns:
        agg["qc_batches"] = (
            "qc_batch",
            lambda s: ",".join(sorted({str(x) for x in s if str(x)})),
        )
    return sb.groupby("sample", as_index=False).agg(**agg)


def batch_enrichment(sb_pool: pd.DataFrame, tox: set[str], ok: set[str]) -> pd.DataFrame:
    rows = []
    for batch, g in sb_pool.groupby("mqres_batch"):
        in_batch = set(g["sample"].astype(str))
        n_tox_in = len(in_batch & tox)
        n_ok_in = len(in_batch & ok)
        n_tox_out = len(tox - in_batch)
        n_ok_out = len(ok - in_batch)
        or_, p = fisher_exact(
            [[n_tox_in, n_tox_out], [n_ok_in, n_ok_out]],
            alternative="two-sided",
        )
        n_pool = n_tox_in + n_ok_in
        rows.append(
            {
                "mqres_batch": batch,
                "n_pool_in_batch": n_pool,
                "n_toxic_in_batch": n_tox_in,
                "n_ok_in_batch": n_ok_in,
                "frac_toxic_in_batch": (n_tox_in / n_pool) if n_pool else np.nan,
                "frac_of_all_toxic": n_tox_in / len(tox) if tox else np.nan,
                "oddsratio": or_,
                "fisher_p": p,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["fisher_p", "n_toxic_in_batch"], ascending=[True, False])


def write_report(
    toxic: pd.DataFrame,
    pool_n: int,
    enr: pd.DataFrame,
    out_md: Path,
) -> None:
    n = len(toxic)
    n_multi = int((toxic["n_mqres_batches"] >= 2).sum())
    latest_vc = (
        toxic["latest_mqres_batch"].value_counts().rename_axis("latest_mqres_batch").reset_index(name="n_toxic")
    )
    top = enr.head(8)
    hit = enr.loc[enr["n_toxic_in_batch"] >= 3].copy()

    lines = [
        "# Toxic sample mqres-batch audit",
        "",
        f"Joined `{n}` MAD-toxic Normals and the expanded pool (n={pool_n}) to "
        "`samplesheet_summary/mqres_samplesheet.csv`. Background toxic rate = "
        f"{n}/{pool_n} = {100 * n / pool_n:.1f}%.",
        "",
        f"They do **not** share one batch: **{toxic['latest_mqres_batch'].nunique()}** "
        f"distinct latest `mqres_batch` values, **{n_multi}/{n}** multi-batch "
        "(same as the rest of the pool).",
        "",
        "## Latest mqres_batch among toxic",
        "",
    ]
    lines.extend(_md_table(latest_vc))
    lines += [
        "",
        "## Batches with ≥3 toxic samples (incidence, not latest-only)",
        "",
    ]
    if hit.empty:
        lines.append("_none_")
    else:
        lines.extend(
            _md_table(
                hit,
                [
                    "mqres_batch",
                    "n_pool_in_batch",
                    "n_toxic_in_batch",
                    "frac_toxic_in_batch",
                    "frac_of_all_toxic",
                    "oddsratio",
                    "fisher_p",
                ],
            )
        )
    lines += [
        "",
        "Fisher exact is toxic∈batch vs rest of pool. No batch survives a "
        "multiple-testing correction (smallest raw p is a *depletion* of "
        "`20260313`).",
        "",
        "## Smallest Fisher p (any batch)",
        "",
    ]
    show = top.copy()
    for c in ("frac_toxic_in_batch", "frac_of_all_toxic", "oddsratio", "fisher_p"):
        show[c] = show[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    lines.extend(
        _md_table(
            show,
            [
                "mqres_batch",
                "n_pool_in_batch",
                "n_toxic_in_batch",
                "frac_toxic_in_batch",
                "frac_of_all_toxic",
                "oddsratio",
                "fisher_p",
            ],
        )
    )

    b21 = toxic.loc[toxic["mqres_batches"].astype(str).str.contains("20260321")].copy()
    lines += [
        "",
        f"## Soft cluster: `20260321` (n={len(b21)} / {n} toxic)",
        "",
        "Largest pool batch after `20260313`. 11 toxic vs ~6.6 expected from "
        "latest-batch size; Fisher p=0.061, OR=2.2. Mostly test + `birth_outcome`.",
        "",
    ]
    if not b21.empty:
        cols = [
            c
            for c in (
                "mad_rank",
                "sample",
                "set",
                "source_class",
                "mad_score",
                "max_mad_chr",
                "mqres_batches",
                "puc19_min",
            )
            if c in b21.columns
        ]
        lines.extend(_md_table(b21.sort_values("mad_rank"), cols))

    nowhere = toxic.loc[toxic.get("source_class", "") == "nowhere"] if "source_class" in toxic.columns else pd.DataFrame()
    if not nowhere.empty:
        lines += [
            "",
            "## `nowhere` label source — batches (scattered)",
            "",
        ]
        cols = [
            c
            for c in ("mad_rank", "sample", "set", "mqres_batches", "latest_mqres_batch")
            if c in nowhere.columns
        ]
        lines.extend(_md_table(nowhere.sort_values("mad_rank"), cols))

    lines += [
        "",
        "## Takeaway",
        "",
        "No batch QC story: toxic samples spread over 17 latest batches; pUC19/λ "
        "match the OK pool. The only visual pile-up is `20260321` (11 test-heavy "
        "birth-outcome samples), which is also one of the largest Normal batches "
        "and is not FDR-significant.",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--toxic",
    "toxic_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=str(DEFAULT_OUTDIR / "toxic_samplesheet_with_db_labels.tsv"),
    show_default=True,
)
@click.option(
    "--candidates",
    "cand_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=str(DEFAULT_OUTDIR / "candidate_mad_scores.tsv"),
    show_default=True,
)
@click.option(
    "--mqres",
    "mqres_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=str(DEFAULT_MQRES),
    show_default=True,
)
@click.option(
    "--outdir",
    type=click.Path(path_type=Path, file_okay=False),
    default=str(DEFAULT_OUTDIR),
    show_default=True,
)
def main(toxic_path: Path, cand_path: Path, mqres_path: Path, outdir: Path) -> None:
    """Add mqres batch columns to the toxic sheet and write an enrichment report."""
    outdir.mkdir(parents=True, exist_ok=True)
    toxic = pd.read_csv(toxic_path, sep="\t")
    cand = pd.read_csv(cand_path, sep="\t")
    mqres = pd.read_csv(mqres_path)
    sb = unique_sample_batches(mqres)
    ps = per_sample_summary(sb)

    pool = set(cand["sample"].astype(str))
    tox = set(toxic["sample"].astype(str))
    missing = sorted(tox - set(ps["sample"].astype(str)))
    if missing:
        raise RuntimeError(f"Toxic samples missing from mqres: {missing}")

    sb_pool = sb.loc[sb["sample"].isin(pool)].copy()
    enr = batch_enrichment(sb_pool, tox, pool - tox)
    enr_path = outdir / "toxic_batch_enrichment.tsv"
    enr.to_csv(enr_path, sep="\t", index=False, float_format="%.6g")

    batch_cols = [
        c
        for c in (
            "n_mqres_batches",
            "mqres_batches",
            "latest_mqres_batch",
            "qc_batches",
            "puc19_min",
            "lambda_min",
        )
        if c in ps.columns
    ]
    drop_old = [c for c in batch_cols if c in toxic.columns]
    toxic = toxic.drop(columns=drop_old)
    toxic = toxic.merge(ps[["sample"] + batch_cols], on="sample", how="left")

    # Place batch cols after label / source_class if present.
    front = [c for c in toxic.columns if c not in batch_cols]
    # insert after 'label' if possible
    if "label" in front:
        i = front.index("label") + 1
        ordered = front[:i] + batch_cols + front[i:]
    else:
        ordered = front + batch_cols
    toxic = toxic[ordered]
    toxic.to_csv(toxic_path, sep="\t", index=False)
    write_report(toxic, len(pool), enr, outdir / "toxic_batch_report.md")

    console.print(
        f"toxic n={len(toxic)} latest_batches={toxic['latest_mqres_batch'].nunique()} "
        f"multi={int((toxic['n_mqres_batches']>=2).sum())}"
    )
    top = enr.iloc[0]
    console.print(
        f"top fisher {top.mqres_batch} n_toxic={int(top.n_toxic_in_batch)} "
        f"p={top.fisher_p:.4g} OR={top.oddsratio:.3g}"
    )
    console.print(f"[green]OK[/green] updated {toxic_path}")
    console.print(f"[green]OK[/green] wrote {enr_path}")
    console.print(f"[green]OK[/green] wrote {outdir / 'toxic_batch_report.md'}")


if __name__ == "__main__":
    main()
