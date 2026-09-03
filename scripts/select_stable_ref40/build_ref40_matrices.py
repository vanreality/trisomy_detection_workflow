#!/usr/bin/env python3
"""Build ref40_*matrix* reference files for a selected stable ref_40.

Writes (under --output-dir):
  ref40_episcore_matrix.tsv   — wide beta/z_intra matrix (early_reference format)
  ref40_zscore_matrix.csv     — long percentage reference
  ref40_ezscore_matrix.csv    — chr mu/sigma of (episcore+zscore) over fixed ez refs
  ref40_reference_samples.tsv
  chr_stats_reference_samples.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()
CHR_LIST = [f"chr{i}" for i in range(1, 23)]
DEFAULT_TEMPLATE = Path(
    "/lustre1/cqyi/AIPT_2.0/workflow/episcore/assets/early_reference_beta_zscore.tsv"
)
DEFAULT_EZ = Path(
    "/lustre1/cqyi/myli/bert/analysis_nipt/multiomics/chr_stats_reference_samples.txt"
)
DEFAULT_EPISCORE_SS = Path(
    "/lustre1/cqyi/syfan/nipt_article_plot/episcore_result_samplesheet.csv"
)


def _load_ez_samples(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "HCPT" in s and len(s) > 8:
            s = s[:8]
        out.append(s)
    return out


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Dir with ref40_samples.txt, beta.csv, percentage.csv, meta.csv, ref40_score.tsv",
)
@click.option(
    "--template-episcore",
    default=str(DEFAULT_TEMPLATE),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--ezscore-ref-samples",
    default=str(DEFAULT_EZ),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--episcore-samplesheet",
    default=str(DEFAULT_EPISCORE_SS),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
def main(
    output_dir: str,
    template_episcore: str,
    ezscore_ref_samples: str,
    episcore_samplesheet: str,
) -> None:
    """Build ref40 episcore / zscore / ezscore reference matrices."""
    out = Path(output_dir)
    ref40 = [
        s.strip()
        for s in (out / "ref40_samples.txt").read_text().splitlines()
        if s.strip()
    ]
    if len(ref40) != 40:
        raise click.ClickException(f"Expected 40 ref samples, got {len(ref40)}")

    meta = pd.read_csv(out / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    beta = pd.read_csv(out / "beta.csv")
    beta["sample"] = beta["sample"].astype(str)
    pct = pd.read_csv(out / "percentage.csv", sep="\t")
    pct["sample"] = pct["sample"].astype(str)
    epi_ss = pd.read_csv(episcore_samplesheet)
    epi_ss["sample"] = epi_ss["sample"].astype(str)
    path_col = "episcore_file" if "episcore_file" in epi_ss.columns else None
    path_map = (
        dict(zip(epi_ss["sample"], epi_ss[path_col].astype(str))) if path_col else {}
    )

    # --- 1) Episcore wide matrix ---
    tmpl_cols = list(pd.read_csv(template_episcore, sep="\t", nrows=0).columns)
    metric_blocks = [
        ("hypo_beta", True),
        ("hyper_beta", True),
        ("hypo_z_intra", True),
        ("hyper_z_intra", True),
        ("s_intra", True),
    ]
    count_suffixes = ["hypo_cpgs_count", "hyper_cpgs_count"]
    rows = []
    missing_beta = []
    for s in ref40:
        m = meta.loc[meta["sample"] == s]
        b = beta.loc[beta["sample"] == s]
        if m.empty or b.empty:
            missing_beta.append(s)
            continue
        m = m.iloc[0]
        b = b.iloc[0]
        row = {
            "sample": s,
            "label": m.get("label", "Normal"),
            "week": m.get("week", np.nan),
            "ff_before_mq": m.get("ff_before_mq", np.nan),
            "ff_after_mq": m.get("ff_after_mq", np.nan),
            "beta_path": path_map.get(s, ""),
        }
        for suffix, has_label in metric_blocks:
            if has_label:
                row[f"label_{suffix}"] = "Normal"
            for c in CHR_LIST:
                col = f"{c}_{suffix}"
                row[col] = b[col] if col in b.index else np.nan
        for suffix in count_suffixes:
            for c in CHR_LIST:
                col = f"{c}_{suffix}"
                row[col] = b[col] if col in b.index else np.nan
        rows.append(row)

    ep_ref = pd.DataFrame(rows)
    for c in tmpl_cols:
        if c not in ep_ref.columns:
            ep_ref[c] = np.nan
    ep_ref = ep_ref[tmpl_cols]
    ep_out = out / "ref40_episcore_matrix.tsv"
    ep_ref.to_csv(ep_out, sep="\t", index=False, float_format="%.6f")
    console.print(f"[green]OK[/green] {ep_out.name}  rows={len(ep_ref)} cols={len(ep_ref.columns)}")
    if missing_beta:
        console.print(f"[yellow]MISSING beta:[/yellow] {missing_beta}")

    # --- 2) Zscore long matrix ---
    count_col = "count" if "count" in pct.columns else "readscount"
    pct_ref = pct[pct["sample"].isin(ref40) & pct["chr"].isin(CHR_LIST)].copy()
    missing_pct = sorted(set(ref40) - set(pct_ref["sample"].unique()))
    z_rows = []
    for s, g in pct_ref.groupby("sample", sort=False):
        g = g.drop_duplicates("chr", keep="first")
        counts = g.set_index("chr").reindex(CHR_LIST)[count_col].astype(float)
        sum_auto = float(counts.sum())
        mean_auto = float(counts.mean())
        std_auto = float(counts.std(ddof=1)) if counts.notna().sum() > 1 else 1.0
        if std_auto == 0 or not np.isfinite(std_auto):
            std_auto = 1.0
        for c in CHR_LIST:
            rc = float(counts.loc[c]) if pd.notna(counts.loc[c]) else np.nan
            percentage = (rc / sum_auto) if sum_auto > 0 and np.isfinite(rc) else np.nan
            adj = ((rc - mean_auto) / std_auto) if np.isfinite(rc) else np.nan
            z_rows.append(
                {
                    "sample": s,
                    "gender": "unknown",
                    "chr": c,
                    "readscount": int(rc) if np.isfinite(rc) else np.nan,
                    "percentage": percentage,
                    "adj_percentage": adj,
                }
            )
    z_ref = pd.DataFrame(z_rows)
    z_out = out / "ref40_zscore_matrix.csv"
    z_ref.to_csv(z_out, index=False)
    console.print(
        f"[green]OK[/green] {z_out.name}  rows={len(z_ref)} samples={z_ref['sample'].nunique()}"
    )
    if missing_pct:
        console.print(f"[yellow]MISSING percentage:[/yellow] {missing_pct}")
    sums = z_ref.groupby("sample")["percentage"].sum()
    console.print(f"  percentage sum range: {sums.min():.6f} .. {sums.max():.6f}")

    # --- 3) EZscore chr stats (fixed 25 ez refs, scores from ref40_score.tsv) ---
    ez_samples = _load_ez_samples(Path(ezscore_ref_samples))
    scores = pd.read_csv(out / "ref40_score.tsv", sep="\t")
    scores["sample"] = scores["sample"].astype(str)
    ez_df = scores[scores["sample"].isin(ez_samples)].copy()
    found = set(ez_df["sample"])
    missing_ez = [s for s in ez_samples if s not in found]
    console.print(f"Ezscore refs found: {len(found)} / {len(ez_samples)}")
    if missing_ez:
        console.print(f"[yellow]MISSING ezscore refs:[/yellow] {missing_ez}")

    chr_rows = []
    for c in CHR_LIST:
        num = c.removeprefix("chr")
        ep = ez_df[f"episcore_chr{num}"].to_numpy(dtype=float)
        zs = ez_df[f"zscore_chr{num}"].to_numpy(dtype=float)
        combined = ep + zs
        chr_rows.append(
            {
                "chr": c,
                "mu": float(np.nanmean(combined)),
                "sigma": float(np.nanstd(combined, ddof=0)),
                "count": int(np.isfinite(combined).sum()),
            }
        )
    ez_stats = pd.DataFrame(chr_rows)
    ez_out = out / "ref40_ezscore_matrix.csv"
    ez_stats.to_csv(ez_out, index=False)
    console.print(f"[green]OK[/green] {ez_out.name}")
    console.print(ez_stats.head(3).to_string(index=False))

    pd.DataFrame({"sample": ref40}).to_csv(
        out / "ref40_reference_samples.tsv", sep="\t", index=False
    )
    (out / "chr_stats_reference_samples.txt").write_text("\n".join(ez_samples) + "\n")
    console.print("[green]OK[/green] Wrote sample lists.")


if __name__ == "__main__":
    try:
        main(standalone_mode=False)
    except click.ClickException as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
