#!/usr/bin/env python3
"""Plot fixed-mode pool_size → AUC curve.

Also recomputes a **fair** ezscore AUC on the intersection eval cohort
(samples still evaluable at the largest pool size), because pool>96 moves
test Normals into the reference pool and otherwise shrinks n_normal.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import plotly.graph_objects as go
from rich.console import Console

from separation import separation_index

console = Console()

DEFAULT_BLACKLIST = (
    "PTAY0577P9S1",
    "PTAY0599P8S1",
    "PTAY0666P7S1",
    "PTAY0682P7S1",
    "PTAY0689P8H1",
)


def _add_curve(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    y: str,
    name: str,
    color: str,
    dash: str = "solid",
) -> None:
    fig.add_trace(
        go.Scatter(
            x=df["pool_size"],
            y=df[y],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=7),
            hovertemplate="pool=%{x}<br>AUC=%{y:.4f}<extra>" + name + "</extra>",
        )
    )


def _fair_auc_table(
    mode_dir: Path,
    ff_min: float,
    blacklist: set[str] | None = None,
) -> pd.DataFrame | None:
    """Recompute ez AUC on intersection samples = eval set of max pool_size."""
    bl = blacklist or set()
    pool_dirs = sorted(
        (p for p in mode_dir.glob("pool_*") if p.is_dir() and p.name.split("_")[1].isdigit()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not pool_dirs:
        return None
    largest = pool_dirs[-1]
    core_path = largest / "abnormality_signal_ratio.tsv"
    if not core_path.is_file():
        return None
    core = pd.read_csv(core_path, sep="\t")
    core_samples = set(core.loc[~core["sample"].astype(str).isin(bl), "sample"].astype(str))
    rows = []
    for pdir in pool_dirs:
        tsv = pdir / "abnormality_signal_ratio.tsv"
        if not tsv.is_file():
            continue
        df = pd.read_csv(tsv, sep="\t")
        sub = df[
            df["sample"].astype(str).isin(core_samples)
            & ~df["sample"].astype(str).isin(bl)
        ].copy()
        sep = separation_index(sub, "ezscore_signal_ratio", ff_min=ff_min)
        pool_size = int(pdir.name.split("_")[1])
        rows.append(
            {
                "pool_size": pool_size,
                "auc_ezscore_fair": sep["sep_auc"],
                "n_normal_fair": sep["n_normal"],
                "n_trisomy_fair": sep["n_trisomy"],
                "core_pool_size": int(largest.name.split("_")[1]),
            }
        )
    return pd.DataFrame(rows).sort_values("pool_size") if rows else None


def _merge_pool_rows(mode_dir: Path) -> pd.DataFrame:
    """Merge per-pool row TSVs written by SLURM array workers."""
    frames = []
    for path in sorted(mode_dir.glob("pool_*/pool_size_auc_row.tsv")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(path, sep="\t"))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]skip[/yellow] {path}: {exc}")
    if not frames:
        raise click.ClickException(f"No pool_*/pool_size_auc_row.tsv under {mode_dir}")
    summary = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("pool_size", keep="last")
        .sort_values("pool_size")
    )
    out = mode_dir / "pool_size_auc.tsv"
    summary.to_csv(out, sep="\t", index=False, float_format="%.6f")
    console.print(f"[green]OK[/green] merged {len(summary)} pool rows -> {out}")
    return summary


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--sweep-base", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", default=None, type=click.Path(path_type=Path))
@click.option("--ff-min", default=0.01, show_default=True, type=float)
@click.option(
    "--blacklist",
    default=",".join(DEFAULT_BLACKLIST),
    show_default=True,
    help="Comma-separated samples excluded from fair AUC",
)
def main(sweep_base: Path, output_dir: Path | None, ff_min: float, blacklist: str) -> None:
    out = output_dir or (sweep_base / "plots")
    out.mkdir(parents=True, exist_ok=True)
    bl = {s.strip() for s in blacklist.split(",") if s.strip()}

    mode_dir = sweep_base / "fixed"
    path = mode_dir / "pool_size_auc.tsv"
    # Prefer fresh merge from per-pool rows (array-safe).
    if any(mode_dir.glob("pool_*/pool_size_auc_row.tsv")):
        df = _merge_pool_rows(mode_dir)
    elif path.is_file():
        df = pd.read_csv(path, sep="\t").sort_values("pool_size")
    else:
        raise click.ClickException(f"Missing pool rows / {path}")
    fair = _fair_auc_table(sweep_base / "fixed", ff_min, blacklist=bl)
    if fair is not None:
        df = df.merge(fair, on="pool_size", how="left")
        console.print(
            f"  fair core N={int(df['n_normal_fair'].iloc[-1])} "
            f"T={int(df['n_trisomy_fair'].iloc[-1])}"
        )

    fig = go.Figure()
    color = "rgb(214,39,40)"
    _add_curve(fig, df, y="auc_ezscore", name="ez (raw eval)", color=color)
    if fair is not None and "auc_ezscore_fair" in df.columns:
        _add_curve(
            fig,
            df,
            y="auc_ezscore_fair",
            name="ez (fair/core eval)",
            color=color,
            dash="dash",
        )
    df.to_csv(out / "pool_size_auc_fixed.tsv", sep="\t", index=False)

    fig.update_layout(
        title=(
            "Fixed-combo pool size vs ezscore AUC"
            "<br><sup>solid=raw eval (n_normal shrinks if pool&gt;96); "
            "dashed=fair AUC on largest-pool eval cohort; ff≥1%; step=2</sup>"
        ),
        xaxis_title="pool_size (= 2 × ref_n)",
        yaxis_title="ROC-AUC (ezscore signal ratio)",
        template="plotly_white",
        height=520,
        width=920,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(t=90, b=90),
    )
    fig.update_yaxes(range=[0.9, 1.005])
    html = out / "pool_size_auc.html"
    fig.write_html(str(html), include_plotlyjs="cdn", full_html=True)
    console.print(f"[green]OK[/green] {html}")


if __name__ == "__main__":
    main()
