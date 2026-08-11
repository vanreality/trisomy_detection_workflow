#!/usr/bin/env python3
"""Plot chrY–FF, conventional recall curves, and male-ref early recall curves."""

from __future__ import annotations

import click
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rich.console import Console

import config as cfg

console = Console()

LABEL_COLOR_MAP = {
    "female": "#9E9E9E",
    "male": "#1F77B4",
    "XO": "#E74C3C",
    "XO/XY嵌合": "#C0392B",
    "XXX": "#8E44AD",
    "XXY": "#E67E22",
    "XX/XY嵌合体": "#16A085",
    "69, XYY": "#2C3E50",
    "Normal": "#7F8C8D",
    "ambiguous": "#BDC3C7",
}


def _color_for(label: str) -> str:
    if label in LABEL_COLOR_MAP:
        return LABEL_COLOR_MAP[label]
    palette = px.colors.qualitative.Dark24
    return palette[hash(str(label)) % len(palette)]


def _scatter_chry(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label in sorted(df["label"].dropna().astype(str).unique(), key=str):
        color = _color_for(label)
        for cohort, symbol, size, suffix in (
            ("old_early", "circle", 10, "old_early"),
            ("new_early", "star", 15, "new_early"),
            ("middle", "diamond", 9, "middle"),
        ):
            sub = df[(df["label"].astype(str) == label) & (df["cohort"] == cohort)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["ff_before_mq"],
                    y=sub["chrY_ratio"],
                    mode="markers",
                    name=f"{label} ({suffix})",
                    legendgroup=str(label),
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=color,
                        line=dict(width=1, color="white"),
                        opacity=0.9,
                    ),
                    customdata=np.stack([sub["sample"].astype(str)], axis=-1),
                    hovertemplate=(
                        f"label={label}<br>cohort={cohort}<br>"
                        "ff=%{x}<br>chrY=%{y}<br>sample=%{customdata[0]}<extra></extra>"
                    ),
                )
            )

    early = df[df["dataset"] == "early"]
    all_x = early["ff_before_mq"].to_numpy(dtype=float)
    all_x = all_x[np.isfinite(all_x)]
    if all_x.size:
        x_left, x_right = float(all_x.min()), float(all_x.max())
        for gender, dash in (("male", "dot"), ("female", "dash")):
            sub = early[early["label"].astype(str) == gender].dropna(
                subset=["ff_before_mq", "chrY_ratio"]
            )
            if len(sub) < 2:
                continue
            xv = sub["ff_before_mq"].to_numpy(dtype=float)
            yv = sub["chrY_ratio"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(xv, yv, 1)
            xr = np.linspace(x_left, x_right, 100)
            fig.add_trace(
                go.Scatter(
                    x=xr,
                    y=slope * xr + intercept,
                    mode="lines",
                    line=dict(color=_color_for(gender), dash=dash, width=2),
                    name=f"{gender} fit (slope={slope:.3g})",
                )
            )
        fig.update_xaxes(range=[x_left, x_right], tickformat=".1%")

    fig.update_layout(
        title=dict(text="chrY ratio vs FF (○ old_early, ★ new_early, ◇ middle)", x=0.02),
        xaxis_title="ff_before_mq",
        yaxis_title="chrY_ratio",
        template="plotly_white",
        width=1000,
        height=520,
        plot_bgcolor="#FAFAFA",
    )
    return fig


def _curves_per_sample(
    df: pd.DataFrame,
    y: str,
    title: str,
    ylabel: str,
    solid_cohorts: set[str] | None = None,
) -> go.Figure:
    if solid_cohorts is None:
        solid_cohorts = {"old_early"}
    fig = go.Figure()
    legend_seen: set[str] = set()
    ordered = df.sort_values(["label", "cohort", "sample", "recall"], kind="mergesort")
    for sample, sub in ordered.groupby("sample", sort=False):
        sub = sub.sort_values("recall")
        if sub.empty or sub[y].isna().all():
            continue
        label = str(sub["label"].iloc[0])
        cohort = str(sub["cohort"].iloc[0])
        color = _color_for(label)
        dash = "solid" if cohort in solid_cohorts else "dot"
        legend_key = f"{label} ({cohort})"
        show = legend_key not in legend_seen
        if show:
            legend_seen.add(legend_key)
        ff = sub["ff_before_mq"].iloc[0] if "ff_before_mq" in sub.columns else None
        ff_txt = (
            "NA"
            if ff is None or (isinstance(ff, float) and np.isnan(ff))
            else f"{float(ff):.4%}"
        )
        fig.add_trace(
            go.Scatter(
                x=sub["recall"],
                y=sub[y],
                mode="lines",
                name=legend_key,
                legendgroup=legend_key,
                showlegend=show,
                line=dict(color=color, dash=dash, width=1.8),
                opacity=0.85,
                hovertemplate=(
                    f"sample={sample}<br>label={label}<br>cohort={cohort}<br>"
                    f"ff_before_mq={ff_txt}<br>"
                    "recall=%{x}<br>" + y + "=%{y}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.02),
        xaxis=dict(title="Recall", range=[0.005, 0.99]),
        yaxis_title=ylabel,
        template="plotly_white",
        width=1000,
        height=520,
        plot_bgcolor="#FAFAFA",
    )
    return fig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if cfg.CHRY_FF_TSV.is_file():
        chry = pd.read_csv(cfg.CHRY_FF_TSV, sep="\t").dropna(
            subset=["ff_before_mq", "chrY_ratio"]
        )
        fig = _scatter_chry(chry)
        out = cfg.PLOTS_DIR / "chry_ff.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        console.print(f"[green]Wrote[/green] {out}")

    for path, y, title, out_name in (
        (
            cfg.EPISCORE_COLLECTED,
            "chrX_episcore",
            "chrX episcore vs recall (conventional early female ref; solid=old, dotted=new)",
            "chrX_episcore_vs_recall.html",
        ),
        (
            cfg.ZSCORE_COLLECTED,
            "chrX_zscore",
            "chrX zscore vs recall (conventional early female ref; solid=old, dotted=new)",
            "chrX_zscore_vs_recall.html",
        ),
        (
            cfg.MALE_REF_EPISCORE_COLLECTED,
            "chrX_episcore",
            "male_ref chrX episcore vs recall (middle-male ref; early samples)",
            "male_ref_chrX_episcore_vs_recall.html",
        ),
        (
            cfg.MALE_REF_ZSCORE_COLLECTED,
            "chrX_zscore",
            "male_ref chrX zscore vs recall (middle-male ref; early samples)",
            "male_ref_chrX_zscore_vs_recall.html",
        ),
        (
            cfg.FEMALE_REF_EPISCORE_COLLECTED,
            "chrX_episcore",
            "female_ref chrX episcore vs recall (middle-female ref; early samples)",
            "female_ref_chrX_episcore_vs_recall.html",
        ),
        (
            cfg.FEMALE_REF_ZSCORE_COLLECTED,
            "chrX_zscore",
            "female_ref chrX zscore vs recall (middle-female ref; early samples)",
            "female_ref_chrX_zscore_vs_recall.html",
        ),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            console.print(f"[yellow]Skip plot[/yellow] missing {path}")
            continue
        df = pd.read_csv(path, sep="\t")
        if df.empty or y not in df.columns:
            console.print(f"[yellow]Skip plot[/yellow] empty {path}")
            continue
        # Only early on male/female ref plots
        if "male_ref" in out_name or "female_ref" in out_name:
            if "dataset" in df.columns:
                df = df[df["dataset"] != "middle"]
        fig = _curves_per_sample(df, y=y, title=title, ylabel=y)
        out = cfg.PLOTS_DIR / out_name
        fig.write_html(str(out), include_plotlyjs="cdn")
        console.print(f"[green]Wrote[/green] {out}")


if __name__ == "__main__":
    main()
