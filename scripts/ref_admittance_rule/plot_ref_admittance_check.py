#!/usr/bin/env python3
"""Plots for the independent test_ref_candidates admittance check.

  * FP+FN density (stacked FP/FN share) for all_96 / toxic_16_excl / random_16_excl
  * Per-chr boxplots of ez-ref mean and ez-ref SD across repeats
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

from common import (
    CHR_LIST,
    FN_COLOR,
    FP_COLOR,
    PERFECT_COLOR,
    density_table,
    fp_fn_summary,
    load_repeat_shards,
)
from plot_admittance import plot_fp_fn_density

console = Console()

POOL_COLORS = {
    "all_96_test": "#6C757D",
    "toxic_16_excluded": "#2A9D8F",
    "random_16_excluded": "#F18F01",
    "mad_pct_excluded": "#264653",
    "random_pct_excluded": "#E9C46A",
    "mad_keep80_excluded": "#1D3557",
    "random_mad_keep80_excluded": "#E9C46A",
    "mad_union_excluded": "#457B9D",
    "random_union_excluded": "#E76F51",
}
POOL_LABELS = {
    "all_96_test": "all 96 test refs",
    "toxic_16_excluded": "toxic 16 excluded (circular lift)",
    "random_16_excluded": "random 16 excluded",
    "mad_pct_excluded": "pct MAD-z>3.5 vs 96-dev excluded",
    "random_pct_excluded": "random drop (matched n, pct MAD)",
    "mad_keep80_excluded": "top-16 MAD-z vs 96-dev excluded",
    "random_mad_keep80_excluded": "random 16 excluded (MAD-keep80 control)",
    "mad_union_excluded": "MAD or FF vs 96-dev excluded",
    "random_union_excluded": "random drop (matched n, MAD∪FF)",
}
DEFAULT_TAGS = ("all_96_test", "toxic_16_excluded", "random_16_excluded")


def _write_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    console.print(f"  wrote {path}")


def plot_density_overlay(loaded: dict[str, dict], out: Path) -> pd.DataFrame:
    rows = []
    fig = make_subplots(
        rows=len(loaded),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[POOL_LABELS.get(k, k) for k in loaded],
    )
    overlay = go.Figure()
    for i, (name, data) in enumerate(loaded.items(), start=1):
        dens = density_table(data["fp"], data["fn"], data["fp_plus_fn"])
        s = fp_fn_summary(data["fp"], data["fn"], data["fp_plus_fn"])
        s["pool"] = name
        rows.append(s)
        custom = np.column_stack(
            [dens["n_repeats"], dens["mean_fp"], dens["mean_fn"], dens["density"]]
        )
        hover = (
            "FP+FN=%{x}<br>density=%{customdata[3]:.4f}<br>n=%{customdata[0]}"
            "<br>mean FP=%{customdata[1]:.2f} · mean FN=%{customdata[2]:.2f}<extra></extra>"
        )
        fig.add_trace(
            go.Bar(
                x=dens["fp_plus_fn"],
                y=dens["perfect_density"],
                name="perfect",
                marker_color=PERFECT_COLOR,
                hovertemplate=hover,
                customdata=custom,
                showlegend=(i == 1),
                legendgroup="perfect",
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=dens["fp_plus_fn"],
                y=dens["fp_density"],
                name="FP share",
                marker_color=FP_COLOR,
                hovertemplate=hover,
                customdata=custom,
                showlegend=(i == 1),
                legendgroup="fp",
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=dens["fp_plus_fn"],
                y=dens["fn_density"],
                name="FN share",
                marker_color=FN_COLOR,
                hovertemplate=hover,
                customdata=custom,
                showlegend=(i == 1),
                legendgroup="fn",
            ),
            row=i,
            col=1,
        )
        overlay.add_trace(
            go.Scatter(
                x=dens["fp_plus_fn"],
                y=dens["density"],
                mode="lines+markers",
                name=f"{POOL_LABELS.get(name, name)} (perfect={s['frac_perfect']:.3f})",
                line=dict(color=POOL_COLORS.get(name, "#333")),
            )
        )
        plot_fp_fn_density(
            dens,
            f"{POOL_LABELS.get(name, name)}  |  n={s['n_repeats']}  "
            f"perfect={s['frac_perfect']:.3f}  mean FP+FN={s['mean_fp_plus_fn']:.3f}",
            out / f"fp_fn_density_{name}.html",
        )
    fig.update_layout(
        title="FP+FN density on the same eval (test_ref_candidates excluded from eval)",
        barmode="stack",
        template="plotly_white",
        height=380 * max(len(loaded), 1),
        width=900,
        legend=dict(orientation="h", y=-0.04),
    )
    fig.update_xaxes(title_text="FP + FN", row=len(loaded), col=1)
    fig.update_yaxes(title_text="repeat density")
    _write_html(fig, out / "fp_fn_density_compare.html")
    overlay.update_layout(
        title="FP+FN density overlay (same eval)",
        xaxis_title="FP + FN",
        yaxis_title="repeat density",
        template="plotly_white",
        height=520,
        width=880,
    )
    _write_html(overlay, out / "fp_fn_density_overlay.html")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "fp_fn_compare_summary.tsv", sep="\t", index=False, float_format="%.6f")
    return summary


def _chr_box_html(
    loaded: dict[str, dict],
    key: str,
    ylabel: str,
    title: str,
    dest: Path,
) -> None:
    fig = make_subplots(
        rows=4,
        cols=6,
        subplot_titles=list(CHR_LIST),
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )
    for i, chr_name in enumerate(CHR_LIST):
        r, c = divmod(i, 6)
        for name, data in loaded.items():
            arr = data[key][:, i]
            qq = np.nanpercentile(arr, [25, 50, 75])
            iqr = qq[2] - qq[0]
            fig.add_trace(
                go.Box(
                    x=[POOL_LABELS.get(name, name)],
                    q1=[float(qq[0])],
                    median=[float(qq[1])],
                    q3=[float(qq[2])],
                    lowerfence=[float(max(np.nanmin(arr), qq[0] - 1.5 * iqr))],
                    upperfence=[float(min(np.nanmax(arr), qq[2] + 1.5 * iqr))],
                    name=POOL_LABELS.get(name, name),
                    legendgroup=name,
                    showlegend=(i == 0),
                    boxpoints=False,
                    marker_color=POOL_COLORS.get(name, "#333"),
                    line=dict(color=POOL_COLORS.get(name, "#333"), width=1),
                ),
                row=r + 1,
                col=c + 1,
            )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1100,
        width=1400,
        boxmode="group",
        legend=dict(orientation="h", y=-0.02),
        margin=dict(t=80, b=60),
    )
    fig.update_yaxes(title_text=ylabel, col=1)
    _write_html(fig, dest)


def _chr_box_png(
    loaded: dict[str, dict],
    key: str,
    ylabel: str,
    title: str,
    dest: Path,
) -> None:
    names = list(loaded)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.ravel()
    for i, chr_name in enumerate(CHR_LIST):
        ax = axes[i]
        data = [loaded[n][key][:, i] for n in names]
        bp = ax.boxplot(
            data,
            labels=[POOL_LABELS.get(n, n).replace(" ", "\n") for n in names],
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="#111", linewidth=1.1),
        )
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(POOL_COLORS.get(n, "#ccc"))
            patch.set_alpha(0.75)
        ax.set_title(chr_name, fontsize=10)
        ax.tick_params(axis="x", labelsize=6)
        if i % 6 == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    for j in range(len(CHR_LIST), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    console.print(f"  wrote {dest}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--check-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option(
    "--tags",
    default=",".join(DEFAULT_TAGS),
    show_default=True,
    help="Comma-separated score-dir names under check-dir.",
)
def main(check_dir: str, output_dir: str | None, tags: str) -> None:
    root = Path(check_dir)
    out = Path(output_dir) if output_dir else root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    tags = tuple(t.strip() for t in tags.split(",") if t.strip())
    loaded = {}
    for tag in tags:
        d = root / tag
        if not d.is_dir() or not any(d.glob("repeats_*.npz")):
            console.print(f"[yellow]skip missing[/yellow] {d}")
            continue
        loaded[tag] = load_repeat_shards(d)
        console.print(
            f"  {tag}: n={loaded[tag]['fp_plus_fn'].size} "
            f"perfect={(loaded[tag]['fp_plus_fn'] == 0).mean():.3f} "
            f"mean FP+FN={loaded[tag]['fp_plus_fn'].mean():.3f}"
        )
    if len(loaded) < 2:
        raise click.ClickException(f"Need ≥2 scored tags under {root}, found {list(loaded)}")

    plot_density_overlay(loaded, out)
    _chr_box_html(
        loaded,
        "ez_mu",
        "ez-ref mean",
        "Ez-ref mean of (episcore+zscore) by chromosome — same eval",
        out / "ez_mu_by_chr_compare.html",
    )
    _chr_box_html(
        loaded,
        "ez_sd",
        "ez-ref SD",
        "Ez-ref SD of (episcore+zscore) by chromosome — same eval",
        out / "ez_sd_by_chr_compare.html",
    )
    _chr_box_png(
        loaded,
        "ez_mu",
        "ez-ref mean",
        "Ez-ref mean of (episcore+zscore) by chromosome — same eval",
        out / "ez_mu_by_chr_compare.png",
    )
    _chr_box_png(
        loaded,
        "ez_sd",
        "ez-ref SD",
        "Ez-ref SD of (episcore+zscore) by chromosome — same eval",
        out / "ez_sd_by_chr_compare.png",
    )
    console.print(f"[green]done[/green] {out}")


if __name__ == "__main__":
    main()
