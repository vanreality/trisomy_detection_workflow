#!/usr/bin/env python3
"""FP+FN densities for MAD-rank drop-16 on the 96-dev and 96-test pools.

Two rows (dev / test) × three conditions (all 96, MAD top-16 excluded, random 16).
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

from common import FN_COLOR, FP_COLOR, PERFECT_COLOR, density_table, fp_fn_summary, load_repeat_shards
from plot_admittance import plot_fp_fn_density

console = Console()

COND_COLORS = {
    "all": "#6C757D",
    "mad16": "#264653",
    "random16": "#F18F01",
}
COND_LABELS = {
    "all": "all 96",
    "mad16": "MAD top-16 excluded",
    "random16": "random 16 excluded",
}


def _write_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    console.print(f"  wrote {path}")


def _load(path: Path) -> dict:
    if not path.is_dir() or not any(path.glob("repeats_*.npz")):
        raise click.ClickException(f"no repeats_*.npz under {path}")
    return load_repeat_shards(path)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dev-all", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--dev-mad16", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--dev-random16", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--test-all", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--test-mad16", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--test-random16", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
def main(
    dev_all: str,
    dev_mad16: str,
    dev_random16: str,
    test_all: str,
    test_mad16: str,
    test_random16: str,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohorts = {
        "dev": {
            "all": _load(Path(dev_all)),
            "mad16": _load(Path(dev_mad16)),
            "random16": _load(Path(dev_random16)),
        },
        "test": {
            "all": _load(Path(test_all)),
            "mad16": _load(Path(test_mad16)),
            "random16": _load(Path(test_random16)),
        },
    }
    cohort_titles = {"dev": "96-dev candidates", "test": "96-test candidates"}

    rows = []
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.05,
        subplot_titles=[
            f"{cohort_titles[c]} · {COND_LABELS[k]}"
            for c in ("dev", "test")
            for k in ("all", "mad16", "random16")
        ],
    )
    overlay = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=[cohort_titles["dev"], cohort_titles["test"]],
    )
    for r, cohort in enumerate(("dev", "test"), start=1):
        for c, cond in enumerate(("all", "mad16", "random16"), start=1):
            data = cohorts[cohort][cond]
            dens = density_table(data["fp"], data["fn"], data["fp_plus_fn"])
            s = fp_fn_summary(data["fp"], data["fn"], data["fp_plus_fn"])
            s["cohort"] = cohort
            s["condition"] = cond
            s["label"] = f"{cohort}_{cond}"
            rows.append(s)
            custom = np.column_stack(
                [dens["n_repeats"], dens["mean_fp"], dens["mean_fn"], dens["density"]]
            )
            hover = (
                "FP+FN=%{x}<br>density=%{customdata[3]:.4f}<br>n=%{customdata[0]}"
                "<br>mean FP=%{customdata[1]:.2f} · mean FN=%{customdata[2]:.2f}<extra></extra>"
            )
            show = r == 1 and c == 1
            fig.add_trace(
                go.Bar(
                    x=dens["fp_plus_fn"],
                    y=dens["perfect_density"],
                    name="perfect",
                    marker_color=PERFECT_COLOR,
                    hovertemplate=hover,
                    customdata=custom,
                    showlegend=show,
                    legendgroup="perfect",
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Bar(
                    x=dens["fp_plus_fn"],
                    y=dens["fp_density"],
                    name="FP share",
                    marker_color=FP_COLOR,
                    hovertemplate=hover,
                    customdata=custom,
                    showlegend=show,
                    legendgroup="fp",
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Bar(
                    x=dens["fp_plus_fn"],
                    y=dens["fn_density"],
                    name="FN share",
                    marker_color=FN_COLOR,
                    hovertemplate=hover,
                    customdata=custom,
                    showlegend=show,
                    legendgroup="fn",
                ),
                row=r,
                col=c,
            )
            overlay.add_trace(
                go.Scatter(
                    x=dens["fp_plus_fn"],
                    y=dens["density"],
                    mode="lines+markers",
                    name=COND_LABELS[cond],
                    line=dict(color=COND_COLORS[cond]),
                    showlegend=(r == 1),
                    legendgroup=cond,
                ),
                row=1,
                col=r,
            )
            plot_fp_fn_density(
                dens,
                f"{cohort_titles[cohort]} | {COND_LABELS[cond]} | n={s['n_repeats']} "
                f"perfect={s['frac_perfect']:.3f} mean FP+FN={s['mean_fp_plus_fn']:.3f}",
                out / f"fp_fn_density_{cohort}_{cond}.html",
            )
            console.print(
                f"  {cohort}/{cond}: n={s['n_repeats']} perfect={s['frac_perfect']:.4f} "
                f"mean FP+FN={s['mean_fp_plus_fn']:.3f} (FP={s['mean_fp']:.3f} FN={s['mean_fn']:.3f})"
            )

    fig.update_layout(
        title="MAD-rank drop-16 (no FF): FP+FN on the same eval within each cohort",
        barmode="stack",
        template="plotly_white",
        height=820,
        width=1400,
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_xaxes(title_text="FP + FN", row=2)
    fig.update_yaxes(title_text="repeat density", col=1)
    _write_html(fig, out / "fp_fn_density_dev_test.html")
    overlay.update_layout(
        title="MAD-rank drop-16 vs all 96 vs random 16",
        template="plotly_white",
        height=480,
        width=1100,
        legend=dict(orientation="h", y=-0.15),
    )
    overlay.update_xaxes(title_text="FP + FN")
    overlay.update_yaxes(title_text="repeat density", col=1)
    _write_html(overlay, out / "fp_fn_density_overlay.html")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "fp_fn_compare_summary.tsv", sep="\t", index=False, float_format="%.6f")

    fig_m, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=False)
    for r, cohort in enumerate(("dev", "test")):
        for c, cond in enumerate(("all", "mad16", "random16")):
            ax = axes[r, c]
            data = cohorts[cohort][cond]
            dens = density_table(data["fp"], data["fn"], data["fp_plus_fn"])
            s = next(x for x in rows if x["cohort"] == cohort and x["condition"] == cond)
            x = dens["fp_plus_fn"].to_numpy()
            ax.bar(x, dens["perfect_density"], color=PERFECT_COLOR, label="perfect")
            ax.bar(
                x,
                dens["fp_density"],
                bottom=dens["perfect_density"],
                color=FP_COLOR,
                label="FP",
            )
            ax.bar(
                x,
                dens["fn_density"],
                bottom=dens["perfect_density"] + dens["fp_density"],
                color=FN_COLOR,
                label="FN",
            )
            ax.set_title(
                f"{cohort_titles[cohort]}\n{COND_LABELS[cond]}\n"
                f"perfect={s['frac_perfect']:.1%}  mean={s['mean_fp_plus_fn']:.2f}",
                fontsize=9,
            )
            ax.grid(axis="y", alpha=0.25)
            if r == 1:
                ax.set_xlabel("FP + FN")
            if c == 0:
                ax.set_ylabel("repeat density")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig_m.suptitle("MAD-rank: exclude top-16 by max(|pct MAD-z|, |z_intra MAD-z|); no FF", fontsize=12)
    fig_m.tight_layout()
    png = out / "fp_fn_density_dev_test.png"
    fig_m.savefig(png, dpi=140)
    plt.close(fig_m)
    console.print(f"  wrote {png}")
    console.print(f"[green]done[/green] {out}")


if __name__ == "__main__":
    main()
