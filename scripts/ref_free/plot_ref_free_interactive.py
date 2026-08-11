#!/usr/bin/env python3
"""Build interactive Plotly HTML for 40+40 ref_free signal-ratio sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

from separation import format_sep_pair, separation_index
from val_blacklist import drop_blacklisted

console = Console()

MARKER = {
    "Normal": dict(color="#9e9e9e", size=7, opacity=0.55),
    "trisomy": dict(color="#d62728", size=9, opacity=0.95),
}
DEFAULT_FF_MIN = 0.01


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = drop_blacklisted(df)
    out["ff_before_mq"] = pd.to_numeric(out["ff_before_mq"], errors="coerce")
    out["is_trisomy"] = out["label"].astype(str).str.match(r"^T\d")
    out["status"] = out["is_trisomy"].map({True: "trisomy", False: "Normal"})
    return out


def _ez_ratio_col(cutoff: float) -> str:
    return f"ezscore_signal_ratio_{cutoff:g}"


def _scatter_traces(
    df: pd.DataFrame,
    y_col: str,
    *,
    visible: bool = True,
    showlegend: bool = False,
) -> list:
    traces = []
    for status in ("Normal", "trisomy"):
        sub = df[df["status"] == status]
        traces.append(
            go.Scatter(
                x=sub["ff_before_mq"],
                y=sub[y_col],
                mode="markers",
                name=status,
                legendgroup=status,
                showlegend=showlegend,
                marker=MARKER[status],
                text=sub["sample"],
                hovertemplate=(
                    "%{text}<br>ff=%{x:.4f}<br>ratio=%{y:.4f}<extra>" + status + "</extra>"
                ),
                visible=visible,
            )
        )
    return traces


def build_figure(
    df: pd.DataFrame,
    ez_cutoffs: list[float],
    title: str,
    subtitle: str,
    ff_min: float = DEFAULT_FF_MIN,
    default_ez: float | None = None,
) -> go.Figure:
    df = _prepare(df)
    # Prefer eval (non-val) for the 3-panel overview
    if "set" in df.columns:
        df = df[df["set"].astype(str).ne("val")].copy()

    if default_ez is None or default_ez not in ez_cutoffs:
        default_ez = 3.0 if 3.0 in ez_cutoffs else ez_cutoffs[0]
    ez_col0 = _ez_ratio_col(default_ez)

    df_ff = df[df["ff_before_mq"] >= ff_min]
    sep_ep = separation_index(df_ff, "episcore_signal_ratio", ff_min=0.0)
    sep_z = separation_index(df_ff, "zscore_signal_ratio", ff_min=0.0)
    sep_ez = {
        c: separation_index(df_ff, _ez_ratio_col(c), ff_min=0.0) for c in ez_cutoffs
    }

    def _full_title(c: float) -> str:
        se = sep_ez.get(c, {})
        return (
            f"{title}<br><sup>{subtitle}</sup><br>"
            f"<sup>ff≥{ff_min*100:.0f}% · cutoff={c:g} · "
            f"ep [{format_sep_pair(sep_ep)}] · z [{format_sep_pair(sep_z)}] · "
            f"ez [{format_sep_pair(se)}]</sup>"
        )

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Episcore", "Zscore", "Ezscore"),
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    for tr in _scatter_traces(df, "episcore_signal_ratio", showlegend=True):
        fig.add_trace(tr, row=1, col=1)
    for tr in _scatter_traces(df, "zscore_signal_ratio"):
        fig.add_trace(tr, row=1, col=2)
    for tr in _scatter_traces(df, ez_col0):
        fig.add_trace(tr, row=1, col=3)

    for tr in _scatter_traces(df_ff, "episcore_signal_ratio", visible=False):
        fig.add_trace(tr, row=1, col=1)
    for tr in _scatter_traces(df_ff, "zscore_signal_ratio", visible=False):
        fig.add_trace(tr, row=1, col=2)
    for tr in _scatter_traces(df_ff, ez_col0, visible=False):
        fig.add_trace(tr, row=1, col=3)

    n_base, n_ff = 6, 6
    ez_y_all = {}
    ez_y_ff = {}
    for c in ez_cutoffs:
        col = _ez_ratio_col(c)
        if col not in df.columns:
            raise click.ClickException(f"Missing column {col}")
        ez_y_all[c] = {
            "Normal": df.loc[~df["is_trisomy"], col].tolist(),
            "trisomy": df.loc[df["is_trisomy"], col].tolist(),
        }
        ez_y_ff[c] = {
            "Normal": df_ff.loc[~df_ff["is_trisomy"], col].tolist(),
            "trisomy": df_ff.loc[df_ff["is_trisomy"], col].tolist(),
        }

    steps = []
    for c in ez_cutoffs:
        steps.append(
            dict(
                method="update",
                args=[
                    {
                        "y": [
                            ez_y_all[c]["Normal"],
                            ez_y_all[c]["trisomy"],
                            ez_y_ff[c]["Normal"],
                            ez_y_ff[c]["trisomy"],
                        ]
                    },
                    {"title": {"text": _full_title(c)}},
                    [4, 5, 10, 11],
                ],
                label=f"{c:g}",
            )
        )

    vis_all = [True] * n_base + [False] * n_ff
    vis_ff = [False] * n_base + [True] * n_ff
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0.0,
            y=1.18,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(label="All samples", method="update", args=[{"visible": vis_all}]),
                dict(
                    label=f"ff ≥ {ff_min * 100:.0f}%",
                    method="update",
                    args=[{"visible": vis_ff}],
                ),
            ],
        )
    ]

    fig.update_layout(
        title=dict(text=_full_title(default_ez), x=0.5),
        height=540,
        width=1200,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(t=120, b=80),
        updatemenus=updatemenus,
        sliders=[
            dict(
                active=ez_cutoffs.index(default_ez),
                currentvalue=dict(prefix="ezscore cutoff: "),
                pad=dict(t=30, b=10),
                steps=steps,
                x=0.15,
                len=0.7,
            )
        ],
    )
    fig.update_xaxes(title_text="ff_before_mq", tickformat=".1%")
    fig.update_yaxes(title_text="signal ratio", range=[-0.02, 1.05], row=1, col=1)
    for col in (2, 3):
        fig.update_yaxes(range=[-0.02, 1.05], row=1, col=col)
    return fig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--result-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output-html", default=None, type=click.Path(path_type=Path))
@click.option("--title", default="40+40 reference-free signal ratio", show_default=True)
@click.option("--ff-min", default=DEFAULT_FF_MIN, show_default=True, type=float)
@click.option(
    "--scores-tsv",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def main(
    result_dir: Path,
    output_html: Path | None,
    title: str,
    ff_min: float,
    scores_tsv: Path | None,
) -> None:
    ref_dir = result_dir / "ref_free_ezscore"
    scores = Path(scores_tsv) if scores_tsv else ref_dir / "abnormality_signal_ratio.tsv"
    config_path = ref_dir / "run_config.json"
    if not scores.is_file():
        raise click.ClickException(f"Missing {scores}")
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    ez_cutoffs = [float(x) for x in config.get("ez_cutoffs", [3.0])]
    if "primary_ez_cutoff" in config:
        primary_ez = float(config["primary_ez_cutoff"])
    elif str(config.get("combo_mode", "")) == "fixed":
        primary_ez = 4.5
    else:
        primary_ez = 3.0

    df = pd.read_csv(scores, sep="\t")
    mode = config.get("combo_mode", "?")
    if mode == "fixed":
        subtitle = (
            f"fixed combo ep {config.get('ep_threshold')}/{config.get('ep_recall')} | "
            f"z {config.get('z_threshold')}/{config.get('z_recall')} | "
            f"primary ez={primary_ez:g}"
        )
    else:
        subtitle = (
            f"filtered combos | ep thr[{config.get('ep_threshold_min')},"
            f"{config.get('ep_threshold_max')}] "
            f"rec[{config.get('ep_recall_min')},{config.get('ep_recall_max')}] | "
            f"z thr[{config.get('z_threshold_min')},{config.get('z_threshold_max')}] "
            f"rec[{config.get('z_recall_min')},{config.get('z_recall_max')}] | "
            f"ez pairs={config.get('n_ez_combos')} ({config.get('ez_pair_mode')})"
        )

    fig = build_figure(
        df, ez_cutoffs, title=title, subtitle=subtitle, ff_min=ff_min, default_ez=primary_ez
    )
    out = output_html or (result_dir / "plots" / "signal_ratio.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    console.print(f"[green]OK[/green] Wrote {out}")


if __name__ == "__main__":
    main()
