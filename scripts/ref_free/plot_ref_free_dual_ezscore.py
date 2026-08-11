#!/usr/bin/env python3
"""Dual ezscore signal-ratio plot: eval (dev/test) | val, with cutoff slider.

Both panels' y-values and separation degrees (AUC + Youden J) update when the
ezscore cutoff slider moves.
"""

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


def _scatter(df: pd.DataFrame, y_col: str, *, showlegend: bool) -> list:
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
            )
        )
    return traces


def _subplot_labels(c: float, sep_eval: dict, sep_val: dict, n_eval: int, n_val: int) -> list:
    se = sep_eval.get(c, {})
    sv = sep_val.get(c, {})
    return [
        dict(
            text=f"Eval (dev/test) n={n_eval}<br>{format_sep_pair(se)}",
            x=0.225,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=13),
        ),
        dict(
            text=(
                f"Val n={n_val}<br>{format_sep_pair(sv)}"
                if n_val
                else "Val (no samples)"
            ),
            x=0.775,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=13),
        ),
    ]


def build_dual_figure(
    eval_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ez_cutoffs: list[float],
    title: str,
    subtitle: str,
    ff_min: float = DEFAULT_FF_MIN,
    default_ez: float | None = None,
) -> go.Figure:
    eval_df = _prepare(eval_df)
    val_df = _prepare(val_df)
    eval_ff = eval_df[eval_df["ff_before_mq"] >= ff_min]
    val_ff = val_df[val_df["ff_before_mq"] >= ff_min] if len(val_df) else val_df

    if default_ez is None or default_ez not in ez_cutoffs:
        default_ez = 3.0 if 3.0 in ez_cutoffs else ez_cutoffs[0]
    col0 = _ez_ratio_col(default_ez)

    sep_eval = {
        float(c): separation_index(eval_ff, _ez_ratio_col(c), ff_min=0.0)
        for c in ez_cutoffs
        if _ez_ratio_col(c) in eval_ff.columns
    }
    sep_val = {
        float(c): separation_index(val_ff, _ez_ratio_col(c), ff_min=0.0)
        for c in ez_cutoffs
        if len(val_ff) and _ez_ratio_col(c) in val_ff.columns
    }

    def _title_text(c: float) -> str:
        se = sep_eval.get(float(c), {})
        sv = sep_val.get(float(c), {})
        return (
            f"{title}<br><sup>{subtitle}</sup><br>"
            f"<sup>ff≥{ff_min*100:.0f}% · ez cutoff={c:g} · "
            f"eval [{format_sep_pair(se)}] · val [{format_sep_pair(sv)}]</sup>"
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Eval", "Val"),  # replaced by annotations in layout
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    for tr in _scatter(eval_ff, col0, showlegend=True):
        fig.add_trace(tr, row=1, col=1)
    has_val = len(val_ff) > 0
    if has_val:
        for tr in _scatter(val_ff, col0, showlegend=False):
            fig.add_trace(tr, row=1, col=2)
    else:
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", name="empty", showlegend=False),
            row=1,
            col=2,
        )

    # Precompute y per cutoff for explicit restyle
    y_by_cut = {}
    for c in ez_cutoffs:
        col = _ez_ratio_col(c)
        ys = [
            eval_ff.loc[~eval_ff["is_trisomy"], col].tolist(),
            eval_ff.loc[eval_ff["is_trisomy"], col].tolist(),
        ]
        idxs = [0, 1]
        if has_val:
            ys += [
                val_ff.loc[~val_ff["is_trisomy"], col].tolist(),
                val_ff.loc[val_ff["is_trisomy"], col].tolist(),
            ]
            idxs += [2, 3]
        y_by_cut[float(c)] = (ys, idxs)

    steps = []
    for c in ez_cutoffs:
        cf = float(c)
        ys, idxs = y_by_cut[cf]
        steps.append(
            dict(
                method="update",
                args=[
                    {"y": ys},
                    {
                        "title": {"text": _title_text(cf)},
                        "annotations": _subplot_labels(
                            cf, sep_eval, sep_val, len(eval_ff), len(val_ff)
                        ),
                    },
                    idxs,
                ],
                label=f"{c:g}",
            )
        )

    fig.update_layout(
        title=dict(text=_title_text(default_ez), x=0.5),
        height=580,
        width=1100,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(t=140, b=80),
        annotations=_subplot_labels(
            default_ez, sep_eval, sep_val, len(eval_ff), len(val_ff)
        ),
        sliders=[
            dict(
                active=ez_cutoffs.index(default_ez),
                currentvalue=dict(prefix="ezscore cutoff: "),
                pad=dict(t=40, b=10),
                steps=steps,
                x=0.15,
                len=0.7,
            )
        ],
    )
    # Hide default subplot titles (we use annotations)
    for ann in fig.layout.annotations:
        if ann.text in ("Eval", "Val"):
            ann.text = ""
    fig.update_xaxes(title_text="ff_before_mq", tickformat=".1%")
    fig.update_yaxes(title_text="ezscore signal ratio", range=[-0.02, 1.05], row=1, col=1)
    fig.update_yaxes(range=[-0.02, 1.05], row=1, col=2)
    return fig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--result-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--scores-tsv", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-html", default=None, type=click.Path(path_type=Path))
@click.option("--title", default="40+40 ezscore signal ratio", show_default=True)
@click.option("--ff-min", default=DEFAULT_FF_MIN, show_default=True, type=float)
def main(
    result_dir: Path,
    scores_tsv: Path | None,
    output_html: Path | None,
    title: str,
    ff_min: float,
) -> None:
    ref_dir = result_dir / "ref_free_ezscore"
    scores = Path(scores_tsv) if scores_tsv else ref_dir / "abnormality_signal_ratio.tsv"
    config = json.loads((ref_dir / "run_config.json").read_text())
    ez_cutoffs = [float(x) for x in config.get("ez_cutoffs", [3.0])]
    if "primary_ez_cutoff" in config:
        primary_ez = float(config["primary_ez_cutoff"])
    elif str(config.get("combo_mode", "")) == "fixed":
        primary_ez = 4.5
    else:
        primary_ez = 3.0
    df = pd.read_csv(scores, sep="\t")
    df = _prepare(df)
    is_val = df["set"].astype(str).eq("val") if "set" in df.columns else pd.Series(False, index=df.index)
    eval_df = df[~is_val]
    val_df = df[is_val]
    mode = config.get("combo_mode", "?")
    subtitle = f"mode={mode} | pairs={config.get('n_ez_combos')} | primary ez={primary_ez:g}"
    fig = build_dual_figure(
        eval_df, val_df, ez_cutoffs, title, subtitle, ff_min=ff_min, default_ez=primary_ez
    )
    out = output_html or (result_dir / "plots" / "ezscore_eval_vs_val.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    console.print(f"[green]OK[/green] Wrote {out} (eval n={len(eval_df)}, val n={len(val_df)})")


if __name__ == "__main__":
    main()
