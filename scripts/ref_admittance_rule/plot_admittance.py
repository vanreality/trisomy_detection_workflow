#!/usr/bin/env python3
"""HTML plots for admittance-rule analysis (Q1 / Q2 / Q3)."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

from common import (
    BAD_COLOR,
    FN_COLOR,
    FP_COLOR,
    OK_COLOR,
    PERFECT_COLOR,
    density_table,
    load_repeat_shards,
)

console = Console()


def _write(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    console.print(f"  wrote {path}")


def plot_enrichment(ranked: pd.DataFrame, out: Path) -> None:
    df = ranked.sort_values("lift_bad", ascending=False)
    colors = df["flag"].map(
        {"toxic": BAD_COLOR, "protective": PERFECT_COLOR, "neutral": OK_COLOR}
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["sample"],
            y=df["lift_bad"],
            marker_color=colors,
            name="lift_bad (either role)",
            hovertemplate=(
                "%{x}<br>lift_bad=%{y:.4f}<br>flag=%{customdata[0]}"
                "<br>ff=%{customdata[1]:.4f}<br>pct MAD-z max=%{customdata[2]:.2f}"
                "<br>intra MAD-z max=%{customdata[3]:.2f}<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    df["flag"],
                    df["ff_before_mq"],
                    df["max_abs_pct_madz"],
                    df["max_abs_intra_madz"],
                ]
            ),
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#888")
    fig.update_layout(
        title="Q1: pool-sample lift in bad (FP+FN≥5) 40+40 draws — role=either",
        xaxis_title="dev Normal pool sample",
        yaxis_title="P(in 80 | bad) / P(in 80)",
        template="plotly_white",
        height=520,
        width=1100,
        xaxis_tickangle=-60,
        margin=dict(b=140, t=80),
    )
    _write(fig, out / "q1_lift_bad.html")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=df["lift_perfect"],
            y=df["lift_bad"],
            mode="markers+text",
            text=np.where(df["flag"] != "neutral", df["sample"], ""),
            textposition="top center",
            marker=dict(color=colors, size=9),
            hovertemplate="%{text}<br>lift_perfect=%{x:.4f}<br>lift_bad=%{y:.4f}<extra></extra>",
        )
    )
    fig2.add_vline(x=1.0, line_dash="dot", line_color="#888")
    fig2.add_hline(y=1.0, line_dash="dot", line_color="#888")
    fig2.update_layout(
        title="Q1: lift_perfect vs lift_bad (either role)",
        xaxis_title="lift in perfect draws",
        yaxis_title="lift in bad draws",
        template="plotly_white",
        height=560,
        width=720,
    )
    _write(fig2, out / "q1_lift_scatter.html")


def plot_feature_vs_lift(ranked: pd.DataFrame, out: Path) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("ff_before_mq", "max |percentage MAD-z|", "max |z_intra MAD-z|"),
    )
    colors = ranked["flag"].map(
        {"toxic": BAD_COLOR, "protective": PERFECT_COLOR, "neutral": OK_COLOR}
    )
    for col, key in enumerate(("ff_before_mq", "max_abs_pct_madz", "max_abs_intra_madz"), start=1):
        fig.add_trace(
            go.Scatter(
                x=ranked[key],
                y=ranked["lift_bad"],
                mode="markers",
                marker=dict(color=colors, size=8),
                text=ranked["sample"],
                hovertemplate="%{text}<br>x=%{x:.4f}<br>lift_bad=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color="#888", row=1, col=col)
    fig.update_yaxes(title_text="lift_bad", col=1)
    fig.update_layout(
        title="Q1: is toxic lift explained by FF or per-chr MAD outliers?",
        template="plotly_white",
        height=420,
        width=1100,
        margin=dict(t=80),
    )
    _write(fig, out / "q1_features_vs_lift.html")


def plot_set_distributions(by_class: pd.DataFrame, compare: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("ff_80_std", "FF std of the 80 refs"),
        ("ff_80_mean", "FF mean of the 80 refs"),
        ("ez_sd_mean", "mean ez-ref SD across chr"),
        ("n_fail_members", "count of MAD/FF-fail members in the 80"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[t for _, t in metrics])
    palette = {"perfect": PERFECT_COLOR, "bad": BAD_COLOR, "ok": OK_COLOR}
    for i, (col, _title) in enumerate(metrics):
        r, c = divmod(i, 2)
        for cls in ("perfect", "ok", "bad"):
            sub = by_class.loc[by_class["class"] == cls, col]
            fig.add_trace(
                go.Histogram(
                    x=sub,
                    name=cls,
                    marker_color=palette[cls],
                    opacity=0.65,
                    nbinsx=40,
                    legendgroup=cls,
                    showlegend=(i == 0),
                ),
                row=r + 1,
                col=c + 1,
            )
    fig.update_layout(
        title="Q2: set-level feature distributions (perfect vs ok vs bad)",
        barmode="overlay",
        template="plotly_white",
        height=720,
        width=1000,
    )
    _write(fig, out / "q2_set_distributions.html")

    top = compare.head(20).iloc[::-1]
    fig2 = go.Figure(
        go.Bar(
            x=top["cliffs_delta_bad_minus_perfect"],
            y=top["metric"],
            orientation="h",
            marker_color=np.where(
                top["cliffs_delta_bad_minus_perfect"] > 0, BAD_COLOR, PERFECT_COLOR
            ),
            hovertemplate=(
                "%{y}<br>Cliff's δ=%{x:.3f}<br>mean perfect=%{customdata[0]:.4g}"
                "<br>mean bad=%{customdata[1]:.4g}<extra></extra>"
            ),
            customdata=np.column_stack([top["mean_perfect"], top["mean_bad"]]),
        )
    )
    fig2.add_vline(x=0, line_color="#888")
    fig2.update_layout(
        title="Q2: Cliff's δ (bad − perfect) for set-level metrics",
        xaxis_title="Cliff's delta (positive = larger in bad sets)",
        template="plotly_white",
        height=max(420, 22 * len(top) + 80),
        width=860,
        margin=dict(l=180, t=70),
    )
    _write(fig2, out / "q2_cliffs_delta.html")


def plot_fp_fn_density(dens: pd.DataFrame, title: str, out_html: Path) -> None:
    fig = go.Figure()
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
            name="perfect (0)",
            marker_color=PERFECT_COLOR,
            hovertemplate=hover,
            customdata=custom,
        )
    )
    fig.add_trace(
        go.Bar(
            x=dens["fp_plus_fn"],
            y=dens["fp_density"],
            name="FP share",
            marker_color=FP_COLOR,
            hovertemplate=hover,
            customdata=custom,
        )
    )
    fig.add_trace(
        go.Bar(
            x=dens["fp_plus_fn"],
            y=dens["fn_density"],
            name="FN share",
            marker_color=FN_COLOR,
            hovertemplate=hover,
            customdata=custom,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="FP + FN (per repeat)",
        yaxis_title="repeat density",
        barmode="stack",
        template="plotly_white",
        height=520,
        width=880,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=80, b=80),
    )
    _write(fig, out_html)


def plot_proof(proof: pd.DataFrame, out: Path, dose_path: Path | None = None) -> None:
    rules = proof.loc[proof["label"].astype(str).str.startswith("rule:")].copy()
    if rules.empty:
        return
    rules["rule"] = rules["label"].str.replace("rule:", "", regex=False)
    if "spearman_nfail_vs_fpfn" in rules.columns:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=rules["rule"],
                y=rules["spearman_nfail_vs_fpfn"],
                name="QC: ρ(n fail members, FP+FN)",
                marker_color=BAD_COLOR,
            )
        )
        if "random_spearman_mean" in rules.columns:
            fig.add_trace(
                go.Bar(
                    x=rules["rule"],
                    y=rules["random_spearman_mean"],
                    name="matched-N random drop",
                    marker_color=OK_COLOR,
                    error_y=dict(
                        type="data",
                        array=rules["random_spearman_sd"].fillna(0).to_numpy()
                        if "random_spearman_sd" in rules.columns
                        else None,
                        visible=True,
                    ),
                )
            )
        fig.update_layout(
            title="Q3: Spearman ρ(n_fail_members, FP+FN) — QC vs matched-N random",
            yaxis_title="Spearman rho (higher = fail members track errors)",
            template="plotly_white",
            height=480,
            width=920,
            barmode="group",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(b=90, t=70),
        )
        _write(fig, out / "q3_spearman_vs_random.html")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=rules["rule"],
            y=rules["frac_perfect"],
            name="QC all-80-pass repeats",
            marker_color=PERFECT_COLOR,
        )
    )
    if "random_frac_perfect_mean" in rules.columns:
        fig2.add_trace(
            go.Bar(
                x=rules["rule"],
                y=rules["random_frac_perfect_mean"],
                name="matched-N random drop",
                marker_color=OK_COLOR,
                error_y=dict(
                    type="data",
                    array=rules.get("random_frac_perfect_sd", 0),
                    visible=True,
                ),
            )
        )
    base = proof.loc[proof["label"] == "all_repeats", "frac_perfect"]
    if len(base):
        fig2.add_hline(
            y=float(base.iloc[0]),
            line_dash="dot",
            annotation_text="all repeats",
        )
    fig2.update_layout(
        title="Q3: frac_perfect among repeats with 0 fail members (may be rare)",
        yaxis_title="fraction of repeats with FP+FN=0",
        template="plotly_white",
        height=480,
        width=920,
        barmode="group",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(b=90, t=70),
    )
    _write(fig2, out / "q3_frac_perfect_vs_random.html")

    if dose_path is not None and dose_path.is_file():
        dose = pd.read_csv(dose_path, sep="\t")
        fig3 = go.Figure()
        for rule, sub in dose.groupby("rule"):
            qc = sub.loc[sub["kind"] == "qc"]
            rnd = sub.loc[sub["kind"].astype(str).str.startswith("random")]
            if qc.empty:
                continue
            fig3.add_trace(
                go.Scatter(
                    x=qc["n_fail_members"],
                    y=qc["mean_fp_plus_fn"],
                    mode="lines+markers",
                    name=f"{rule} QC",
                )
            )
            if not rnd.empty:
                agg = (
                    rnd.groupby("n_fail_members", as_index=False)
                    .agg(mean=("mean_fp_plus_fn", "mean"), sd=("mean_fp_plus_fn", "std"))
                )
                fig3.add_trace(
                    go.Scatter(
                        x=agg["n_fail_members"],
                        y=agg["mean"],
                        mode="lines",
                        name=f"{rule} random",
                        line=dict(dash="dot"),
                        error_y=dict(type="data", array=agg["sd"].fillna(0)),
                    )
                )
        fig3.update_layout(
            title="Q3: dose-response — mean FP+FN vs number of fail-rule members in the 80",
            xaxis_title="n fail members in the 40+40",
            yaxis_title="mean FP+FN",
            template="plotly_white",
            height=520,
            width=920,
        )
        _write(fig3, out / "q3_dose_response.html")


def plot_redraw_compare(score_dirs: list[tuple[str, Path]], out: Path) -> None:
    fig = go.Figure()
    rows = []
    for name, path in score_dirs:
        if not path.is_dir() or not any(path.glob("repeats_*.npz")):
            continue
        data = load_repeat_shards(path)
        dens = density_table(data["fp"], data["fn"], data["fp_plus_fn"])
        s = {
            "pool": name,
            "n_repeats": int(data["fp_plus_fn"].size),
            "frac_perfect": float((data["fp_plus_fn"] == 0).mean()),
            "mean_fp_plus_fn": float(data["fp_plus_fn"].mean()),
            "mean_fn": float(data["fn"].mean()),
            "mean_fp": float(data["fp"].mean()),
        }
        rows.append(s)
        fig.add_trace(
            go.Scatter(
                x=dens["fp_plus_fn"],
                y=dens["density"],
                mode="lines+markers",
                name=f"{name} (perfect={s['frac_perfect']:.3f})",
            )
        )
    if not rows:
        return
    fig.update_layout(
        title="Q3: prospective 40+40 FP+FN density (original vs admitted vs random-N)",
        xaxis_title="FP + FN",
        yaxis_title="repeat density",
        template="plotly_white",
        height=520,
        width=880,
    )
    _write(fig, out / "q3_redraw_density.html")
    pd.DataFrame(rows).to_csv(out / "q3_redraw_summary.tsv", sep="\t", index=False, float_format="%.6f")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--analysis-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--baseline-score-dir", default=None, type=click.Path(file_okay=False))
@click.option("--admitted-score-dir", default=None, type=click.Path(file_okay=False))
@click.option("--random-score-dir", default=None, type=click.Path(file_okay=False))
def main(
    analysis_dir: str,
    output_dir: str | None,
    baseline_score_dir: str | None,
    admitted_score_dir: str | None,
    random_score_dir: str | None,
) -> None:
    analysis = Path(analysis_dir)
    out = Path(output_dir) if output_dir else analysis / "plots"
    out.mkdir(parents=True, exist_ok=True)
    console.rule("[bold blue]plot admittance")

    dens = pd.read_csv(analysis / "fp_fn_density.tsv", sep="\t")
    plot_fp_fn_density(
        dens,
        "Baseline 40+40: FP+FN density (ez@4.5, ff≥1%)",
        out / "fp_fn_density.html",
    )
    ranked = pd.read_csv(analysis / "toxic_protective.tsv", sep="\t")
    plot_enrichment(ranked, out)
    plot_feature_vs_lift(ranked, out)
    by_class = pd.read_csv(analysis / "set_features_by_class.tsv", sep="\t")
    compare = pd.read_csv(analysis / "set_feature_compare.tsv", sep="\t")
    plot_set_distributions(by_class, compare, out)

    proof_tsv = analysis / "proof" / "proof_retrospective.tsv"
    if proof_tsv.is_file():
        plot_proof(
            pd.read_csv(proof_tsv, sep="\t"),
            out,
            dose_path=analysis / "proof" / "proof_dose_response.tsv",
        )

    pairs = []
    if baseline_score_dir:
        pairs.append(("baseline96", Path(baseline_score_dir)))
    if admitted_score_dir:
        pairs.append(("admitted", Path(admitted_score_dir)))
    if random_score_dir:
        pairs.append(("random_n", Path(random_score_dir)))
    if pairs:
        plot_redraw_compare(pairs, out)
    console.print(f"[green]OK[/green] plots in {out}")


if __name__ == "__main__":
    main()
