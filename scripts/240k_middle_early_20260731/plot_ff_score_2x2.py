#!/usr/bin/env python3
"""2×2 scatter: male/female-ref chrX episcore & zscore vs ff_before_mq (PDF).

Layout
------
  [ male_ref episcore | female_ref episcore ]
  [ male_ref zscore   | female_ref zscore   ]

Rules
-----
- Color by label only (no old_early / new_early marker split).
- Display labels strip Chinese: ``XO/XY``, ``XX/XY``.
- Drop ``69, XYY``.
- Ref assignment:
    male_ref   ← male, XO/XY, XX/XY, XXY
    female_ref ← female, XO, XXX
- One recall level (default 0.5).
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from rich.console import Console

import config as cfg

console = Console()

# Raw label → display label (ASCII only)
LABEL_RENAME = {
    "XO/XY嵌合": "XO/XY",
    "XX/XY嵌合体": "XX/XY",
}

EXCLUDE_LABELS = {"69, XYY"}

# Which display labels go on which reference panel
MALE_REF_LABELS = {"male", "XO/XY", "XX/XY", "XXY"}
FEMALE_REF_LABELS = {"female", "XO", "XXX"}

LABEL_COLOR_MAP = {
    "female": "#9E9E9E",
    "male": "#1F77B4",
    "XO": "#E74C3C",
    "XO/XY": "#C0392B",
    "XXX": "#8E44AD",
    "XXY": "#E67E22",
    "XX/XY": "#16A085",
}

FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _color_for(label: str) -> str:
    if label in LABEL_COLOR_MAP:
        return LABEL_COLOR_MAP[label]
    return FALLBACK_COLORS[hash(str(label)) % len(FALLBACK_COLORS)]


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].astype(str).replace(LABEL_RENAME)
    out = out[~out["label"].isin(EXCLUDE_LABELS)]
    return out


def _load_at_recall(path: Path, score_col: str, recall: float) -> pd.DataFrame:
    if not path.is_file():
        raise click.ClickException(f"Missing {path}")
    df = pd.read_csv(path, sep="\t")
    if score_col not in df.columns:
        raise click.ClickException(f"{path} missing column {score_col}")
    sub = df[np.isclose(df["recall"].astype(float), float(recall), atol=1e-8)].copy()
    if sub.empty:
        raise click.ClickException(f"No rows at recall={recall} in {path}")
    sub = sub.dropna(subset=["ff_before_mq", score_col])
    return _normalize_labels(sub)


def _filter_for_ref(df: pd.DataFrame, ref_gender: str) -> pd.DataFrame:
    """Keep samples assigned to this reference panel."""
    allowed = MALE_REF_LABELS if ref_gender == "male" else FEMALE_REF_LABELS
    return df.loc[df["label"].astype(str).isin(allowed)].copy()


def _scatter_ax(ax, df: pd.DataFrame, score_col: str, title: str) -> None:
    labels = sorted(df["label"].dropna().astype(str).unique(), key=str)
    for label in labels:
        sub = df[df["label"].astype(str) == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["ff_before_mq"].to_numpy(dtype=float),
            sub[score_col].to_numpy(dtype=float),
            s=36,
            c=_color_for(label),
            label=label,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
            zorder=3,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("ff_before_mq")
    ax.set_ylabel(score_col)
    ax.grid(True, color="#ECECEC", linewidth=0.6)
    ax.set_facecolor("#FAFAFA")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _pos: f"{100 * x:.1f}%")
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--recall", type=float, default=0.5, show_default=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="PDF path (default: plots/chrX_ff_score_male_female_ref_2x2_recall*.pdf)",
)
def main(recall: float, output_path: Path | None) -> None:
    """Write a 2×2 PDF of ff_before_mq vs male/female-ref scores."""
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (
        cfg.PLOTS_DIR / f"chrX_ff_score_male_female_ref_2x2_recall{recall:g}.pdf"
    )

    panels = [
        (
            "male",
            cfg.MALE_REF_EPISCORE_COLLECTED,
            "chrX_episcore",
            "Male ref — chrX episcore",
        ),
        (
            "female",
            cfg.FEMALE_REF_EPISCORE_COLLECTED,
            "chrX_episcore",
            "Female ref — chrX episcore",
        ),
        (
            "male",
            cfg.MALE_REF_ZSCORE_COLLECTED,
            "chrX_zscore",
            "Male ref — chrX zscore",
        ),
        (
            "female",
            cfg.FEMALE_REF_ZSCORE_COLLECTED,
            "chrX_zscore",
            "Female ref — chrX zscore",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    legend_handles: dict[str, plt.Line2D] = {}
    for ax, (ref_g, path, score_col, title) in zip(ax_list, panels):
        df = _filter_for_ref(_load_at_recall(path, score_col, recall), ref_g)
        _scatter_ax(ax, df, score_col, f"{title}\n(recall={recall:g}, n={len(df)})")
        for label in sorted(df["label"].dropna().astype(str).unique(), key=str):
            if label not in legend_handles:
                legend_handles[label] = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=_color_for(label),
                    markeredgecolor="white",
                    markersize=8,
                    label=label,
                )
        console.print(
            f"  {title}: n={len(df)}  labels={sorted(df['label'].astype(str).unique())}"
        )

    fig.suptitle(
        f"chrX scores vs FF (middle male/female refs; recall={recall:g})",
        fontsize=13,
        x=0.02,
        ha="left",
    )
    if legend_handles:
        fig.legend(
            handles=[legend_handles[k] for k in sorted(legend_handles)],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            title="label",
        )

    with PdfPages(out) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Wrote[/green] {out}")


if __name__ == "__main__":
    main()
