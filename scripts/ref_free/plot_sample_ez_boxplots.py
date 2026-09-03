#!/usr/bin/env python3
"""Publication-quality per-chromosome ezscore boxplots for selected samples.

Replays the same 40+40 fixed-combo draws as ``ref_free_ezscore`` and writes one
PDF/PNG per sample. Annotations are the ez@3 signal ratio, placed horizontally
above each box (unlike the vertical Set D labels).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
if str(REF40_DIR) not in sys.path:
    sys.path.insert(0, str(REF40_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_search_ref40 import CHR_LIST  # noqa: E402
from plot_batch_qc_sets import _load_ez_repeat_matrix  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()

COLOR_NONE = "#9ECAE1"
COLOR_WEAK = "#FDAE6B"
COLOR_STRONG = "#E6550D"
EDGE = "#2F3E4E"
MEDIAN = "#1B1B1B"


def _apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _load_check_meta(input_dir: Path) -> pd.DataFrame:
    check_path = input_dir / "check_samples.tsv"
    meta_path = input_dir / "meta.csv"
    if check_path.is_file():
        df = pd.read_csv(check_path, sep="\t")
    else:
        df = pd.read_csv(meta_path)
    df["sample"] = df["sample"].astype(str)
    return df.drop_duplicates("sample", keep="first")


def _title_fields(row: pd.Series, sample_id: str) -> tuple[str, str, float]:
    orig = str(row.get("orig_sample") or sample_id)
    batch = str(row.get("batch") or row.get("batch_key") or "")
    ff = pd.to_numeric(row.get("ff_before_mq"), errors="coerce")
    ff_val = float(ff) if pd.notna(ff) else float("nan")
    return orig, batch, ff_val


RATIO_COLOR_CUTOFF = 0.3


def _box_color(ratio: float) -> str:
    if ratio >= RATIO_COLOR_CUTOFF:
        return COLOR_STRONG
    if ratio >= 0.005:
        return COLOR_WEAK
    return COLOR_NONE


def plot_one(
    ez_repeats: np.ndarray,
    orig: str,
    batch: str,
    ff: float,
    out_stem: Path,
) -> None:
    """``ez_repeats`` shape ``[n_repeats, n_chr]``."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_chr = ez_repeats.shape[1]
    data = [ez_repeats[:, hi] for hi in range(n_chr)]
    ratio3 = np.array([(vals > 3.0).mean() for vals in data], dtype=float)
    ratio45 = np.array([(vals > 4.5).mean() for vals in data], dtype=float)
    sample_ez3 = float((ez_repeats > 3.0).any(axis=1).mean())
    sample_ez45 = float((ez_repeats > 4.5).any(axis=1).mean())

    fig, ax = plt.subplots(figsize=(13.6, 5.8))
    positions = np.arange(1, n_chr + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": MEDIAN, "linewidth": 1.8},
        whiskerprops={"color": EDGE, "linewidth": 1.0},
        capprops={"color": EDGE, "linewidth": 1.0},
        boxprops={"linewidth": 0.9, "edgecolor": EDGE},
        zorder=3,
    )
    for patch, ratio in zip(bp["boxes"], ratio3):
        patch.set_facecolor(_box_color(float(ratio)))
        patch.set_alpha(0.92)

    ymin = min(float(np.nanmin(ez_repeats)), 0.0)
    p95 = np.array([np.nanpercentile(vals, 95) for vals in data], dtype=float)
    ymax_data = max(float(np.nanmax(p95)), 4.5)
    yrange = max(ymax_data - ymin, 1.0)
    pad = 0.04 * yrange
    label_top = ymax_data
    for r3, r45, y in zip(ratio3, ratio45, p95):
        if r3 >= 0.005 or r45 >= 0.005:
            label_top = max(label_top, y + pad + 0.10 * yrange)
    ax.set_ylim(ymin - 0.04 * yrange, label_top)

    for i, (r3, r45, y) in enumerate(zip(ratio3, ratio45, p95), start=1):
        if r3 < 0.005 and r45 < 0.005:
            continue
        ax.text(
            i,
            y + pad,
            f"{r3:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=0,
            color=EDGE,
            clip_on=False,
            zorder=4,
        )

    ax.axhline(3.0, linestyle=(0, (4, 2.5)), color="#4D4D4D", linewidth=1.15, zorder=0)
    ax.axhline(4.5, linestyle=(0, (1, 2)), color="#1A1A1A", linewidth=1.25, zorder=0)
    ax.yaxis.grid(True, linestyle=":", color="#B0B0B0", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)

    ax.set_xticks(positions)
    ax.set_xticklabels([c.replace("chr", "") for c in CHR_LIST])
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("ezscore")

    ff_txt = f"{ff * 100:.2f}%" if np.isfinite(ff) else "NA"
    ax.set_title(
        f"{orig}\n{batch}    FF = {ff_txt}    ez@3 = {sample_ez3:.2f}    ez@4.5 = {sample_ez45:.2f}",
        pad=12,
        fontweight="semibold",
    )

    legend_handles = [
        Line2D([0], [0], color="#4D4D4D", linestyle=(0, (4, 2.5)), linewidth=1.15, label="ez = 3.0"),
        Line2D([0], [0], color="#1A1A1A", linestyle=(0, (1, 2)), linewidth=1.25, label="ez = 4.5"),
        Patch(facecolor=COLOR_NONE, edgecolor=EDGE, label="ez@3 = 0"),
        Patch(facecolor=COLOR_WEAK, edgecolor=EDGE, label="0 < ez@3 < 0.3"),
        Patch(facecolor=COLOR_STRONG, edgecolor=EDGE, label="ez@3 ≥ 0.3"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        ncol=1,
        borderpad=0.45,
        fancybox=False,
    )

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    console.print(f"[green]OK[/green] {pdf}")
    console.print(f"[green]OK[/green] {png}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    help="input_dir|result_dir|sample[,sample...]  (repeatable)",
)
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
def main(runs: tuple[str, ...], output_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _apply_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in runs:
        parts = spec.split("|")
        if len(parts) != 3:
            raise click.ClickException(
                f"Expected input_dir|result_dir|samples, got: {spec}"
            )
        input_dir = Path(parts[0])
        result_dir = Path(parts[1])
        samples = [s.strip() for s in parts[2].split(",") if s.strip()]
        if not samples:
            raise click.ClickException(f"No samples in spec: {spec}")

        console.print(f"Replay ez matrix: n={len(samples)} input={input_dir}")
        ez_mat, keep, _cfg = _load_ez_repeat_matrix(input_dir, result_dir, samples)
        meta = _load_check_meta(input_dir).set_index("sample")
        keep_index = {u: i for i, u in enumerate(keep)}
        for uid in keep:
            if uid in meta.index:
                orig, batch, ff = _title_fields(meta.loc[uid], uid)
            else:
                orig, batch, ff = uid, "", float("nan")
            safe_batch = batch.replace("/", "_") if batch else "nobatch"
            stem = out_dir / f"{orig}_{safe_batch}_ez_boxplot"
            plot_one(
                ez_repeats=ez_mat[:, :, keep_index[uid]],
                orig=orig,
                batch=batch,
                ff=ff,
                out_stem=stem,
            )


if __name__ == "__main__":
    main()
