#!/usr/bin/env python3
"""Stable ref40 LOO vs fixed E(μ/σ) vs 40+40 signal_ratio (dev/test).

Plot 1: 20260812 stable ref40 as epi/z reference; ez μ/σ from LOO on those 40.
        Dev (excl. the 40) and test, depth pass, Normal+single T#, FF≥0.01.
        2×2: ez-vs-chr scatters + Sens/Spec/PPV bars.

Plot 2: same eval cohorts, ez from fixed E(μ)/E(σ) of the 220 clean-pool draws.

Plot 3: 40+40 signal_ratio vs FF. Pool = dev Normal with the same QC+FF filters
        (falls back to even split if n<80). Eval = dev single T# + all test
        samples with the same QC+FF filters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_ref40_vs_fixed_profiles import (  # noqa: E402
    BLUE,
    DEFAULT_CUTOFF,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    GRAY,
    PURITY_LO,
    RED,
    _cohort_frame,
    _resolve_n_jobs,
    _target_mask_from_labels,
    build_universe,
    fixed_ez_profiles_fully_fixed,
    label_to_target_chr,
    load_fixed_params,
    plot_fixed_ez_scatter,
    run_ref40_signal_ratio,
)
from pool_size_ez_ref_bands import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_OUT,
    DEFAULT_PARQUET,
    DEFAULT_TOXIC,
    combined_on_query,
    loo_combined,
)

console = Console()

DEFAULT_REF40 = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260812-stable_ref40/ref40_samples.txt"
)
ORANGE = "#E07A3D"
PURPLE = "#7C3AED"
TEAL = "#0F766E"
BAR_OK = "#2A9D8F"
BAR_LO = "#E76F51"


def _read_ref40(path: Path, sample_index: dict[str, int]) -> np.ndarray:
    names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    missing = [s for s in names if s not in sample_index]
    if missing:
        raise click.ClickException(f"ref40 not in parquet/meta: {missing[:8]}")
    return np.array([sample_index[s] for s in names], dtype=np.int64)


def _qc_ff_mask(ctx: dict) -> np.ndarray:
    return (ctx["depth_arr"] == "pass") & (ctx["ff_arr"] >= 0.01)


def _norm_or_single(ctx: dict) -> np.ndarray:
    return (ctx["label_arr"] == "Normal") | np.array(
        [label_to_target_chr(x) is not None for x in ctx["label_arr"]]
    )


def ez_stable_ref40_loo(arrays: dict, ref_idx: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    """epi/z vs ref40; ez μ/σ from LOO combined scores on the same 40."""
    comb = combined_on_query(
        arrays["hypo"],
        arrays["hyper"],
        arrays["hypo_cnt"],
        arrays["hyper_cnt"],
        arrays["pct"],
        ref_idx,
        eval_idx,
    )
    loo = loo_combined(
        arrays["hypo"][:, ref_idx],
        arrays["hyper"][:, ref_idx],
        arrays["hypo_cnt"][:, ref_idx],
        arrays["hyper_cnt"][:, ref_idx],
        arrays["pct"][:, ref_idx],
    )
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(loo, axis=1)
        sd = np.nanstd(loo, axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (comb - mu[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]


def detection_metrics(labels: np.ndarray, call: np.ndarray) -> dict[str, float | int]:
    is_pos = np.array([label_to_target_chr(x) is not None for x in labels])
    is_neg = labels == "Normal"
    keep = is_pos | is_neg
    y = is_pos[keep]
    p = np.asarray(call, dtype=bool)[keep]
    tp = int((y & p).sum())
    fn = int((y & ~p).sum())
    fp = int((~y & p).sum())
    tn = int((~y & ~p).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
    }


def _plot_metric_bars(ax: plt.Axes, metrics: dict, title: str) -> None:
    names = ("Sens", "Spec", "PPV")
    vals = [float(metrics["sens"]), float(metrics["spec"]), float(metrics["ppv"])]
    colors = [BAR_OK if np.isfinite(v) and abs(v - 1.0) < 1e-12 else BAR_LO for v in vals]
    heights = [0.0 if not np.isfinite(v) else v for v in vals]
    ax.bar(names, heights, color=colors, width=0.38, edgecolor="white", linewidth=0.4, zorder=3)
    for i, v in enumerate(vals):
        label = "n/a" if not np.isfinite(v) else f"{v:.3f}"
        ypos = 0.04 if not np.isfinite(v) else min(v + 0.025, 1.06)
        ax.text(i, ypos, label, ha="center", va="bottom", fontsize=8.5, color="#222")
    ax.set_ylim(0, 1.14)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=10, pad=7)
    ax.axhline(1.0, color="#C5CAD1", lw=0.8, ls=":", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8B939E")
    ax.spines["bottom"].set_color("#8B939E")
    ax.tick_params(color="#8B939E", labelcolor="#333", labelsize=8.5)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.5,
        -0.18,
        f"T#={metrics['n_pos']}  N={metrics['n_neg']}\n"
        f"TP {metrics['tp']}  FN {metrics['fn']}  FP {metrics['fp']}  TN {metrics['tn']}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555",
        linespacing=1.35,
    )


def plot_dev_test_panel(
    *,
    ctx: dict,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    ez_dev: np.ndarray,
    ez_test: np.ndarray,
    cutoff: float,
    out: Path,
    suptitle: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 10.4),
        layout="constrained",
        gridspec_kw={"width_ratios": [3.55, 0.95]},
    )
    for split, idx, ez, ax_s, ax_b, seed in (
        ("dev", dev_idx, ez_dev, axes[0, 0], axes[0, 1], 0),
        ("test", test_idx, ez_test, axes[1, 0], axes[1, 1], 1),
    ):
        labels = ctx["label_arr"][idx]
        df = _cohort_frame(ctx, idx, labels)
        call = np.nanmax(ez, axis=0) > cutoff
        met = detection_metrics(labels, call)
        plot_fixed_ez_scatter(
            df,
            ez,
            None,
            cutoff=cutoff,
            seed=seed,
            target_mask=_target_mask_from_labels(labels.tolist()),
            title=f"{split}  n={idx.size}  (T#={met['n_pos']}, Normal={met['n_neg']})",
            ax=ax_s,
        )
        _plot_metric_bars(ax_b, met, f"{split}  Sens / Spec / PPV")
        console.print(
            f"  {split}: Sens={met['sens']:.3f} Spec={met['spec']:.3f} "
            f"PPV={met['ppv']:.3f}  TP/FN/FP/TN="
            f"{met['tp']}/{met['fn']}/{met['fp']}/{met['tn']}"
        )
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    console.print(f"  wrote {out}")


def plot_signal_ratio_grouped(
    df: pd.DataFrame,
    out: Path,
    cutoff: float,
    n_repeats: int,
    pool_n: int,
    ref_n: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    purity = pd.to_numeric(df["purity"], errors="coerce")
    lo = (purity < PURITY_LO).to_numpy()
    is_dev = df["set"].to_numpy() == "dev"
    is_t = np.array([label_to_target_chr(x) is not None for x in df["label"]])
    is_norm = df["label"].to_numpy() == "Normal"
    is_unk = df["label"].astype(str).to_numpy() == "Unknown"
    groups = [
        (is_dev & is_t & ~lo, RED, "o", 36, f"dev T# (purity≥{PURITY_LO:g})", 4),
        (is_dev & is_t & lo, BLUE, "D", 48, f"dev T# (purity<{PURITY_LO:g})", 5),
        ((~is_dev) & is_t & ~lo, ORANGE, "o", 38, f"test T# (purity≥{PURITY_LO:g})", 4),
        ((~is_dev) & is_t & lo, PURPLE, "D", 50, f"test T# (purity<{PURITY_LO:g})", 5),
        ((~is_dev) & is_unk, TEAL, "^", 34, "test Unknown", 3),
        ((~is_dev) & is_norm, GRAY, "s", 26, "test Normal", 3),
    ]
    for mask, color, marker, size, label, z in groups:
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "ff_before_mq"],
            df.loc[mask, "signal_ratio"],
            s=size,
            alpha=0.88,
            c=color,
            marker=marker,
            edgecolors="#333",
            linewidths=0.35,
            label=label,
            zorder=z,
        )
    ax.axhline(0.5, color="#888", ls="--", lw=1.0, label="ratio=0.5")
    ax.set_xlabel("ff_before_mq")
    ax.set_ylabel(f"ezscore signal_ratio (cutoff={cutoff:g})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"{ref_n}+{ref_n} from dev Normal (n={pool_n}): "
        f"signal_ratio vs FF  ({n_repeats} repeats, {len(df)} samples)"
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    console.print(f"  wrote {out}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--ref40", default=str(DEFAULT_REF40), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--cutoff", default=DEFAULT_CUTOFF, show_default=True, type=float)
@click.option("--total-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option("--skip-plot3", is_flag=True, default=False)
@click.option("--skip-plot1", is_flag=True, default=False)
def main(
    parquet: str,
    meta: str,
    toxic: str,
    ref40: str,
    output_dir: str,
    cutoff: float,
    total_repeats: int,
    seed: int,
    n_jobs: int,
    skip_plot3: bool,
    skip_plot1: bool,
) -> None:
    out = Path(output_dir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    ctx = build_universe(Path(parquet), Path(meta), Path(toxic))
    ref_idx = _read_ref40(Path(ref40), ctx["sample_index"])
    ref_set = set(ref_idx.tolist())
    qc = _qc_ff_mask(ctx)
    lab = _norm_or_single(ctx)
    not_ref = np.array([i not in ref_set for i in range(len(ctx["ordered"]))])
    dev_idx = np.flatnonzero(
        (ctx["set_arr"] == "dev") & qc & lab & not_ref
    ).astype(np.int64)
    test_idx = np.flatnonzero((ctx["set_arr"] == "test") & qc & lab).astype(np.int64)
    console.print(
        f"ref40={ref_idx.size}  dev_eval={dev_idx.size}  test_eval={test_idx.size}  "
        f"cutoff={cutoff:g}  n_jobs={_resolve_n_jobs(n_jobs)}"
    )

    if not skip_plot1:
        console.rule("[cyan]Plot 1  stable ref40 + LOO ez")
        ez_dev = ez_stable_ref40_loo(ctx["arrays"], ref_idx, dev_idx)
        ez_test = ez_stable_ref40_loo(ctx["arrays"], ref_idx, test_idx)
        plot_dev_test_panel(
            ctx=ctx,
            dev_idx=dev_idx,
            test_idx=test_idx,
            ez_dev=ez_dev,
            ez_test=ez_test,
            cutoff=cutoff,
            out=figdir / "stable_ref40_loo_dev_test.png",
            suptitle="Stable ref40 as epi/z/ez (LOO ez)  ·  depth pass, Normal+single T#, FF≥0.01",
        )

    console.rule("[cyan]Plot 2  fixed E(μ)/E(σ) from 220 clean pool")
    params = load_fixed_params(out / "detection" / "fixed_epiz_ez_params.npz")
    ez_dev_f = fixed_ez_profiles_fully_fixed(ctx["arrays"], dev_idx, params)
    ez_test_f = fixed_ez_profiles_fully_fixed(ctx["arrays"], test_idx, params)
    plot_dev_test_panel(
        ctx=ctx,
        dev_idx=dev_idx,
        test_idx=test_idx,
        ez_dev=ez_dev_f,
        ez_test=ez_test_f,
        cutoff=cutoff,
        out=figdir / "fixed_ez_220_dev_test.png",
        suptitle="Fixed E(μ)/E(σ) from 220 clean-pool draws  ·  depth pass, Normal+single T#, FF≥0.01",
    )

    if skip_plot3:
        return
    console.rule("[cyan]Plot 3  40+40 signal_ratio")
    pool_idx = np.flatnonzero(
        (ctx["set_arr"] == "dev") & (ctx["label_arr"] == "Normal") & qc
    ).astype(np.int64)
    ref_n = 40
    if pool_idx.size < 2 * ref_n:
        ref_n = int(pool_idx.size // 2)
        console.print(
            f"  [yellow]dev Normal QC+FF pool n={pool_idx.size} < 80; "
            f"using {ref_n}+{ref_n}[/yellow]"
        )
    eval_dev_t = np.flatnonzero(
        (ctx["set_arr"] == "dev")
        & qc
        & np.array([label_to_target_chr(x) is not None for x in ctx["label_arr"]])
    ).astype(np.int64)
    eval_test = np.flatnonzero(
        (ctx["set_arr"] == "test")
        & qc
        & (
            (ctx["label_arr"] == "Normal")
            | (ctx["label_arr"] == "Unknown")
            | np.array([label_to_target_chr(x) is not None for x in ctx["label_arr"]])
        )
    ).astype(np.int64)
    eval_idx = np.concatenate([eval_dev_t, eval_test])
    flags = run_ref40_signal_ratio(
        arrays=ctx["arrays"],
        pool_idx=pool_idx,
        eval_idx=eval_idx,
        n_repeats=total_repeats,
        ref_n=ref_n,
        seed=seed,
        cutoff=cutoff,
        n_jobs=n_jobs,
    )
    df = _cohort_frame(ctx, eval_idx, ctx["label_arr"][eval_idx])
    df["signal_ratio"] = flags.mean(axis=0)
    plot_signal_ratio_grouped(
        df,
        figdir / "ref40_signal_ratio_vs_ff_devT_testAll.png",
        cutoff=cutoff,
        n_repeats=total_repeats,
        pool_n=int(pool_idx.size),
        ref_n=ref_n,
    )


if __name__ == "__main__":
    main()
