#!/usr/bin/env python3
"""Render PNG figures for the admittance-rule report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from common import (
    CHR_LIST,
    DEFAULT_INPUT_DIR,
    MAD_K,
    density_table,
    load_repeat_shards,
    load_universe,
    mad_z,
)

ANALYSIS = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/baseline96/analysis"
)
OUT_BASE = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
)
FIG_DIR = Path(__file__).resolve().parent / "report" / "figures"

PERFECT = "#7EB8BE"
FP = "#E07A3D"
FN = "#2E6F9E"
BAD = "#B42318"
OK = "#6B7280"
PROT = "#3D7A5A"
NEUT = "#9CA3AF"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def fig_fpfn_density() -> None:
    dens = pd.read_csv(ANALYSIS / "fp_fn_density.tsv", sep="\t")
    x = dens["fp_plus_fn"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, dens["perfect_density"], color=PERFECT, width=0.8, label="Perfect (FP+FN = 0)")
    ax.bar(x, dens["fp_density"], bottom=dens["perfect_density"], color=FP, width=0.8, label="FP share")
    ax.bar(
        x,
        dens["fn_density"],
        bottom=dens["perfect_density"] + dens["fp_density"],
        color=FN,
        width=0.8,
        label="FN share",
    )
    ax.axvline(0.5, color="#bbb", ls=":", lw=0.8)
    ax.axvline(4.5, color="#bbb", ls=":", lw=0.8)
    ax.set_xlabel("FP + FN per 40+40 repeat")
    ax.set_ylabel("Repeat density")
    ax.set_title("100k random 40+40 draws: error count is FN-dominated")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xticks(range(0, 11))
    ax.annotate("perfect", xy=(0, 0.12), ha="center", fontsize=8, color="#3a6")
    ax.annotate("bad (≥5)", xy=(5, 0.11), ha="center", fontsize=8, color=BAD)
    _save(fig, "fig1_fpfn_density.png")


def fig_lift_scatter() -> None:
    df = pd.read_csv(ANALYSIS / "toxic_protective.tsv", sep="\t")
    colors = {"toxic": BAD, "protective": PROT, "neutral": NEUT}
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for flag, sub in df.groupby("flag"):
        ax.scatter(
            sub["lift_perfect"],
            sub["lift_bad"],
            c=colors[flag],
            s=28,
            alpha=0.85,
            label=f"{flag} (n={len(sub)})",
            zorder=3 if flag != "neutral" else 2,
        )
        if flag != "neutral":
            top = sub.head(4)
            for _, r in top.iterrows():
                ax.annotate(
                    str(r["sample"]).replace("PTAY", ""),
                    (r["lift_perfect"], r["lift_bad"]),
                    fontsize=6.5,
                    xytext=(4, 3),
                    textcoords="offset points",
                )
    ax.axhline(1.0, color="#bbb", ls=":", lw=0.8)
    ax.axvline(1.0, color="#bbb", ls=":", lw=0.8)
    ax.set_xlabel("Lift in perfect draws  P(in 80 | perfect) / P(in 80)")
    ax.set_ylabel("Lift in bad draws  P(in 80 | bad) / P(in 80)")
    ax.set_title("Pool-sample enrichment in the 40+40 (role = either half)")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig2_lift_scatter.png")


def fig_features_vs_lift() -> None:
    df = pd.read_csv(ANALYSIS / "toxic_protective.tsv", sep="\t")
    colors = df["flag"].map({"toxic": BAD, "protective": PROT, "neutral": NEUT})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    specs = [
        ("ff_before_mq", "Fetal fraction (ff_before_mq)"),
        ("max_abs_pct_madz", "Max |percentage MAD-z|"),
        ("max_abs_intra_madz", "Max |z_intra MAD-z|"),
    ]
    for ax, (col, title) in zip(axes, specs):
        ax.scatter(df[col], df["lift_bad"], c=colors, s=22, alpha=0.85)
        ax.axhline(1.0, color="#bbb", ls=":", lw=0.8)
        ax.set_xlabel(title)
        if col != "ff_before_mq":
            ax.axvline(3.5, color="#888", ls="--", lw=0.8)
    axes[0].set_ylabel("Lift in bad draws")
    axes[1].set_title("Toxic lift is only partly explained by FF / MAD outliers")
    handles = [
        Patch(facecolor=BAD, label="toxic"),
        Patch(facecolor=PROT, label="protective"),
        Patch(facecolor=NEUT, label="neutral"),
    ]
    axes[2].legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig3_features_vs_lift.png")


def fig_ez_sd() -> None:
    byc = pd.read_csv(ANALYSIS / "set_features_by_class.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    bins_sd = np.linspace(1.2, 2.2, 40)
    bins_ff = np.linspace(0.0, 0.04, 40)
    for cls, color, label in (
        ("perfect", PERFECT, "perfect"),
        ("bad", BAD, "bad (FP+FN≥5)"),
    ):
        sub = byc.loc[byc["class"] == cls]
        axes[0].hist(sub["ez_sd_mean"], bins=bins_sd, density=True, alpha=0.65, color=color, label=label)
        axes[1].hist(sub["ff_80_std"], bins=bins_ff, density=True, alpha=0.65, color=color, label=label)
    axes[0].set_xlabel("Mean ez-ref SD across chr1–22")
    axes[0].set_ylabel("Density of repeats")
    axes[0].set_title("Bad sets inflate the ezscore denominator")
    axes[1].set_xlabel("FF std of the 80 refs")
    axes[1].set_ylabel("Density of repeats")
    axes[1].set_title("FF spread is a weaker separator")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig4_ez_sd_and_ff.png")


def fig_cliffs() -> None:
    cmp_ = pd.read_csv(ANALYSIS / "set_feature_compare.tsv", sep="\t")
    keep = [
        "ez_sd_chr14",
        "ez_sd_mean",
        "ez_sd_chr22",
        "ez_sd_max",
        "ff_80_std",
        "ff_80_mean",
        "n_fail_members",
        "pct_sd_mean",
        "pct_sd_chr14",
    ]
    sub = cmp_.loc[cmp_["metric"].isin(keep)].copy()
    sub["_abs"] = sub["cliffs_delta_bad_minus_perfect"].abs()
    sub = sub.sort_values("_abs")
    colors = np.where(sub["cliffs_delta_bad_minus_perfect"] > 0, BAD, FN)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh(sub["metric"], sub["cliffs_delta_bad_minus_perfect"], color=colors)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Cliff's δ  (positive = larger in bad sets)")
    ax.set_title("Set-level shift: ez-ref SD, not FF, separates bad 40+40")
    _save(fig, "fig5_cliffs_delta.png")


def fig_spearman() -> None:
    proof = pd.read_csv(ANALYSIS / "proof" / "proof_retrospective.tsv", sep="\t")
    rules = proof.loc[proof["label"].astype(str).str.startswith("rule:")].copy()
    rules["rule"] = rules["label"].str.replace("rule:", "", regex=False)
    order = [
        "ff_tail_5_95",
        "intra_mad_3_5",
        "mad_or_ff",
        "pct_mad_3_5",
        "mad_or_ff_and_toxic",
        "toxic_heldout",
        "toxic_keep80",
    ]
    rules["_ord"] = rules["rule"].map({r: i for i, r in enumerate(order)})
    rules = rules.sort_values("_ord")
    x = np.arange(len(rules))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(x - w / 2, rules["spearman_nfail_vs_fpfn"], w, color=BAD, label="QC rule")
    ax.bar(
        x + w / 2,
        rules["random_spearman_mean"],
        w,
        color=OK,
        label="Matched-N random drop",
        yerr=rules["random_spearman_sd"].fillna(0),
        capsize=2,
        ecolor="#555",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rules["rule"], rotation=25, ha="right")
    ax.set_ylabel("Spearman ρ (n fail members vs FP+FN)")
    ax.set_title("Membership toxicity tracks errors; MAD/FF filters do so only weakly")
    ax.legend(frameon=False)
    ax.axhline(0, color="#888", lw=0.6)
    _save(fig, "fig6_spearman_vs_random.png")


def _jitter(n: int, seed: int, width: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def _mad_fence(values: np.ndarray, k: float = 3.5) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    half = k * mad / MAD_K if mad > 0 else 0.0
    return med, med - half, med + half


def _box_strip_star(
    ax,
    prot: np.ndarray,
    ok: np.ndarray,
    toxic: float,
    *,
    seed: int,
    ylabel: str,
    madz: float,
    show_legend: bool = False,
) -> None:
    data = [prot[np.isfinite(prot)], ok[np.isfinite(ok)]]
    bp = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222", "lw": 1.2},
        whiskerprops={"color": "#444", "lw": 0.8},
        capprops={"color": "#444", "lw": 0.8},
        boxprops={"edgecolor": "#444", "lw": 0.8},
    )
    bp["boxes"][0].set_facecolor(PROT)
    bp["boxes"][0].set_alpha(0.35)
    bp["boxes"][1].set_facecolor(OK)
    bp["boxes"][1].set_alpha(0.35)
    ax.scatter(
        1 + _jitter(data[0].size, seed),
        data[0],
        s=14,
        c=PROT,
        alpha=0.75,
        zorder=3,
        edgecolors="none",
        label="protective",
    )
    ax.scatter(
        2 + _jitter(data[1].size, seed + 1),
        data[1],
        s=14,
        c=OK,
        alpha=0.75,
        zorder=3,
        edgecolors="none",
        label="OK / neutral",
    )
    ax.scatter(
        [3],
        [toxic],
        marker="*",
        s=180,
        c=BAD,
        zorder=5,
        edgecolors="#4a0d0d",
        linewidths=0.4,
        label="toxic example",
    )
    background = np.concatenate(data)
    _, lo, hi = _mad_fence(background)
    ax.axhline(lo, color="#888", ls="--", lw=0.8, zorder=1)
    ax.axhline(hi, color="#888", ls="--", lw=0.8, zorder=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["protective", "OK / neutral", "toxic"])
    ax.set_xlim(0.45, 3.55)
    ax.set_ylabel(ylabel)
    ax.annotate(
        f"MAD-z = {madz:+.1f}",
        xy=(3, toxic),
        xytext=(8, 8 if toxic >= np.median(background) else -14),
        textcoords="offset points",
        fontsize=8,
        color=BAD,
        fontweight="bold",
    )
    if show_legend:
        ax.legend(frameon=False, fontsize=7.5, loc="best")


def fig_toxic_mad_boxplot() -> None:
    """Top toxic samples vs OK/protective on the chromosome that fails MAD."""
    flags = pd.read_csv(ANALYSIS / "toxic_protective.tsv", sep="\t")
    flag_map = dict(zip(flags["sample"].astype(str), flags["flag"].astype(str)))
    ctx = load_universe(DEFAULT_INPUT_DIR)
    pool_idx = ctx["ref_pool_idx"]
    names = np.array([ctx["universe"][i] for i in pool_idx], dtype=object)
    hypo = ctx["ep_arrays"][0][:, pool_idx]
    hyper = ctx["ep_arrays"][1][:, pool_idx]
    pct = ctx["z_array"][:, pool_idx]
    pct_z = mad_z(pct, axis=1)
    hypo_z = mad_z(hypo, axis=1)
    hyper_z = mad_z(hyper, axis=1)
    intra_z = np.where(np.abs(hypo_z) >= np.abs(hyper_z), hypo_z, hyper_z)
    flag_arr = np.array([flag_map.get(str(s), "neutral") for s in names])
    prot_m = flag_arr == "protective"
    ok_m = flag_arr == "neutral"

    examples = [
        ("PTAY0614P10S1", "chr14"),
        ("PTAY0503P7H1", "chr15"),
        ("PTAY1000P6S1", None),
    ]
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}
    fig, axes = plt.subplots(len(examples), 2, figsize=(8.8, 8.6), squeeze=False)
    for row, (sample, chr_name) in enumerate(examples):
        j = int(np.flatnonzero(names == sample)[0])
        if chr_name is None:
            ci = int(np.nanargmax(np.abs(intra_z[:, j])))
            chr_name = CHR_LIST[ci]
        else:
            ci = chr_index[chr_name]
        use_hypo = abs(hypo_z[ci, j]) >= abs(hyper_z[ci, j])
        intra_vals = hypo[ci] if use_hypo else hyper[ci]
        intra_label = "hypo z_intra" if use_hypo else "hyper z_intra"
        intra_mad = float(hypo_z[ci, j] if use_hypo else hyper_z[ci, j])
        _box_strip_star(
            axes[row, 0],
            pct[ci, prot_m],
            pct[ci, ok_m],
            float(pct[ci, j]),
            seed=10 * row,
            ylabel=f"{chr_name} percentage",
            madz=float(pct_z[ci, j]),
            show_legend=(row == 0),
        )
        _box_strip_star(
            axes[row, 1],
            intra_vals[prot_m],
            intra_vals[ok_m],
            float(intra_vals[j]),
            seed=10 * row + 5,
            ylabel=f"{chr_name} {intra_label}",
            madz=intra_mad,
            show_legend=False,
        )
        axes[row, 0].set_title(sample, loc="left", fontsize=10)
    fig.suptitle(
        "Top toxic samples vs OK / protective on their outlier chromosome\n"
        "Dashed lines: MAD-z = ±3.5 vs the OK+protective background",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig9_toxic_mad_boxplot.png")


def fig_redraw() -> None:
    pairs = [
        ("baseline 96", OUT_BASE / "baseline96"),
        ("admitted (drop 16 toxic)", OUT_BASE / "admitted"),
        ("random drop of 16", OUT_BASE / "random_n"),
    ]
    colors = ["#4B5563", PROT, OK]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = []
    for (name, path), color in zip(pairs, colors):
        data = load_repeat_shards(path)
        dens = density_table(data["fp"], data["fn"], data["fp_plus_fn"])
        ax.plot(
            dens["fp_plus_fn"],
            dens["density"],
            marker="o",
            ms=4,
            color=color,
            label=f"{name}\nperfect={float((data['fp_plus_fn']==0).mean()):.1%}",
        )
        rows.append(
            {
                "pool": name,
                "frac_perfect": float((data["fp_plus_fn"] == 0).mean()),
                "mean_fpfn": float(data["fp_plus_fn"].mean()),
            }
        )
    ax.set_xlabel("FP + FN per repeat")
    ax.set_ylabel("Repeat density")
    ax.set_title("Prospective 40+40 redraw: toxic filter vs size-matched random drop")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(range(0, 11))
    _save(fig, "fig7_redraw_density.png")

    sumdf = pd.DataFrame(rows)
    short = ["baseline 96", "drop 16 toxic", "random drop 16"]
    fig2, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    axes[0].bar(short, sumdf["frac_perfect"], color=colors)
    axes[0].set_ylabel("Fraction perfect (FP+FN = 0)")
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(sumdf["frac_perfect"]):
        axes[0].text(i, v + 0.03, f"{v:.1%}", ha="center", fontsize=8)
    axes[1].bar(short, sumdf["mean_fpfn"], color=colors)
    axes[1].set_ylabel("Mean FP+FN")
    for i, v in enumerate(sumdf["mean_fpfn"]):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].set_title("Perfect rate")
    axes[1].set_title("Error count")
    fig2.tight_layout()
    _save(fig2, "fig8_redraw_bars.png")


def main() -> None:
    fig_fpfn_density()
    fig_lift_scatter()
    fig_features_vs_lift()
    fig_ez_sd()
    fig_cliffs()
    fig_spearman()
    fig_toxic_mad_boxplot()
    fig_redraw()


if __name__ == "__main__":
    main()
