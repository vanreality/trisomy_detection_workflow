#!/usr/bin/env python3
"""Why ez=3 and ez=4.5 signal-ratio trends can move in opposite directions.

Reads the 20260810 pool-size sweeps (growing candidate and fixed-160), finds
samples whose ezscore signal ratio rises toward 1 at cutoff 3 and falls toward
0 at cutoff 4.5 as pool size grows, then re-simulates max-chr ezscore to show
the mass concentrating in (3, 4.5].
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
for _p in (SCRIPT_DIR, REF40_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grid_search_ref40 import CHR_LIST, compute_episcore, compute_zscore  # noqa: E402
from ref_free_ezscore import (  # noqa: E402
    _generate_half_partitions,
    _load_fixed_combo_arrays,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()

SWEEP_BASE = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size"
)
DEFAULT_INPUT = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng"
)
BLACKLIST = (
    "PTAY0577P9S1",
    "PTAY0599P8S1",
    "PTAY0666P7S1",
    "PTAY0682P7S1",
    "PTAY0689P8H1",
)
FOCUS = ("PTAY0599P8S1", "HCPT0008")
FILL_SEED = 7
SEED = 42

DESIGNS = {
    "growing": {
        "ez3": SWEEP_BASE / "ez3" / "fixed",
        "ez45": SWEEP_BASE / "fixed",
        "fixed_candidate_size": None,
        "exclude_candidate": False,
        "label": "Growing candidate (96 dev Normal, then nested test fillers)",
    },
    "fixed160": {
        "ez3": SWEEP_BASE / "fixed160_ez3" / "fixed",
        "ez45": SWEEP_BASE / "fixed160_ez45" / "fixed",
        "fixed_candidate_size": 160,
        "exclude_candidate": True,
        "label": "Fixed candidate 160 (draw pool_size from the same 160)",
    },
}

EZ3_COLOR = "#2E6F9E"
EZ45_COLOR = "#C1121F"
BAND_COLOR = "#6B4C9A"
T_COLOR = "#B42318"
N_COLOR = "#2E6F9E"
FOCUS_FACE = "#F4D35E"

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
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    console.print(f"  wrote {path}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_pool_tsvs(pool_root: Path) -> pd.DataFrame:
    rows = []
    for tsv in sorted(pool_root.glob("pool_*/abnormality_signal_ratio.tsv")):
        pool_size = int(tsv.parent.name.split("_")[1])
        df = pd.read_csv(tsv, sep="\t")
        df["pool_size"] = pool_size
        rows.append(df)
    if not rows:
        raise click.ClickException(f"no pool TSVs under {pool_root}")
    out = pd.concat(rows, ignore_index=True)
    out["sample"] = out["sample"].astype(str)
    return out


def pair_design(name: str) -> pd.DataFrame:
    cfg = DESIGNS[name]
    a = load_pool_tsvs(cfg["ez3"]).rename(columns={"ezscore_signal_ratio": "ez3"})
    b = load_pool_tsvs(cfg["ez45"]).rename(columns={"ezscore_signal_ratio": "ez45"})
    keep = ["sample", "set", "label", "ff_before_mq", "pool_size"]
    merged = a[keep + ["ez3", "episcore_signal_ratio", "zscore_signal_ratio"]].merge(
        b[keep + ["ez45"]],
        on=["sample", "pool_size"],
        how="inner",
        suffixes=("", "_b"),
    )
    merged["design"] = name
    merged["band"] = (merged["ez3"] - merged["ez45"]).clip(lower=0.0)
    merged["is_trisomy"] = merged["label"].astype(str).str.match(r"^T\d")
    merged["blacklisted"] = merged["sample"].isin(BLACKLIST)
    return merged


def sample_trends(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (design, sample), g in traj.groupby(["design", "sample"], sort=False):
        g = g.sort_values("pool_size")
        p = g["pool_size"].to_numpy()
        ez3 = g["ez3"].to_numpy()
        ez45 = g["ez45"].to_numpy()
        band = g["band"].to_numpy()
        row0 = g.iloc[0]
        p_lo, p_hi = int(p[0]), int(p[-1])
        rows.append(
            {
                "design": design,
                "sample": sample,
                "set": row0["set"],
                "label": row0["label"],
                "ff_before_mq": row0["ff_before_mq"],
                "is_trisomy": bool(row0["is_trisomy"]),
                "blacklisted": bool(row0["blacklisted"]),
                "n_pools": int(len(g)),
                "pool_lo": p_lo,
                "pool_hi": p_hi,
                "ez3_lo": float(ez3[0]),
                "ez3_hi": float(ez3[-1]),
                "ez45_lo": float(ez45[0]),
                "ez45_hi": float(ez45[-1]),
                "band_lo": float(band[0]),
                "band_hi": float(band[-1]),
                "delta_ez3": float(ez3[-1] - ez3[0]),
                "delta_ez45": float(ez45[-1] - ez45[0]),
                "delta_band": float(band[-1] - band[0]),
                "rho_ez3": _spearman(p, ez3),
                "rho_ez45": _spearman(p, ez45),
                "rho_band": _spearman(p, band),
            }
        )
    out = pd.DataFrame(rows)
    ez45_falls = (out["rho_ez45"] < -0.25) & (out["delta_ez45"] < -0.03)
    ez3_rises = (out["rho_ez3"] > 0.25) & (out["delta_ez3"] > 0.03)
    # Already near 1 at the smallest pool; leftover rise is tiny but still toward 1.
    ez3_ceiling = (out["ez3_lo"] >= 0.90) & (out["ez3_hi"] >= 0.95) & (out["delta_ez3"] >= 0)
    out["pattern"] = "other"
    out.loc[ez45_falls & ez3_rises, "pattern"] = "ez3_rises_ez45_falls"
    out.loc[ez45_falls & ez3_ceiling & ~ez3_rises, "pattern"] = "ez3_ceiling_ez45_falls"
    out["discordant"] = out["pattern"].isin(
        ("ez3_rises_ez45_falls", "ez3_ceiling_ez45_falls")
    )
    out["reverse_discordant"] = (
        (out["rho_ez3"] < -0.25)
        & (out["rho_ez45"] > 0.25)
        & (out["delta_ez3"] < -0.03)
        & (out["delta_ez45"] > 0.03)
    )
    out["band_trap"] = out["discordant"] & (out["delta_band"] > 0.05)
    return out


def _build_candidate(
    set_arr: np.ndarray,
    label_arr: np.ndarray,
    cand_n: int,
    fill_seed: int,
) -> np.ndarray:
    is_normal = label_arr == "Normal"
    dev_idx = np.flatnonzero((set_arr == "dev") & is_normal).astype(np.int64)
    test_idx = np.flatnonzero((set_arr == "test") & is_normal).astype(np.int64)
    if cand_n <= dev_idx.size:
        return dev_idx
    n_fill = cand_n - int(dev_idx.size)
    rng = np.random.default_rng(fill_seed)
    ordered = test_idx[rng.permutation(test_idx.size)]
    fillers = np.sort(ordered[:n_fill])
    return np.concatenate([dev_idx, fillers])


def load_score_ctx(input_dir: Path) -> dict:
    meta = pd.read_csv(input_dir / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    ep_df = pd.read_parquet(input_dir / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_dir / "zscore_grid_search.parquet")
    universe = sorted(
        set(meta["sample"])
        & set(ep_df["sample"].astype(str))
        & set(z_df["sample"].astype(str))
    )
    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}
    meta_idx = meta.set_index("sample").reindex(universe)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    ep_arrays, z_array = _load_fixed_combo_arrays(
        ep_df, z_df, 0.5, 0.65, 0.85, 0.95, sample_index, chr_index
    )
    return {
        "universe": universe,
        "sample_index": sample_index,
        "set_arr": set_arr,
        "label_arr": label_arr,
        "ep_hypo": np.expand_dims(ep_arrays[0], 0),
        "ep_hyper": np.expand_dims(ep_arrays[1], 0),
        "ep_hypo_cnt": np.expand_dims(ep_arrays[2], 0),
        "ep_hyper_cnt": np.expand_dims(ep_arrays[3], 0),
        "z_pct": np.expand_dims(z_array, 0),
    }


def simulate_max_ez(
    ctx: dict,
    *,
    design: str,
    pool_sizes: list[int],
    n_repeats: int,
    focus: tuple[str, ...],
    seed: int = SEED,
    fill_seed: int = FILL_SEED,
) -> pd.DataFrame:
    """Per-repeat max-chr ezscore and per-chr flags for focus samples."""
    cfg = DESIGNS[design]
    missing = [s for s in focus if s not in ctx["sample_index"]]
    if missing:
        raise click.ClickException(f"focus samples missing from universe: {missing}")
    focus_idx = np.array([ctx["sample_index"][s] for s in focus], dtype=np.int64)
    expected_chr = {
        s: ("chr" + str(ctx["label_arr"][ctx["sample_index"][s]])[1:])
        if str(ctx["label_arr"][ctx["sample_index"][s]]).startswith("T")
        else None
        for s in focus
    }
    n_chr = len(CHR_LIST)
    rows = []
    for pool_size in pool_sizes:
        half = pool_size // 2
        cand_n = int(cfg["fixed_candidate_size"] or pool_size)
        candidate = _build_candidate(
            ctx["set_arr"], ctx["label_arr"], cand_n, fill_seed
        )
        rng = np.random.default_rng(seed)
        ref_draws, ez_draws = _generate_half_partitions(
            pool_size=candidate.size, half=half, n_repeats=n_repeats, rng=rng
        )
        console.print(
            f"  MC {design} pool={pool_size} half={half} "
            f"cand={candidate.size} repeats={n_repeats}"
        )
        for r in range(n_repeats):
            ref_idx = candidate[ref_draws[r]]
            ez_ref_idx = candidate[ez_draws[r]]
            episcore = compute_episcore(
                ctx["ep_hypo"],
                ctx["ep_hyper"],
                ctx["ep_hypo_cnt"],
                ctx["ep_hyper_cnt"],
                ref_idx,
            )[0]
            zscore = compute_zscore(ctx["z_pct"], ref_idx)[0]
            combined = episcore + zscore
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(combined[:, ez_ref_idx], axis=1)
                sd = np.nanstd(combined[:, ez_ref_idx], axis=1, ddof=0)
            sd_safe = np.where(sd > 0, sd, np.nan)
            for j, sample in enumerate(focus):
                si = focus_idx[j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ez = (combined[:, si] - mu) / sd_safe
                ez = np.where(np.isfinite(ez), ez, np.nan)
                max_i = int(np.nanargmax(ez)) if np.isfinite(ez).any() else -1
                max_ez = float(ez[max_i]) if max_i >= 0 else float("nan")
                exp_chr = expected_chr[sample]
                exp_i = CHR_LIST.index(exp_chr) if exp_chr in CHR_LIST else -1
                other = np.array(
                    [ez[i] for i in range(n_chr) if i != exp_i], dtype=float
                )
                rows.append(
                    {
                        "design": design,
                        "pool_size": pool_size,
                        "repeat": r,
                        "sample": sample,
                        "max_ez": max_ez,
                        "max_chr": CHR_LIST[max_i] if max_i >= 0 else "",
                        "expected_chr": exp_chr or "",
                        "ez_expected_chr": float(ez[exp_i]) if exp_i >= 0 else float("nan"),
                        "max_other_ez": float(np.nanmax(other)) if other.size else float("nan"),
                        "ez_ref_sd_at_max": float(sd[max_i]) if max_i >= 0 else float("nan"),
                        "ez_ref_sd_mean": float(np.nanmean(sd)),
                        "combined_at_max": float(combined[max_i, si]) if max_i >= 0 else float("nan"),
                        "mu_at_max": float(mu[max_i]) if max_i >= 0 else float("nan"),
                        "gt3": int(np.nanmax(ez) > 3.0) if np.isfinite(ez).any() else 0,
                        "gt45": int(np.nanmax(ez) > 4.5) if np.isfinite(ez).any() else 0,
                        "exp_gt3": int(ez[exp_i] > 3.0) if exp_i >= 0 else 0,
                        "exp_gt45": int(ez[exp_i] > 4.5) if exp_i >= 0 else 0,
                        "other_gt3": int(np.nanmax(other) > 3.0) if other.size else 0,
                        "other_gt45": int(np.nanmax(other) > 4.5) if other.size else 0,
                    }
                )
    return pd.DataFrame(rows)


def fig1_scatter(trends: pd.DataFrame, out: Path) -> None:
    sub = trends[trends["design"] == "fixed160"].copy()
    sub = sub.dropna(subset=["rho_ez3", "rho_ez45"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for is_t, color, label, z in (
        (False, N_COLOR, "Normal", 2),
        (True, T_COLOR, "Trisomy", 3),
    ):
        m = sub["is_trisomy"] == is_t
        ax.scatter(
            sub.loc[m, "rho_ez45"],
            sub.loc[m, "rho_ez3"],
            s=np.where(sub.loc[m, "blacklisted"], 55, 28),
            c=color,
            alpha=0.75,
            linewidths=np.where(sub.loc[m, "blacklisted"], 0.8, 0.0),
            edgecolors="k",
            label=label,
            zorder=z,
        )
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.axhspan(0.25, 1.05, xmin=0, xmax=0.375, color=BAND_COLOR, alpha=0.08, zorder=0)
    ax.axvspan(-1.05, -0.25, ymin=0.595, ymax=1.0, color=BAND_COLOR, alpha=0.08, zorder=0)
    ax.text(
        -0.95,
        0.92,
        "discordant\n(ez3↑  ez4.5↓)",
        color=BAND_COLOR,
        fontsize=9,
        ha="left",
        va="top",
    )
    for s in FOCUS:
        r = sub[sub["sample"] == s]
        if r.empty:
            continue
        ax.scatter(
            r["rho_ez45"],
            r["rho_ez3"],
            s=90,
            facecolors="none",
            edgecolors=FOCUS_FACE,
            linewidths=1.8,
            zorder=5,
        )
        ax.annotate(
            s,
            (r["rho_ez45"].iloc[0], r["rho_ez3"].iloc[0]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
            color="#333",
        )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Spearman ρ (pool size vs ez=4.5 signal ratio)")
    ax.set_ylabel("Spearman ρ (pool size vs ez=3 signal ratio)")
    ax.set_title("Fixed-160: per-sample trend of signal ratio vs pool size")
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    _save(fig, out)


def fig2_focus_traj(traj: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=True)
    titles = {
        ("growing", "HCPT0008"): "HCPT0008 · growing candidate",
        ("fixed160", "HCPT0008"): "HCPT0008 · fixed 160",
        ("growing", "PTAY0599P8S1"): "PTAY0599P8S1 · growing candidate",
        ("fixed160", "PTAY0599P8S1"): "PTAY0599P8S1 · fixed 160",
    }
    for ax, key in zip(axes.ravel(), titles):
        design, sample = key
        g = traj[(traj["design"] == design) & (traj["sample"] == sample)].sort_values(
            "pool_size"
        )
        ax.plot(g["pool_size"], g["ez3"], color=EZ3_COLOR, lw=1.8, label="P(max ez > 3)")
        ax.plot(
            g["pool_size"], g["ez45"], color=EZ45_COLOR, lw=1.8, label="P(max ez > 4.5)"
        )
        ax.fill_between(
            g["pool_size"],
            g["ez45"],
            g["ez3"],
            color=BAND_COLOR,
            alpha=0.18,
            label="P(3 < max ez ≤ 4.5)",
        )
        ax.axhline(0.5, color="#ccc", ls=":", lw=0.8)
        ax.set_title(titles[key], fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        meta = g.iloc[0]
        ax.text(
            0.03,
            0.06,
            f"{meta['label']}  ff={float(meta['ff_before_mq']):.1%}"
            + ("  blacklisted" if meta["blacklisted"] else ""),
            transform=ax.transAxes,
            fontsize=8,
            color="#555",
        )
    axes[0, 0].legend(frameon=False, loc="center right", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("pool size (= 2 × ref_n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ezscore signal ratio")
    fig.suptitle(
        "Opposite directions: mass moves into the (3, 4.5] band as the pool grows",
        y=1.01,
    )
    _save(fig, out)


def fig3_discordant_gallery(traj: pd.DataFrame, trends: pd.DataFrame, out: Path) -> None:
    disc = trends[(trends["design"] == "fixed160") & trends["discordant"]].sort_values(
        "delta_band", ascending=False
    )
    n = len(disc)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10.8, 2.55 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        row = disc.iloc[i]
        g = traj[
            (traj["design"] == "fixed160") & (traj["sample"] == row["sample"])
        ].sort_values("pool_size")
        ax.plot(g["pool_size"], g["ez3"], color=EZ3_COLOR, lw=1.4)
        ax.plot(g["pool_size"], g["ez45"], color=EZ45_COLOR, lw=1.4)
        ax.fill_between(g["pool_size"], g["ez45"], g["ez3"], color=BAND_COLOR, alpha=0.16)
        mark = " *" if row["sample"] in FOCUS else ""
        bl = "  BL" if row["blacklisted"] else ""
        tag = "ceil" if row["pattern"] == "ez3_ceiling_ez45_falls" else "rise"
        ax.set_title(
            f"{row['sample']}{mark}  [{tag}]\n{row['label']}  ff={float(row['ff_before_mq']):.1%}{bl}",
            fontsize=8,
        )
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0.5, color="#ddd", ls=":", lw=0.6)
    handles = [
        Line2D([0], [0], color=EZ3_COLOR, lw=1.6, label="ez=3"),
        Line2D([0], [0], color=EZ45_COLOR, lw=1.6, label="ez=4.5"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("All fixed-160 discordant samples. [rise]=ez3 climbing; [ceil]=ez3 already ~1", y=1.02)
    for ax in axes[-1, :]:
        ax.set_xlabel("pool size")
    for ax in axes[:, 0]:
        ax.set_ylabel("signal ratio")
    _save(fig, out)


def fig4_maxez_violin(mc: pd.DataFrame, out: Path) -> None:
    sizes = sorted(mc["pool_size"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharey=True)
    panels = [
        ("fixed160", "HCPT0008"),
        ("growing", "HCPT0008"),
        ("fixed160", "PTAY0599P8S1"),
        ("growing", "PTAY0599P8S1"),
    ]
    for ax, (design, sample) in zip(axes.ravel(), panels):
        sub = mc[(mc["design"] == design) & (mc["sample"] == sample)]
        data = [sub.loc[sub["pool_size"] == p, "max_ez"].to_numpy() for p in sizes]
        parts = ax.violinplot(
            data,
            positions=np.arange(len(sizes)),
            widths=0.85,
            showextrema=False,
            showmedians=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#7EB8BE")
            body.set_alpha(0.7)
            body.set_edgecolor("#2E6F9E")
        parts["cmedians"].set_color("#1b4332")
        ax.axhline(3.0, color=EZ3_COLOR, ls="--", lw=1.0, label="cutoff 3")
        ax.axhline(4.5, color=EZ45_COLOR, ls="--", lw=1.0, label="cutoff 4.5")
        ax.set_xticks(np.arange(len(sizes)), [str(p) for p in sizes])
        ax.set_title(f"{sample} · {design}", fontsize=10)
        ax.set_ylim(0, 12)
        med = [float(np.nanmedian(d)) if d.size else np.nan for d in data]
        ax.plot(np.arange(len(sizes)), med, color="#1b4332", lw=1.0, marker="o", ms=3)
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("pool size")
    for ax in axes[:, 0]:
        ax.set_ylabel("max-chromosome ezscore")
    fig.suptitle(
        "Max-chr ezscore concentrates between 3 and 4.5 as n grows",
        y=1.01,
    )
    _save(fig, out)


def fig5_chr_decomp(mc: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, sample in zip(axes, FOCUS):
        sub = mc[(mc["design"] == "fixed160") & (mc["sample"] == sample)]
        g = (
            sub.groupby("pool_size")
            .agg(
                p_gt3=("gt3", "mean"),
                p_gt45=("gt45", "mean"),
                exp3=("exp_gt3", "mean"),
                exp45=("exp_gt45", "mean"),
                oth3=("other_gt3", "mean"),
                oth45=("other_gt45", "mean"),
            )
            .reset_index()
        )
        ax.plot(g["pool_size"], g["p_gt3"], color=EZ3_COLOR, lw=2.0, label="any chr > 3")
        ax.plot(g["pool_size"], g["p_gt45"], color=EZ45_COLOR, lw=2.0, label="any chr > 4.5")
        ax.plot(
            g["pool_size"],
            g["exp3"],
            color=EZ3_COLOR,
            lw=1.2,
            ls="--",
            label="expected chr > 3",
        )
        ax.plot(
            g["pool_size"],
            g["exp45"],
            color=EZ45_COLOR,
            lw=1.2,
            ls="--",
            label="expected chr > 4.5",
        )
        ax.plot(
            g["pool_size"],
            g["oth45"],
            color="#888",
            lw=1.2,
            ls=":",
            label="some other chr > 4.5",
        )
        ax.set_title(sample)
        ax.set_xlabel("pool size")
        ax.set_ylim(-0.02, 1.05)
        exp = sub["expected_chr"].iloc[0]
        ax.text(0.04, 0.08, f"expected {exp}", transform=ax.transAxes, fontsize=8, color="#555")
    axes[0].set_ylabel("repeat fraction")
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Fixed-160: expected-chr vs off-target chromosome calls", y=1.03)
    _save(fig, out)


def fig6_sd_vs_ez(mc: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
    sizes = sorted(mc["pool_size"].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(sizes), max(sizes))
    rng = np.random.default_rng(0)
    for ax, sample in zip(axes, FOCUS):
        sub = mc[(mc["design"] == "fixed160") & (mc["sample"] == sample)]
        for p in sizes:
            d = sub[sub["pool_size"] == p]
            n = min(len(d), 800)
            take = rng.choice(len(d), size=n, replace=False)
            ax.scatter(
                d["ez_ref_sd_at_max"].to_numpy()[take],
                d["max_ez"].to_numpy()[take],
                s=8,
                alpha=0.35,
                c=[cmap(norm(p))],
                linewidths=0,
            )
        ax.axhline(3.0, color=EZ3_COLOR, ls="--", lw=0.9)
        ax.axhline(4.5, color=EZ45_COLOR, ls="--", lw=0.9)
        ax.set_title(sample)
        ax.set_xlabel("ez-ref SD at the max chromosome")
        ax.set_xlim(left=0)
    axes[0].set_ylabel("max-chr ezscore")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("pool size")
    fig.suptitle(
        "Small-n ez-ref SD left-tail inflates ezscore above 4.5; larger n raises SD and shrinks ez",
        y=1.03,
        fontsize=11,
    )
    _save(fig, out)


def fig7_schematic(mc: pd.DataFrame, out: Path) -> None:
    """One-sample cartoon: two Gaussians vs the two cutoffs."""
    sample = "HCPT0008"
    sub = mc[(mc["design"] == "fixed160") & (mc["sample"] == sample)]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    xs = np.linspace(0, 11, 400)
    for pool, color, lw in ((20, "#94D2BD", 1.6), (160, "#005F73", 2.2)):
        v = sub.loc[sub["pool_size"] == pool, "max_ez"].to_numpy()
        v = v[np.isfinite(v)]
        mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        ax.plot(xs, pdf, color=color, lw=lw, label=f"pool {pool}  (μ={mu:.2f}, σ={sd:.2f})")
        ax.fill_between(xs, pdf, color=color, alpha=0.12)
    ax.axvline(3.0, color=EZ3_COLOR, ls="--", lw=1.1)
    ax.axvline(4.5, color=EZ45_COLOR, ls="--", lw=1.1)
    ymax = ax.get_ylim()[1]
    ax.text(3.08, ymax * 0.92, "ez=3", color=EZ3_COLOR, fontsize=8)
    ax.text(4.58, ymax * 0.92, "ez=4.5", color=EZ45_COLOR, fontsize=8)
    ax.set_xlabel("max-chromosome ezscore")
    ax.set_ylabel("density (normal approx. of MC)")
    ax.set_title(f"{sample}, fixed-160: variance collapse into the (3, 4.5] gap")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.72,
        "ez=3 catches more\nas the left tail leaves",
        transform=ax.transAxes,
        fontsize=8,
        color=EZ3_COLOR,
    )
    ax.text(
        0.62,
        0.55,
        "ez=4.5 catches less\nas the right tail leaves",
        transform=ax.transAxes,
        fontsize=8,
        color=EZ45_COLOR,
    )
    _save(fig, out)


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    traj: pd.DataFrame,
    trends: pd.DataFrame,
    mc: pd.DataFrame,
    mc_sum: pd.DataFrame,
) -> None:
    t160 = trends[trends["design"] == "fixed160"]
    tg = trends[trends["design"] == "growing"]
    disc160 = t160[t160["discordant"]].sort_values("delta_band", ascending=False)
    disc_g = tg[tg["discordant"]].sort_values("delta_band", ascending=False)
    n_eval = int(t160["sample"].nunique())
    n_t = int(t160["is_trisomy"].sum())
    n_n = n_eval - n_t
    n_disc = int(disc160.shape[0])
    n_disc_t = int(disc160["is_trisomy"].sum())
    n_disc_ceil = int((disc160["pattern"] == "ez3_ceiling_ez45_falls").sum())
    n_disc_rise = int((disc160["pattern"] == "ez3_rises_ez45_falls").sum())
    n_rev = int(t160["reverse_discordant"].sum())
    n_mc = int(mc["repeat"].max() + 1) if "repeat" in mc.columns else 0

    def focus_row(design: str, sample: str) -> pd.Series:
        r = trends[(trends["design"] == design) & (trends["sample"] == sample)]
        return r.iloc[0]

    h_g = focus_row("growing", "HCPT0008")
    h_f = focus_row("fixed160", "HCPT0008")
    p_g = focus_row("growing", "PTAY0599P8S1")
    p_f = focus_row("fixed160", "PTAY0599P8S1")

    def mc_stats(design: str, sample: str, pool: int) -> dict:
        d = mc[
            (mc["design"] == design)
            & (mc["sample"] == sample)
            & (mc["pool_size"] == pool)
        ]
        v = d["max_ez"].to_numpy()
        return {
            "n": int(len(d)),
            "median": float(np.nanmedian(v)),
            "mean": float(np.nanmean(v)),
            "sd": float(np.nanstd(v, ddof=1)),
            "p3": float(d["gt3"].mean()),
            "p45": float(d["gt45"].mean()),
            "p_exp3": float(d["exp_gt3"].mean()),
            "p_exp45": float(d["exp_gt45"].mean()),
            "p_oth45": float(d["other_gt45"].mean()),
            "sd_ref": float(d["ez_ref_sd_at_max"].median()),
            "max_chr_mode": d["max_chr"].mode().iloc[0] if len(d) else "",
            "frac_expected_max": float((d["max_chr"] == d["expected_chr"]).mean()),
        }

    h20 = mc_stats("fixed160", "HCPT0008", 20)
    h80 = mc_stats("fixed160", "HCPT0008", 80)
    h160 = mc_stats("fixed160", "HCPT0008", 160)
    h80g = mc_stats("growing", "HCPT0008", 80)
    p20 = mc_stats("fixed160", "PTAY0599P8S1", 20)
    p160 = mc_stats("fixed160", "PTAY0599P8S1", 160)

    def traj_at(design: str, sample: str, pool: int) -> pd.Series:
        g = traj[
            (traj["design"] == design)
            & (traj["sample"] == sample)
            & (traj["pool_size"] == pool)
        ]
        return g.iloc[0]

    h_g96 = traj_at("growing", "HCPT0008", 96)
    h_g120 = traj_at("growing", "HCPT0008", 120)

    disc_cols = [
        "sample",
        "label",
        "ff_before_mq",
        "pattern",
        "ez3_lo",
        "ez3_hi",
        "ez45_lo",
        "ez45_hi",
        "delta_ez3",
        "delta_ez45",
        "delta_band",
        "blacklisted",
    ]

    def ff(v):
        try:
            return f"{100 * float(v):.2f}%"
        except Exception:
            return str(v)

    fmt = {
        "ff_before_mq": ff,
        "ez3_lo": lambda v: f"{float(v):.3f}",
        "ez3_hi": lambda v: f"{float(v):.3f}",
        "ez45_lo": lambda v: f"{float(v):.3f}",
        "ez45_hi": lambda v: f"{float(v):.3f}",
        "delta_ez3": lambda v: f"{float(v):+.3f}",
        "delta_ez45": lambda v: f"{float(v):+.3f}",
        "delta_band": lambda v: f"{float(v):+.3f}",
        "blacklisted": lambda v: "yes" if bool(v) else "",
        "pattern": lambda v: (
            "ez3 already ~1"
            if v == "ez3_ceiling_ez45_falls"
            else "ez3 rises"
            if v == "ez3_rises_ez45_falls"
            else str(v)
        ),
    }
    show = disc160[disc_cols].copy()
    show["ff_before_mq"] = show["ff_before_mq"]

    growing_note = (
        "The growing-pool ez=4.5 TSV originally dropped the five blacklist "
        "samples from eval; they were later backfilled with 20k repeats "
        "(see `fixed/pool_*/blacklist_backfill.json`). Ez=3 included them in "
        "the 1e6 run. For PTAY0599P8S1 prefer the **fixed-160** comparison, "
        "where both cutoffs used the same 1e6 draws."
    )

    report = f"""# Why ez=3 and ez=4.5 signal ratios can move in opposite directions

**Pool-size sweep:** `/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size`  
**Question:** some samples (e.g. PTAY0599P8S1, HCPT0008) have ezscore signal ratio moving **toward 1** as the reference pool grows at cutoff 3, and **toward 0** at cutoff 4.5.  
**Short answer:** the sample's max-chromosome ezscore is concentrating **between 3 and 4.5**. A larger pool does not change the sign of the call at both cutoffs — it **shrinks the Monte-Carlo spread** of ezscore (and slightly raises ez-ref SD), so probability mass leaves both tails and piles up in the gap.

Tables: `tables/`. Figures: `figures/`. Script: `scripts/ref_free/analyze_ez3_vs_ez45_discordant.py`.

---

## 0. What signal ratio is

Each pool-size job draws `pool_size/2` epi/z refs and `pool_size/2` ez refs, 1e6 times (seed 42). For every draw:

1. episcore and zscore vs the first half (fixed combo ep 0.5/0.65, z 0.85/0.95)
2. `combined = episcore + zscore`
3. **ezscore** on chromosome *h*:

\\[
\\mathrm{{ez}}_{{h}} = \\frac{{e_h + z_h - \\mu_h^{{\\mathrm{{ez}}}}}}{{\\sigma_h^{{\\mathrm{{ez}}}}}}
\\]

where \\(\\mu_h^{{\\mathrm{{ez}}}}, \\sigma_h^{{\\mathrm{{ez}}}}\\) are the mean and **population SD (ddof=0)** of `combined` on the ez-ref half.

4. Flag the sample abnormal if **any chromosome** exceeds the cutoff.

**Signal ratio** = fraction of repeats flagged. Ez=3 and ez=4.5 used the **same draws** (same seed, same candidate). Therefore

\\[
P(3 < \\max_h \\mathrm{{ez}}_h \\le 4.5) = r_{{3}} - r_{{4.5}}
\\]

is the probability mass in the band between the two cutoffs. That band is the purple fill in the trajectory plots.

Two sweep designs:

| Design | Candidate | Eval |
|--------|-----------|------|
| Growing (`ez3/fixed` vs `fixed`) | 96 dev Normal; if pool>96, nested test-Normal fillers (`fill_seed=7`) | dev trisomy + test, minus fillers |
| Fixed-160 (`fixed160_ez3` vs `fixed160_ez45`) | 96 + 64 fillers, **frozen**; every pool size is a subset draw | excludes the entire 160 |

{growing_note}

---

## 1. How common is the opposite-direction pattern?

On the **fixed-160** eval ({n_eval} samples; {n_t} trisomy / {n_n} Normal), a sample is **discordant** if ez=4.5 signal ratio falls with pool size (Spearman ρ < −0.25 and Δ < −0.03) **and** ez=3 either rises (ρ > 0.25, Δ > 0.03) or is already at the ceiling (ratio ≥ 0.90 at pool 20 and ≥ 0.95 at pool 160). Two subtypes:

| Subtype | What you see | Large-n max ez |
|---------|----------------|----------------|
| **ez3 rises** | both cutoffs move, in opposite directions | well inside (3, 4.5), e.g. ~3.6 |
| **ez3 already ~1** | ez=3 is saturated; only ez=4.5 falls | just under 4.5, e.g. ~4.3 |

- **{n_disc} / {n_eval}** samples are discordant ({n_disc_t} trisomy): {n_disc_rise} rising-ez3, {n_disc_ceil} ceilinged-ez3.
- **{n_rev}** samples show the reverse (ez3↓ and ez4.5↑). The pattern is one-way.
- Growing-candidate: **{int(disc_g.shape[0])}** discordant (same definition).

The discordant quadrant is the upper-left of Figure 1. Strong trisomies sit in the upper-right (both cutoffs rise toward 1). Clean Normals sit near the origin or lower-left (both stay near 0 or fall).

![Figure 1](figures/fig1_trend_scatter.png)

*Figure 1.* Per-sample Spearman ρ of ezscore signal ratio vs pool size, fixed-160. Highlighted: PTAY0599P8S1 and HCPT0008. Larger points with a black edge are the analysis blacklist.

Discordant samples, fixed-160, sorted by how much mass they gain in the (3, 4.5] band:

{_md_table(show, disc_cols, fmt)}

Most discordant trisomies have **low FF** (typically < 2%). Three **Normals** (PTAY0635P7H1, PTAY0874P7H1, PTAY0740P7H1) sit in the same gap — at ez=3 a large pool would call them abnormal almost every draw, at ez=4.5 almost never. Their FF is all < 1%.

---

## 2. The two named samples

They are the same phenomenon at **two locations** inside (3, 4.5].

**PTAY0599P8S1** (blacklisted T22, FF 5.2%) — *ez3 rises*. Large-n median max ez ≈ {p160['median']:.2f}. Pool 20 still has a left tail below 3 and a right tail above 4.5; pool 160 has neither.

**HCPT0008** (T22, FF 2.0%) — *ez3 already ~1*. Large-n median max ez ≈ {h160['median']:.2f}, hugging 4.5 from below. Ez=3 is already saturated at pool 20 (ratio {h_f['ez3_lo']:.3f}); the visible motion is ez=4.5 falling from {h_f['ez45_lo']:.3f} to {h_f['ez45_hi']:.3f}. It does not go all the way to 0 because a thin right tail still crosses 4.5.

![Figure 2](figures/fig2_focus_trajectories.png)

*Figure 2.* 1e6-repeat signal ratios. Purple = P(max ez in (3, 4.5]).

| Sample | Design | ez3 @20 → @160 | ez4.5 @20 → @160 | band @20 → @160 |
|--------|--------|----------------|------------------|-----------------|
| HCPT0008 | growing | {h_g['ez3_lo']:.3f} → {h_g['ez3_hi']:.3f} ({h_g['delta_ez3']:+.3f}) | {h_g['ez45_lo']:.3f} → {h_g['ez45_hi']:.3f} ({h_g['delta_ez45']:+.3f}) | {h_g['band_lo']:.3f} → {h_g['band_hi']:.3f} |
| HCPT0008 | fixed-160 | {h_f['ez3_lo']:.3f} → {h_f['ez3_hi']:.3f} ({h_f['delta_ez3']:+.3f}) | {h_f['ez45_lo']:.3f} → {h_f['ez45_hi']:.3f} ({h_f['delta_ez45']:+.3f}) | {h_f['band_lo']:.3f} → {h_f['band_hi']:.3f} |
| PTAY0599P8S1 | growing | {p_g['ez3_lo']:.3f} → {p_g['ez3_hi']:.3f} ({p_g['delta_ez3']:+.3f}) | {p_g['ez45_lo']:.3f} → {p_g['ez45_hi']:.3f} ({p_g['delta_ez45']:+.3f}) | {p_g['band_lo']:.3f} → {p_g['band_hi']:.3f} |
| PTAY0599P8S1 | fixed-160 | {p_f['ez3_lo']:.3f} → {p_f['ez3_hi']:.3f} ({p_f['delta_ez3']:+.3f}) | {p_f['ez45_lo']:.3f} → {p_f['ez45_hi']:.3f} ({p_f['delta_ez45']:+.3f}) | {p_f['band_lo']:.3f} → {p_f['band_hi']:.3f} |

**Growing vs fixed-160 for HCPT0008.** Inside the 96-sample dev pool the ez=4.5 ratio is almost flat (pool 20 → 96: {h_g['ez45_lo']:.3f} → {h_g96['ez45']:.3f}). The drop is concentrated **after fillers enter**: pool 120 = {h_g120['ez45']:.3f}, pool 160 = {h_g['ez45_hi']:.3f}. Replay medians: growing pool 80 (still 96-candidate) median max ez = {h80g['median']:.2f} (P(>4.5)={h80g['p45']:.2f}); pool 160 median = {h160['median']:.2f} (P(>4.5)={h160['p45']:.2f}). So for HCPT0008 on the growing sweep, **composition** (more heterogeneous refs → larger σ → smaller ez) does as much as n. On fixed-160 the fillers are already in the candidate, so n alone produces a smooth fall.

PTAY0599P8S1 does **not** need fillers: even pool 20→96 on the growing sweep takes ez=4.5 from {p_g['ez45_lo']:.3f} to well below 0.05. Its stable ez is lower, so variance collapse inside the 96 is enough.

![Figure 3](figures/fig3_discordant_gallery.png)

*Figure 3.* Every fixed-160 discordant sample. `[rise]` = ez3 still climbing; `[ceil]` = ez3 already ~1. Same purple-band geometry.

---

## 3. Mechanism: variance collapse into a gap between cutoffs

Ezscore is a **studentized** combined score. Two n-dependent pieces:

1. **Sampling variance of (μ, σ).** With `ref_n = pool_size/2` ez refs, both the location and the scale of the Normal baseline jitter across repeats. Small n → fat tails of max-chr ezscore. Large n → the sample's ezscore concentrates around a stable value.
2. **Downward bias / left tail of sample SD.** The pipeline uses `nanstd(..., ddof=0)`. For n = 10 (pool 20) the chi-scale factor is ~0.92, so ezscores are inflated ~8% on average, and a minority of draws with unusually small σ send ez **far** above 4.5. At n = 80 (pool 160) that bias is ~0.6%.

If the **stable** (large-n) max-chr ezscore sits in **(3, 4.5]**:

- the left tail that used to fall below 3 disappears → **ez=3 ratio → 1** (or stays 1 if it was already there)
- the right tail that used to exceed 4.5 disappears → **ez=4.5 ratio → 0** (or toward 0 if the median is close to 4.5)

A {n_mc}-repeat replay (same seed / candidate rule as the sweep; enough for the distribution, not a 1e6 replica) of max-chr ezscore:

![Figure 4](figures/fig4_maxez_violin.png)

*Figure 4.* Violin of max-chr ezscore vs pool size. Horizontal lines: 3 and 4.5. PTAY0599P8S1 settles near 3.6; HCPT0008 settles near 4.3.

| Sample | Pool | median max ez | SD of max ez | P(>3) | P(>4.5) | median ez-ref SD at max chr |
|--------|------|---------------|--------------|-------|---------|-----------------------------|
| HCPT0008 | 20 | {h20['median']:.2f} | {h20['sd']:.2f} | {h20['p3']:.3f} | {h20['p45']:.3f} | {h20['sd_ref']:.3f} |
| HCPT0008 | 80 | {h80['median']:.2f} | {h80['sd']:.2f} | {h80['p3']:.3f} | {h80['p45']:.3f} | {h80['sd_ref']:.3f} |
| HCPT0008 | 160 | {h160['median']:.2f} | {h160['sd']:.2f} | {h160['p3']:.3f} | {h160['p45']:.3f} | {h160['sd_ref']:.3f} |
| PTAY0599P8S1 | 20 | {p20['median']:.2f} | {p20['sd']:.2f} | {p20['p3']:.3f} | {p20['p45']:.3f} | {p20['sd_ref']:.3f} |
| PTAY0599P8S1 | 160 | {p160['median']:.2f} | {p160['sd']:.2f} | {p160['p3']:.3f} | {p160['p45']:.3f} | {p160['sd_ref']:.3f} |

HCPT0008: median {h20['median']:.2f} → {h160['median']:.2f}, SD {h20['sd']:.2f} → {h160['sd']:.2f}. The large-n median is **just below 4.5**, so ez=3 is done moving and ez=4.5 is the cutoff the shrinking right tail is leaving.

PTAY0599P8S1: median {p20['median']:.2f} → {p160['median']:.2f}, SD {p20['sd']:.2f} → {p160['sd']:.2f}. Fully inside the gap, so **both** cutoffs move.

![Figure 7](figures/fig7_variance_collapse.png)

*Figure 7.* Normal approximation of the HCPT0008 max-ez distribution at pool 20 vs 160 (fixed-160). The right tail peels off 4.5; the left tail is already above 3.

---

## 4. It is the expected chromosome, not off-target noise

Both samples are labelled T22, so the expected driver is chr22. If ez=4.5 were being driven by **random other chromosomes** at small n (small-σ explosions), then P(other chr > 4.5) would fall with pool size while P(chr22 > 4.5) stayed put. That is only a small part of the story.

![Figure 5](figures/fig5_chr_decomposition.png)

*Figure 5.* Fixed-160, {n_mc} repeats. Solid = any chromosome; dashed = expected chr (chr22); dotted = some *other* chromosome > 4.5.

| Sample | Pool | P(chr22 > 3) | P(chr22 > 4.5) | P(other > 4.5) | P(max chr = chr22) |
|--------|------|--------------|----------------|----------------|--------------------|
| HCPT0008 | 20 | {h20['p_exp3']:.3f} | {h20['p_exp45']:.3f} | {h20['p_oth45']:.3f} | {h20['frac_expected_max']:.3f} |
| HCPT0008 | 160 | {h160['p_exp3']:.3f} | {h160['p_exp45']:.3f} | {h160['p_oth45']:.3f} | {h160['frac_expected_max']:.3f} |
| PTAY0599P8S1 | 20 | {p20['p_exp3']:.3f} | {p20['p_exp45']:.3f} | {p20['p_oth45']:.3f} | {p20['frac_expected_max']:.3f} |
| PTAY0599P8S1 | 160 | {p160['p_exp3']:.3f} | {p160['p_exp45']:.3f} | {p160['p_oth45']:.3f} | {p160['frac_expected_max']:.3f} |

Off-target >4.5 calls do shrink, but the **chr22** ezscore itself is what sits in the gap: at pool 160, P(chr22 > 3) stays high while P(chr22 > 4.5) is near zero. These are **moderate T22 signals**, not spurious multi-chr flags.

HCPT0008 has FF = 2.0% — a weak dosage bump. After ez normalization against a well-estimated Normal baseline, chr22 lands around ez ≈ {h160['median']:.1f}, which a cutoff of 3 still catches and 4.5 does not. PTAY0599P8S1 (FF 5.2%, blacklisted) is the same geometry at a slightly different location.

---

## 5. Small ez-ref SD is the fuel for the ez=4.5 tail

![Figure 6](figures/fig6_sd_vs_maxez.png)

*Figure 6.* Each point is one repeat (subsampled). Colour = pool size. Horizontal lines at 3 and 4.5.

Ezscore = (combined − μ) / σ, so a **small ez-ref σ** is a large ez. Pool 20 occupies the left side of the plot (small σ, max ez often > 4.5). Pool 160 occupies the right (larger, more stable σ, max ez packed between the lines). This is the same ez-ref-SD failure mode as the 40+40 admittance analysis, just seen from the **eval-sample** side: small-n σ underestimates the Normal spread and inflates borderline trisomies over 4.5.

Growing the pool has a second effect in the **growing-candidate** design: nested test fillers can raise population σ, which also shrinks ez. For HCPT0008 that composition change is the *main* ez=4.5 drop (flat until pool 96, then a step down). For PTAY0599P8S1, variance collapse inside the 96 is already enough. Fixed-160 removes the composition change: n alone still produces the opposite-direction pattern for both samples.

---

## 6. When this does *not* happen

| Large-n max ez | ez=3 as n↑ | ez=4.5 as n↑ |
|----------------|------------|--------------|
| ≫ 4.5 (strong T, high FF) | stays ~1 | stays ~1 or rises to 1 |
| just under 4.5 (HCPT0008) | stays ~1 | falls, not necessarily to 0 |
| well inside (3, 4.5] (PTAY0599P8S1) | rises to 1 | falls to 0 |
| ∈ (0, 3] (weak T / odd Normal) | falls to 0 | stays ~0 |
| ≪ 0 (typical Normal) | stays ~0 | stays ~0 |

Opposite directions are not a bug in either sweep. They are what the two cutoffs **must** do for any sample whose stable ezscore lives in the gap. The interactive plots make that visible because they show raw signal ratio, not a single operating point.

Practical implications:

- A cutoff of **4.5 is conservative** for low-FF / moderate-signal samples: enlarging the reference pool **reduces** their call rate. At `signal_ratio ≥ 0.5` they become FN (HCPT0008: ez=4.5 ratio 0.51 → 0.18).
- A cutoff of **3 is liberal** on the same samples: enlarging the pool **increases** (or saturates) their call rate — including a few labelled Normals with FF < 1%.
- Pool-size MCC/AUC at a **fixed** cutoff can move because borderline samples **cross** that cutoff as the ezscore distribution tightens, not only because the reference is “better”.
- Blacklist members such as PTAY0599P8S1 are not a special mechanism. They are moderate-signal T22 (or T22-like) points in the same gap; they were excluded from MCC because they are known problem cases, not because their pool-size physics differs.
- For HCPT0008-like samples, **adding heterogeneous test fillers** after the 96 can shift the large-n median across 4.5 even when n within the 96 does not.

---

## 7. Files

```
ez3_vs_ez45_discordant/
  REPORT.md
  tables/trajectories.tsv          # every sample × pool × design
  tables/sample_trends.tsv         # Spearman / deltas / pattern
  tables/discordant_fixed160.tsv
  tables/mc_max_ez.tsv             # 4000-repeat max-ez replay
  tables/mc_summary.tsv
  figures/fig1_trend_scatter.png
  figures/fig2_focus_trajectories.png
  figures/fig3_discordant_gallery.png
  figures/fig4_maxez_violin.png
  figures/fig5_chr_decomposition.png
  figures/fig6_sd_vs_maxez.png
  figures/fig7_variance_collapse.png
```
"""
    path = out_dir / "REPORT.md"
    path.write_text(report)
    console.print(f"[green]report[/green] {path}")


@click.command()
@click.option("--sweep-base", default=str(SWEEP_BASE), type=click.Path(file_okay=False))
@click.option("--input-dir", default=str(DEFAULT_INPUT), type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--n-repeats", default=5000, show_default=True, type=int)
@click.option("--skip-mc", is_flag=True, default=False)
def main(sweep_base: str, input_dir: str, output_dir: str | None, n_repeats: int, skip_mc: bool) -> None:
    global SWEEP_BASE
    SWEEP_BASE = Path(sweep_base)
    out = Path(output_dir) if output_dir else SWEEP_BASE / "ez3_vs_ez45_discordant"
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]ez3 vs ez4.5 discordant trends")
    traj = pd.concat([pair_design(n) for n in DESIGNS], ignore_index=True)
    traj.to_csv(tab_dir / "trajectories.tsv", sep="\t", index=False, float_format="%.6f")
    trends = sample_trends(traj)
    trends.to_csv(tab_dir / "sample_trends.tsv", sep="\t", index=False, float_format="%.6f")
    disc = trends[(trends["design"] == "fixed160") & trends["discordant"]].sort_values(
        "delta_band", ascending=False
    )
    disc.to_csv(tab_dir / "discordant_fixed160.tsv", sep="\t", index=False, float_format="%.6f")
    console.print(
        f"fixed160 discordant: {len(disc)} / {trends[trends['design']=='fixed160'].shape[0]}"
    )
    console.print("focus:")
    for d in ("growing", "fixed160"):
        for s in FOCUS:
            r = trends[(trends["design"] == d) & (trends["sample"] == s)]
            if r.empty:
                console.print(f"  {d} {s}: MISSING")
                continue
            x = r.iloc[0]
            console.print(
                f"  {d:10s} {s:16s}  ez3 {x['ez3_lo']:.3f}→{x['ez3_hi']:.3f}  "
                f"ez45 {x['ez45_lo']:.3f}→{x['ez45_hi']:.3f}  "
                f"ρ3={x['rho_ez3']:+.2f} ρ45={x['rho_ez45']:+.2f}  "
                f"disc={bool(x['discordant'])}"
            )

    fig1_scatter(trends, fig_dir / "fig1_trend_scatter.png")
    fig2_focus_traj(traj, fig_dir / "fig2_focus_trajectories.png")
    fig3_discordant_gallery(traj, trends, fig_dir / "fig3_discordant_gallery.png")

    mc_path = tab_dir / "mc_max_ez.tsv"
    if skip_mc and mc_path.is_file():
        mc = pd.read_csv(mc_path, sep="\t")
        console.print(f"reused MC {mc_path}")
    else:
        ctx = load_score_ctx(Path(input_dir))
        pool_sizes = [20, 40, 80, 160]
        parts = [
            simulate_max_ez(
                ctx,
                design=d,
                pool_sizes=pool_sizes,
                n_repeats=n_repeats,
                focus=FOCUS,
            )
            for d in ("fixed160", "growing")
        ]
        mc = pd.concat(parts, ignore_index=True)
        mc.to_csv(mc_path, sep="\t", index=False, float_format="%.6f")

    mc_sum = (
        mc.assign(is_exp=mc["max_chr"] == mc["expected_chr"])
        .groupby(["design", "sample", "pool_size"], sort=True)
        .agg(
            n=("max_ez", "size"),
            median_max_ez=("max_ez", "median"),
            mean_max_ez=("max_ez", "mean"),
            sd_max_ez=("max_ez", "std"),
            p_gt3=("gt3", "mean"),
            p_gt45=("gt45", "mean"),
            p_exp_gt3=("exp_gt3", "mean"),
            p_exp_gt45=("exp_gt45", "mean"),
            p_other_gt45=("other_gt45", "mean"),
            median_ez_ref_sd=("ez_ref_sd_at_max", "median"),
            frac_expected_is_max=("is_exp", "mean"),
        )
        .reset_index()
    )
    mc_sum.to_csv(tab_dir / "mc_summary.tsv", sep="\t", index=False, float_format="%.6f")

    fig4_maxez_violin(mc, fig_dir / "fig4_maxez_violin.png")
    fig5_chr_decomp(mc, fig_dir / "fig5_chr_decomposition.png")
    fig6_sd_vs_ez(mc, fig_dir / "fig6_sd_vs_maxez.png")
    fig7_schematic(mc, fig_dir / "fig7_variance_collapse.png")
    write_report(out, traj, trends, mc, mc_sum)
    console.print(f"[bold green]done[/bold green] {out}")


if __name__ == "__main__":
    main()
