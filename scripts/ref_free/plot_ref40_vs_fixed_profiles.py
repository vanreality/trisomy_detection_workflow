#!/usr/bin/env python3
"""Compare 40+40 (dev Normal pool) vs fully fixed E(μ/σ) on 80 single-T# samples.

1. 40+40: draw from set=dev, label=Normal, depth_qc=pass. Signal_ratio vs FF.

2. Fixed: estimate E(μ)/E(σ) for epi (hypo/hyper z_intra), z (percentage), and
   combined ez from clean-pool MC draws (pool=220), then score eval samples with
   those fixed parameters only (no per-draw reference set).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
for _p in (SCRIPT_DIR, REF40_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grid_search_ref40 import CHR_LIST  # noqa: E402
from pool_size_ez_ref_bands import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_OUT,
    DEFAULT_PARQUET,
    DEFAULT_TOXIC,
    _chr_block,
    combined_on_query,
    expected_ez_params,
    load_candidates,
)

console = Console()
_WORKER: dict = {}

DEFAULT_REF_N = 40
DEFAULT_REPEATS = 10_000
DEFAULT_SEED = 42
DEFAULT_CUTOFF = 3.0
DEFAULT_SUM_CUTOFF = 4.243
GRAY = "#9E9E9E"
RED = "#C1121F"
BLUE = "#1D4ED8"
CUT_COLOR = "#222222"
PURITY_LO = 0.8


def labels_to_target_chrs(label: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"T(\d+)", str(label)):
        n = int(m.group(1))
        if 1 <= n <= 22:
            c = f"chr{n}"
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def label_to_target_chr(label: str) -> str | None:
    m = re.fullmatch(r"T(\d+)", str(label).strip())
    if not m:
        return None
    n = int(m.group(1))
    return f"chr{n}" if 1 <= n <= 22 else None


def _target_mask_from_labels(labels) -> np.ndarray:
    n = len(labels)
    mask = np.zeros((n, len(CHR_LIST)), dtype=bool)
    for i, lab in enumerate(labels):
        for c in labels_to_target_chrs(lab):
            mask[i, CHR_LIST.index(c)] = True
    return mask


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs > 0:
        return int(n_jobs)
    for key in ("SLURM_CPUS_PER_TASK", "N_JOBS"):
        raw = os.environ.get(key)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return max(1, int(mp.cpu_count() or 1))


def _chunk_spans(n: int, n_jobs: int) -> list[tuple[int, int]]:
    n_jobs = max(1, min(int(n_jobs), int(n)))
    base, rem = divmod(n, n_jobs)
    spans, start = [], 0
    for i in range(n_jobs):
        end = start + base + (1 if i < rem else 0)
        if end > start:
            spans.append((start, end))
        start = end
    return spans


def build_universe(parquet: Path, meta_path: Path, toxic_path: Path) -> dict:
    pack = load_candidates(parquet, meta_path, toxic_path)
    mat = pd.read_parquet(parquet)
    mat["sample"] = mat["sample"].astype(str)
    meta = pd.read_csv(meta_path)
    meta["sample"] = meta["sample"].astype(str)
    want = [s for s in meta["sample"].tolist() if s in set(mat["sample"])]
    total_set = set(pack["total_samples"])
    ordered = list(pack["total_samples"]) + [s for s in want if s not in total_set]
    sub = mat.set_index("sample").reindex(ordered)
    arrays = {
        "hypo": _chr_block(sub, "hypo_z_intra_after_mq"),
        "hyper": _chr_block(sub, "hyper_z_intra_after_mq"),
        "hypo_cnt": _chr_block(sub, "hypo_cpg_count_after_mq"),
        "hyper_cnt": _chr_block(sub, "hyper_cpg_count_after_mq"),
        "pct": _chr_block(sub, "percentage_after_mq"),
    }
    sample_index = {s: i for i, s in enumerate(ordered)}
    meta_idx = meta.set_index("sample").reindex(ordered)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    depth_arr = meta_idx["depth_qc"].astype(str).to_numpy()
    ff_arr = pd.to_numeric(meta_idx["ff_before_mq"], errors="coerce").to_numpy()
    purity_arr = pd.to_numeric(meta_idx["purity"], errors="coerce").to_numpy()
    week_arr = pd.to_numeric(meta_idx["week"], errors="coerce").to_numpy()
    pred_arr = (
        meta_idx["pred_label"]
        .where(meta_idx["pred_label"].notna(), "")
        .astype(str)
        .to_numpy()
    )

    dev_normal = np.flatnonzero(
        (set_arr == "dev") & (label_arr == "Normal") & (depth_arr == "pass")
    ).astype(np.int64)
    clean_idx = np.array([sample_index[s] for s in pack["clean_samples"]], dtype=np.int64)

    pos_mask = (
        np.isin(set_arr, ["dev", "test"])
        & (depth_arr == "pass")
        & (ff_arr > 0.01)
        & np.array([label_to_target_chr(x) is not None for x in label_arr])
    )
    pos_idx = np.flatnonzero(pos_mask).astype(np.int64)
    target_chr = [label_to_target_chr(label_arr[i]) for i in pos_idx]
    target_ci = np.array([CHR_LIST.index(c) for c in target_chr], dtype=np.int64)
    pos = pd.DataFrame(
        {
            "sample": [ordered[i] for i in pos_idx],
            "set": set_arr[pos_idx],
            "label": label_arr[pos_idx],
            "ff_before_mq": ff_arr[pos_idx],
            "purity": purity_arr[pos_idx],
            "target_chr": target_chr,
        }
    )
    eval_mask = np.isin(set_arr, ["dev", "test"]) & (depth_arr == "pass")
    eval_idx = np.flatnonzero(eval_mask).astype(np.int64)
    eval_labels = label_arr[eval_idx]
    eval_targets = [",".join(labels_to_target_chrs(x)) for x in eval_labels]
    eval_df = pd.DataFrame(
        {
            "sample": [ordered[i] for i in eval_idx],
            "set": set_arr[eval_idx],
            "label": eval_labels,
            "ff_before_mq": ff_arr[eval_idx],
            "purity": purity_arr[eval_idx],
            "target_chr": eval_targets,
        }
    )
    return {
        "ordered": ordered,
        "arrays": arrays,
        "sample_index": sample_index,
        "dev_normal_idx": dev_normal,
        "clean_idx": clean_idx,
        "pos_idx": pos_idx,
        "target_ci": target_ci,
        "pos": pos,
        "eval_idx": eval_idx,
        "eval_df": eval_df,
        "pack": pack,
        "set_arr": set_arr,
        "label_arr": label_arr,
        "depth_arr": depth_arr,
        "ff_arr": ff_arr,
        "purity_arr": purity_arr,
        "week_arr": week_arr,
        "pred_arr": pred_arr,
    }


def _init_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)


def _run_ref40_chunk(span: tuple[int, int]) -> np.ndarray:
    """Return [n_repeats_chunk, n_pos] bool: any-chr ez > cutoff."""
    start, end = span
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    half = int(_WORKER["half"])
    arrays = _WORKER["arrays"]
    eval_idx = _WORKER["eval_idx"]
    cutoff = float(_WORKER["cutoff"])
    n_pos = int(eval_idx.size)
    flags = np.zeros((end - start, n_pos), dtype=bool)
    for i, rid in enumerate(range(start, end)):
        drawn = pool_idx[draws[rid]]
        ref_idx = drawn[:half]
        ez_idx = drawn[half:]
        comb = combined_on_query(
            arrays["hypo"],
            arrays["hyper"],
            arrays["hypo_cnt"],
            arrays["hyper_cnt"],
            arrays["pct"],
            ref_idx,
            eval_idx,
        )
        comb_ez = combined_on_query(
            arrays["hypo"],
            arrays["hyper"],
            arrays["hypo_cnt"],
            arrays["hyper_cnt"],
            arrays["pct"],
            ref_idx,
            ez_idx,
        )
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(comb_ez, axis=1)
            sd = np.nanstd(comb_ez, axis=1, ddof=0)
        sd_safe = np.where(sd > 0, sd, np.nan)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            ez = (comb - mu[:, None]) / sd_safe
            max_ez = np.nanmax(ez, axis=0)
        flags[i] = max_ez > cutoff
    return flags


def run_ref40_signal_ratio(
    *,
    arrays: dict,
    pool_idx: np.ndarray,
    eval_idx: np.ndarray,
    n_repeats: int,
    ref_n: int,
    seed: int,
    cutoff: float,
    n_jobs: int,
) -> np.ndarray:
    need = 2 * ref_n
    if pool_idx.size < need:
        raise click.ClickException(f"dev Normal pool {pool_idx.size} < {need}")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_repeats, need), dtype=np.int64)
    n_cand = int(pool_idx.size)
    for i in range(n_repeats):
        draws[i] = rng.permutation(n_cand)[:need]
    payload = {
        "pool_idx": pool_idx,
        "draws": draws,
        "half": ref_n,
        "arrays": arrays,
        "eval_idx": eval_idx,
        "cutoff": cutoff,
    }
    workers = _resolve_n_jobs(n_jobs)
    spans = _chunk_spans(n_repeats, workers)
    flags = np.zeros((n_repeats, eval_idx.size), dtype=bool)
    if workers == 1 or len(spans) == 1:
        _init_worker(payload)
        return _run_ref40_chunk((0, n_repeats))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_init_worker,
        initargs=(payload,),
    ) as pool:
        futs = {pool.submit(_run_ref40_chunk, span): span for span in spans}
        for fut in as_completed(futs):
            start, end = futs[fut]
            flags[start:end] = fut.result()
    return flags


def _track_mu_sd(block: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = block[:, idx]
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(sub, axis=1)
        sd = np.nanstd(sub, axis=1, ddof=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu.astype(np.float64), sd.astype(np.float64)


def scores_with_fixed_epiz(
    arrays: dict,
    query_idx: np.ndarray,
    *,
    hypo_mu: np.ndarray,
    hypo_sd: np.ndarray,
    hyper_mu: np.ndarray,
    hyper_sd: np.ndarray,
    pct_mu: np.ndarray,
    pct_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Episcore and zscore on query_idx using fixed reference μ/σ (n_chr, n_query)."""
    hypo_safe = np.where(hypo_sd > 0, hypo_sd, np.nan)[:, None]
    hyper_safe = np.where(hyper_sd > 0, hyper_sd, np.nan)[:, None]
    pct_safe = np.where(pct_sd > 0, pct_sd, np.nan)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        hypo_z = (arrays["hypo"][:, query_idx] - hypo_mu[:, None]) / hypo_safe
        hyper_z = (arrays["hyper"][:, query_idx] - hyper_mu[:, None]) / hyper_safe
        zscore = (arrays["pct"][:, query_idx] - pct_mu[:, None]) / pct_safe
    w_hypo = np.sqrt(np.nan_to_num(arrays["hypo_cnt"][:, query_idx], nan=0.0))
    w_hyper = np.sqrt(np.nan_to_num(arrays["hyper_cnt"][:, query_idx], nan=0.0))
    total_w = np.sqrt(w_hypo**2 + w_hyper**2)
    total_w = np.where(total_w > 0, total_w, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_inter = (hyper_z * w_hyper - hypo_z * w_hypo) / total_w
    episcore = np.where(np.isnan(s_inter), 0.0, s_inter)
    return episcore, zscore


def estimate_fixed_epiz_ez_params(
    *,
    arrays: dict,
    pool_idx: np.ndarray,
    pool_size: int,
    n_repeats: int,
    seed: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    """MC E(μ)/E(σ) for epi/z ref tracks and for combined ez under those fixed params.

    Per draw of ``pool_size`` from ``pool_idx``, record mean/std of hypo, hyper,
    percentage. Aggregate to E(μ)/E(σ). Then, for each draw, score the draw with
    those *global* fixed epi/z params and aggregate E(μ)/E(σ) of combined.
    """
    del n_jobs  # serial; mean/std + scoring is cheap relative to 40+40
    if pool_size > pool_idx.size:
        raise click.ClickException(f"pool_size={pool_size} > candidate n={pool_idx.size}")
    n_chr = arrays["hypo"].shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_repeats, pool_size), dtype=np.int64)
    n_cand = int(pool_idx.size)
    for i in range(n_repeats):
        draws[i] = rng.permutation(n_cand)[:pool_size]

    hypo_mu_r = np.zeros((n_repeats, n_chr), dtype=np.float64)
    hypo_sd_r = np.zeros((n_repeats, n_chr), dtype=np.float64)
    hyper_mu_r = np.zeros((n_repeats, n_chr), dtype=np.float64)
    hyper_sd_r = np.zeros((n_repeats, n_chr), dtype=np.float64)
    pct_mu_r = np.zeros((n_repeats, n_chr), dtype=np.float64)
    pct_sd_r = np.zeros((n_repeats, n_chr), dtype=np.float64)

    for rid in range(n_repeats):
        drawn = pool_idx[draws[rid]]
        hypo_mu_r[rid], hypo_sd_r[rid] = _track_mu_sd(arrays["hypo"], drawn)
        hyper_mu_r[rid], hyper_sd_r[rid] = _track_mu_sd(arrays["hyper"], drawn)
        pct_mu_r[rid], pct_sd_r[rid] = _track_mu_sd(arrays["pct"], drawn)
        if rid % 2000 == 0:
            console.print(f"  epi/z stats repeat {rid}/{n_repeats}")

    e_hypo_mu, e_hypo_sd = expected_ez_params(hypo_mu_r, hypo_sd_r)
    e_hyper_mu, e_hyper_sd = expected_ez_params(hyper_mu_r, hyper_sd_r)
    e_pct_mu, e_pct_sd = expected_ez_params(pct_mu_r, pct_sd_r)

    ez_mu_r = np.zeros((n_repeats, n_chr), dtype=np.float64)
    ez_sd_r = np.zeros((n_repeats, n_chr), dtype=np.float64)
    for rid in range(n_repeats):
        drawn = pool_idx[draws[rid]]
        ep, zs = scores_with_fixed_epiz(
            arrays,
            drawn,
            hypo_mu=e_hypo_mu,
            hypo_sd=e_hypo_sd,
            hyper_mu=e_hyper_mu,
            hyper_sd=e_hyper_sd,
            pct_mu=e_pct_mu,
            pct_sd=e_pct_sd,
        )
        comb = ep + zs
        with np.errstate(invalid="ignore"):
            ez_mu_r[rid] = np.nanmean(comb, axis=1)
            ez_sd_r[rid] = np.nanstd(comb, axis=1, ddof=0)
        if rid % 2000 == 0:
            console.print(f"  ez stats repeat {rid}/{n_repeats}")

    e_ez_mu, e_ez_sd = expected_ez_params(ez_mu_r, ez_sd_r)
    return {
        "hypo_mu": e_hypo_mu.astype(np.float32),
        "hypo_sd": e_hypo_sd.astype(np.float32),
        "hyper_mu": e_hyper_mu.astype(np.float32),
        "hyper_sd": e_hyper_sd.astype(np.float32),
        "pct_mu": e_pct_mu.astype(np.float32),
        "pct_sd": e_pct_sd.astype(np.float32),
        "ez_mu": e_ez_mu.astype(np.float32),
        "ez_sd": e_ez_sd.astype(np.float32),
    }


def fixed_ez_profiles_fully_fixed(
    arrays: dict,
    eval_idx: np.ndarray,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    """ezscore [n_chr, n_eval] with fixed epi/z and fixed ez E(μ)/E(σ)."""
    ep, zs = scores_with_fixed_epiz(
        arrays,
        eval_idx,
        hypo_mu=params["hypo_mu"].astype(np.float64),
        hypo_sd=params["hypo_sd"].astype(np.float64),
        hyper_mu=params["hyper_mu"].astype(np.float64),
        hyper_sd=params["hyper_sd"].astype(np.float64),
        pct_mu=params["pct_mu"].astype(np.float64),
        pct_sd=params["pct_sd"].astype(np.float64),
    )
    comb = ep + zs
    e_mu = params["ez_mu"].astype(np.float64)
    e_sd = params["ez_sd"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (comb - e_mu[:, None]) / np.where(e_sd > 0, e_sd, np.nan)[:, None]


def fixed_sum_epi_z(
    arrays: dict,
    eval_idx: np.ndarray,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    """sum_epi_z = epi + z using fixed E(μ)/E(σ) for hypo, hyper, percentage."""
    ep, zs = scores_with_fixed_epiz(
        arrays,
        eval_idx,
        hypo_mu=params["hypo_mu"].astype(np.float64),
        hypo_sd=params["hypo_sd"].astype(np.float64),
        hyper_mu=params["hyper_mu"].astype(np.float64),
        hyper_sd=params["hyper_sd"].astype(np.float64),
        pct_mu=params["pct_mu"].astype(np.float64),
        pct_sd=params["pct_sd"].astype(np.float64),
    )
    return ep + zs


def fixed_ez_profiles(
    arrays: dict,
    ref_idx: np.ndarray,
    eval_idx: np.ndarray,
    e_mu: np.ndarray,
    e_sd: np.ndarray,
) -> np.ndarray:
    """Legacy: epi/z vs empirical ref_idx, ez with fixed E(μ)/E(σ)."""
    comb = combined_on_query(
        arrays["hypo"],
        arrays["hyper"],
        arrays["hypo_cnt"],
        arrays["hyper_cnt"],
        arrays["pct"],
        ref_idx,
        eval_idx,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return (comb - e_mu[:, None]) / np.where(e_sd > 0, e_sd, np.nan)[:, None]


def plot_signal_ratio_vs_ff(pos: pd.DataFrame, out: Path, cutoff: float, n_repeats: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    purity = pd.to_numeric(pos["purity"], errors="coerce")
    lo = purity < PURITY_LO
    hi = ~lo
    if hi.any():
        ax.scatter(
            pos.loc[hi, "ff_before_mq"],
            pos.loc[hi, "signal_ratio"],
            s=36,
            alpha=0.85,
            c=RED,
            marker="o",
            edgecolors="#5A0000",
            linewidths=0.4,
            label=f"purity≥{PURITY_LO:g}",
            zorder=3,
        )
    if lo.any():
        ax.scatter(
            pos.loc[lo, "ff_before_mq"],
            pos.loc[lo, "signal_ratio"],
            s=48,
            alpha=0.95,
            c=BLUE,
            marker="D",
            edgecolors="#1E3A8A",
            linewidths=0.45,
            label=f"purity<{PURITY_LO:g}",
            zorder=4,
        )
    ax.axhline(0.5, color="#888", ls="--", lw=1.0, label="ratio=0.5")
    ax.set_xlabel("ff_before_mq")
    ax.set_ylabel(f"ezscore signal_ratio (cutoff={cutoff:g})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"40+40 from dev Normal (n={int(pos.attrs.get('pool_n', 98))}): "
        f"signal_ratio vs FF  ({n_repeats} repeats, {len(pos)} single T#)"
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    console.print(f"  wrote {out}")


def plot_fixed_ez_scatter(
    pos: pd.DataFrame,
    ez: np.ndarray,
    out: Path | None,
    cutoff: float,
    *,
    jitter: float = 0.22,
    seed: int = 0,
    target_mask: np.ndarray | None = None,
    ylabel: str = "ezscore",
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """One scatter: x=chr (jittered), y=score; gray=other, red/blue=T# target."""
    n = len(pos)
    n_chr = len(CHR_LIST)
    rng = np.random.default_rng(seed)
    chr_i = np.tile(np.arange(n_chr), n)
    sample_i = np.repeat(np.arange(n), n_chr)
    y = ez.T.reshape(-1)
    if target_mask is None:
        target = pos["target_ci"].to_numpy(dtype=int)
        is_tgt = chr_i == target[sample_i]
    else:
        is_tgt = target_mask[sample_i, chr_i]
    purity = pd.to_numeric(pos["purity"], errors="coerce").to_numpy()
    lo_samp = purity < PURITY_LO
    is_tgt_lo = is_tgt & lo_samp[sample_i]
    is_tgt_hi = is_tgt & ~lo_samp[sample_i]
    x = chr_i.astype(np.float64) + rng.uniform(-jitter, jitter, size=chr_i.size)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11.2, 5.6))
    for i in range(n_chr):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#F2F4F7", zorder=0, lw=0)
    for i in range(1, n_chr):
        ax.axvline(i - 0.5, color="#8B939E", lw=0.85, zorder=1, solid_capstyle="butt")
    ax.axhline(0, color="#C5CAD1", lw=0.8, zorder=2)
    ax.axhline(cutoff, color=CUT_COLOR, ls="--", lw=1.15, zorder=2)
    ax.scatter(
        x[~is_tgt],
        y[~is_tgt],
        s=11 if not created else 14,
        c=GRAY,
        alpha=0.42,
        linewidths=0,
        zorder=3,
        label="other chr",
        rasterized=True,
    )
    if is_tgt_hi.any():
        ax.scatter(
            x[is_tgt_hi],
            y[is_tgt_hi],
            s=26 if not created else 30,
            c=RED,
            marker="o",
            alpha=0.92,
            edgecolors="#5A0000",
            linewidths=0.35,
            zorder=4,
            label=f"T# target (purity≥{PURITY_LO:g})",
        )
    if is_tgt_lo.any():
        ax.scatter(
            x[is_tgt_lo],
            y[is_tgt_lo],
            s=38 if not created else 44,
            c=BLUE,
            marker="D",
            alpha=0.95,
            edgecolors="#1E3A8A",
            linewidths=0.4,
            zorder=5,
            label=f"T# target (purity<{PURITY_LO:g})",
        )
    ax.set_xticks(np.arange(n_chr), [c.replace("chr", "") for c in CHR_LIST], fontsize=8)
    ax.set_xlim(-0.5, n_chr - 0.5)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(ylabel)
    ax.set_title(
        title or f"Fixed E(μ/σ) for epi, z & ez: ezscore vs chr  ({n} single T#)",
        fontsize=10,
        pad=7,
    )
    ax.legend(
        frameon=False,
        fontsize=7.5 if not created else 9,
        loc="upper right",
        handletextpad=0.4,
        borderaxespad=0.2,
        markerscale=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8B939E")
    ax.spines["bottom"].set_color("#8B939E")
    ax.tick_params(color="#8B939E", labelcolor="#333")
    if created and out is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(out, dpi=160, facecolor="white")
        plt.close(ax.figure)
        console.print(f"  wrote {out}")
    return ax


def load_fixed_params(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise click.ClickException(f"missing {path}; run without --sum-epi-z-only first")
    pack = np.load(path)
    return {k: pack[k] for k in pack.files}


def write_sum_epi_z(
    ctx: dict,
    params: dict[str, np.ndarray],
    *,
    det: Path,
    figdir: Path,
    cutoff: float,
) -> pd.DataFrame:
    eval_df = ctx["eval_df"].copy()
    score = fixed_sum_epi_z(ctx["arrays"], ctx["eval_idx"], params)
    target_mask = _target_mask_from_labels(eval_df["label"].tolist())
    n = len(eval_df)
    finite = np.where(np.isfinite(score), score, -np.inf)
    max_i = np.argmax(finite, axis=0)
    max_v = score[max_i, np.arange(n)]
    target_v = np.full(n, np.nan, dtype=np.float64)
    for j in range(n):
        if target_mask[j].any():
            target_v[j] = float(np.nanmax(score[target_mask[j], j]))
    summary = eval_df.copy()
    summary["max_sum_epi_z"] = max_v
    summary["max_chr"] = [CHR_LIST[int(i)] for i in max_i]
    summary["target_sum_epi_z"] = target_v
    summary["call"] = max_v > cutoff
    summary["target_call"] = np.where(np.isfinite(target_v), target_v > cutoff, False)
    wide = eval_df.copy()
    for i, chr_name in enumerate(CHR_LIST):
        wide[chr_name] = score[i]
    wide_path = det / "fixed_sum_epi_z_profiles.tsv"
    sum_path = det / "fixed_sum_epi_z_calls.tsv"
    wide.to_csv(wide_path, sep="\t", index=False, float_format="%.6f")
    summary.to_csv(sum_path, sep="\t", index=False, float_format="%.6f")
    console.print(f"  wrote {wide_path}")
    console.print(f"  wrote {sum_path}")

    plot_fixed_ez_scatter(
        eval_df,
        score,
        figdir / "fixed_sum_epi_z_vs_chr.png",
        cutoff=cutoff,
        target_mask=target_mask,
        ylabel="sum_epi_z",
        title=(
            f"Fixed E(μ/σ) epi & z: sum_epi_z vs chr  "
            f"({n} dev/test, depth_qc=pass, cutoff={cutoff:g})"
        ),
    )

    labels = eval_df["label"].astype(str)
    is_normal = labels == "Normal"
    is_single_t = np.array([label_to_target_chr(x) is not None for x in labels])
    n_norm = int(is_normal.sum())
    n_t = int(is_single_t.sum())
    fp = int((is_normal & summary["call"].to_numpy()).sum()) if n_norm else 0
    tp = int((is_single_t & summary["target_call"].to_numpy()).sum()) if n_t else 0
    fn = n_t - tp
    console.print(
        f"[green]sum_epi_z[/green] n={n} Normal={n_norm} single_T#={n_t}  "
        f"cutoff={cutoff:g}  Normal any>cut {fp}/{n_norm}  "
        f"T# target>cut {tp}/{n_t}  FN={fn}"
    )
    return summary


def _cohort_frame(ctx: dict, idx: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": [ctx["ordered"][i] for i in idx],
            "set": ctx["set_arr"][idx],
            "label": labels,
            "ff_before_mq": ctx["ff_arr"][idx],
            "purity": ctx["purity_arr"][idx],
            "target_chr": [",".join(labels_to_target_chrs(x)) for x in labels],
        }
    )


def plot_cohort_fixed_ez(
    ctx: dict,
    params: dict[str, np.ndarray],
    idx: np.ndarray,
    labels: np.ndarray,
    *,
    out: Path,
    cutoff: float,
    title: str,
) -> None:
    if idx.size == 0:
        console.print(f"  skip empty {out.name}")
        return
    ez = fixed_ez_profiles_fully_fixed(ctx["arrays"], idx, params)
    df = _cohort_frame(ctx, idx, labels)
    plot_fixed_ez_scatter(
        df,
        ez,
        out,
        cutoff=cutoff,
        target_mask=_target_mask_from_labels(df["label"].tolist()),
        ylabel="ezscore",
        title=title,
    )


def write_cohort_ez_plots(
    ctx: dict,
    params: dict[str, np.ndarray],
    *,
    figdir: Path,
    cutoff: float,
) -> None:
    set_arr = ctx["set_arr"]
    label_arr = ctx["label_arr"]
    depth_arr = ctx["depth_arr"]
    ff_arr = ctx["ff_arr"]
    pred_arr = ctx["pred_arr"]
    week_arr = ctx["week_arr"]
    is_single = np.array([label_to_target_chr(x) is not None for x in label_arr])
    base = (
        np.isin(set_arr, ["dev", "test"])
        & (depth_arr == "pass")
        & ((label_arr == "Normal") | is_single)
    )
    ge1 = np.flatnonzero(base & (ff_arr >= 0.01)).astype(np.int64)
    lt1 = np.flatnonzero(base & (ff_arr < 0.01)).astype(np.int64)
    jptay = np.flatnonzero((set_arr == "emergency") & (depth_arr == "pass")).astype(np.int64)
    middle = np.flatnonzero(
        (set_arr == "buffer") & (depth_arr == "pass") & (week_arr > 15)
    ).astype(np.int64)

    specs = [
        (
            ge1,
            label_arr[ge1],
            figdir / "fixed_ez_vs_chr_all_dev_test_ge_1.png",
            (
                f"Fixed ez vs chr  (dev/test, depth pass, Normal+single T#, "
                f"FF≥0.01, n={ge1.size})"
            ),
        ),
        (
            lt1,
            label_arr[lt1],
            figdir / "fixed_ez_vs_chr_all_dev_test_lt_1.png",
            (
                f"Fixed ez vs chr  (dev/test, depth pass, Normal+single T#, "
                f"FF<0.01, n={lt1.size})"
            ),
        ),
        (
            jptay,
            pred_arr[jptay],
            figdir / "fixed_ez_vs_chr_jptay.png",
            f"Fixed ez vs chr  (emergency/jptay, depth pass, pred_label, n={jptay.size})",
        ),
        (
            middle,
            label_arr[middle],
            figdir / "fixed_ez_vs_chr_middle_stage.png",
            f"Fixed ez vs chr  (buffer, depth pass, week>15, n={middle.size})",
        ),
    ]
    console.rule(f"[cyan]cohort fixed ez plots  cutoff={cutoff:g}")
    for idx, labels, out, title in specs:
        n_tgt = int(_target_mask_from_labels(labels).any(axis=1).sum()) if idx.size else 0
        console.print(f"  {out.name}: n={idx.size} with T# target={n_tgt}")
        plot_cohort_fixed_ez(ctx, params, idx, labels, out=out, cutoff=cutoff, title=title)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--total-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--ref-n", default=DEFAULT_REF_N, show_default=True, type=int)
@click.option("--cutoff", default=DEFAULT_CUTOFF, show_default=True, type=float)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option("--e-pool-size", default=220, show_default=True, type=int)
@click.option("--skip-ref40", is_flag=True, default=False, help="Reuse existing signal_ratio TSV")
@click.option(
    "--plots-only",
    is_flag=True,
    default=False,
    help="Regenerate plots from existing detection TSVs (no MC)",
)
@click.option(
    "--sum-epi-z-only",
    is_flag=True,
    default=False,
    help="Score all depth-pass dev/test samples as epi+z using saved fixed params",
)
@click.option(
    "--sum-epi-z-cutoff",
    default=DEFAULT_SUM_CUTOFF,
    show_default=True,
    type=float,
)
@click.option(
    "--cohort-ez-plots",
    is_flag=True,
    default=False,
    help="Write the four cohort fixed-ez scatters from saved params (no MC)",
)
def main(
    parquet: str,
    meta: str,
    toxic: str,
    output_dir: str,
    total_repeats: int,
    ref_n: int,
    cutoff: float,
    seed: int,
    n_jobs: int,
    e_pool_size: int,
    skip_ref40: bool,
    plots_only: bool,
    sum_epi_z_only: bool,
    sum_epi_z_cutoff: float,
    cohort_ez_plots: bool,
) -> None:
    out = Path(output_dir)
    figdir = out / "figures"
    det = out / "detection"
    figdir.mkdir(parents=True, exist_ok=True)
    det.mkdir(parents=True, exist_ok=True)

    ctx = build_universe(Path(parquet), Path(meta), Path(toxic))
    pos = ctx["pos"].copy()
    pos["target_ci"] = ctx["target_ci"]
    console.print(
        f"dev_Normal={ctx['dev_normal_idx'].size} clean={ctx['clean_idx'].size} "
        f"pos_T#={len(pos)} repeats={total_repeats} cutoff={cutoff} "
        f"purity<{PURITY_LO:g}: {int((pos['purity'] < PURITY_LO).sum())}"
    )

    params_path = det / "fixed_epiz_ez_params.npz"
    params_tsv = det / "fixed_epiz_ez_params.tsv"
    ratio_tsv = det / "ref40_dev_signal_ratio.tsv"
    fixed_tsv = det / "fixed_ez_profiles.tsv"

    if sum_epi_z_only:
        console.rule(f"[cyan]sum_epi_z = epi + z  cutoff={sum_epi_z_cutoff:g}")
        params = load_fixed_params(params_path)
        write_sum_epi_z(
            ctx,
            params,
            det=det,
            figdir=figdir,
            cutoff=sum_epi_z_cutoff,
        )
        return

    if cohort_ez_plots:
        params = load_fixed_params(params_path)
        write_cohort_ez_plots(ctx, params, figdir=figdir, cutoff=cutoff)
        return

    if plots_only:
        console.rule("[cyan]plots-only from existing TSVs")
        if not ratio_tsv.is_file() or not fixed_tsv.is_file():
            raise click.ClickException(f"need {ratio_tsv.name} and {fixed_tsv.name}")
        prev = pd.read_csv(ratio_tsv, sep="\t")
        pos = pos.drop(columns=["signal_ratio"], errors="ignore").merge(
            prev[["sample", "signal_ratio"]], on="sample", how="left"
        )
        pos.attrs["pool_n"] = int(ctx["dev_normal_idx"].size)
        pos.to_csv(ratio_tsv, sep="\t", index=False, float_format="%.6f")
        wide_raw = pd.read_csv(fixed_tsv, sep="\t")
        wide = pos[["sample", "set", "label", "ff_before_mq", "purity", "target_chr"]].merge(
            wide_raw.drop(
                columns=["set", "label", "ff_before_mq", "purity", "target_chr"],
                errors="ignore",
            ),
            on="sample",
            how="left",
        )
        wide.to_csv(fixed_tsv, sep="\t", index=False, float_format="%.6f")
        ez = np.vstack([wide[c].to_numpy(dtype=float) for c in CHR_LIST])
        plot_signal_ratio_vs_ff(
            pos,
            figdir / "ref40_dev_signal_ratio_vs_ff.png",
            cutoff=cutoff,
            n_repeats=total_repeats,
        )
        plot_fixed_ez_scatter(pos, ez, figdir / "fixed_ez_vs_chr_T80.png", cutoff=cutoff)
        return

    console.rule(f"[cyan]estimate fixed E(μ/σ) epi/z/ez  pool={e_pool_size}")
    # Prefer candidate-indexed arrays for MC (same as load_candidates order).
    pack_arrays = ctx["pack"]["arrays"]
    pack_clean = ctx["pack"]["clean_idx"]
    params = estimate_fixed_epiz_ez_params(
        arrays=pack_arrays,
        pool_idx=pack_clean,
        pool_size=e_pool_size,
        n_repeats=total_repeats,
        seed=seed + 99,
        n_jobs=n_jobs,
    )
    np.savez_compressed(params_path, **params)
    rows = []
    for i, chr_name in enumerate(CHR_LIST):
        rows.append(
            {
                "chr": chr_name,
                "hypo_mu": float(params["hypo_mu"][i]),
                "hypo_sd": float(params["hypo_sd"][i]),
                "hyper_mu": float(params["hyper_mu"][i]),
                "hyper_sd": float(params["hyper_sd"][i]),
                "pct_mu": float(params["pct_mu"][i]),
                "pct_sd": float(params["pct_sd"][i]),
                "ez_mu": float(params["ez_mu"][i]),
                "ez_sd": float(params["ez_sd"][i]),
            }
        )
    pd.DataFrame(rows).to_csv(params_tsv, sep="\t", index=False, float_format="%.8f")
    console.print(f"  wrote {params_path}")

    if skip_ref40 and ratio_tsv.is_file():
        console.rule("[cyan]reuse 40+40 signal_ratio TSV")
        prev = pd.read_csv(ratio_tsv, sep="\t")
        pos = pos.drop(columns=["signal_ratio"], errors="ignore").merge(
            prev[["sample", "signal_ratio"]], on="sample", how="left"
        )
        pos.attrs["pool_n"] = int(ctx["dev_normal_idx"].size)
        pos.to_csv(ratio_tsv, sep="\t", index=False, float_format="%.6f")
    else:
        console.rule("[cyan]40+40 signal_ratio (dev Normal pool)")
        flags = run_ref40_signal_ratio(
            arrays=ctx["arrays"],
            pool_idx=ctx["dev_normal_idx"],
            eval_idx=ctx["pos_idx"],
            n_repeats=total_repeats,
            ref_n=ref_n,
            seed=seed,
            cutoff=cutoff,
            n_jobs=n_jobs,
        )
        pos["signal_ratio"] = flags.mean(axis=0)
        pos.attrs["pool_n"] = int(ctx["dev_normal_idx"].size)
        pos.to_csv(ratio_tsv, sep="\t", index=False, float_format="%.6f")
    plot_signal_ratio_vs_ff(
        pos,
        figdir / "ref40_dev_signal_ratio_vs_ff.png",
        cutoff=cutoff,
        n_repeats=total_repeats,
    )

    console.rule("[cyan]fixed epi/z/ez profiles")
    ez = fixed_ez_profiles_fully_fixed(ctx["arrays"], ctx["pos_idx"], params)
    wide = pos[["sample", "set", "label", "ff_before_mq", "purity", "target_chr"]].copy()
    for i, chr_name in enumerate(CHR_LIST):
        wide[chr_name] = ez[i]
    wide.to_csv(fixed_tsv, sep="\t", index=False, float_format="%.6f")
    plot_fixed_ez_scatter(pos, ez, figdir / "fixed_ez_vs_chr_T80.png", cutoff=cutoff)
    write_cohort_ez_plots(ctx, params, figdir=figdir, cutoff=cutoff)

    cfg = {
        "ref40_pool": "dev Normal depth_qc=pass",
        "n_dev_normal": int(ctx["dev_normal_idx"].size),
        "n_pos": len(pos),
        "total_repeats": total_repeats,
        "ref_n": ref_n,
        "cutoff": cutoff,
        "seed": seed,
        "e_pool_size": e_pool_size,
        "fixed_params": str(params_path),
        "fixed_mode": "E(μ)/E(σ) for hypo, hyper, percentage and combined ez",
        "mean_signal_ratio": float(pos["signal_ratio"].mean()),
        "frac_signal_ge_0_5": float((pos["signal_ratio"] >= 0.5).mean()),
        "frac_target_ez_gt_cutoff": float(
            (ez[ctx["target_ci"], np.arange(len(pos))] > cutoff).mean()
        ),
    }
    (det / "profile_compare_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    console.print(
        f"[green]OK[/green] mean signal_ratio={cfg['mean_signal_ratio']:.3f} "
        f"≥0.5: {cfg['frac_signal_ge_0_5']:.1%}  "
        f"fixed target>={cutoff:g}: {cfg['frac_target_ez_gt_cutoff']:.1%}"
    )


if __name__ == "__main__":
    main()
