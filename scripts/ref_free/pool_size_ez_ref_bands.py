#!/usr/bin/env python3
"""Ez-ref mean/SD vs pool size: total vs clean, even-split vs LOO.

Candidate pool is meta ``set ∈ {dev, test}``, ``depth_qc == pass``,
``label == Normal`` (273). Clean pool drops MAD-toxic samples (227).

For each even pool size in [20, 220] step 2, draw that many candidates and
record per-chr mean and SD of (episcore + zscore) on the ez-reference
distribution, ``--total-repeats`` times (default 10k).

Even split: half the draw is the epi/z reference; the other half is the
ez-reference (out-of-sample combined scores).

LOO cross-fitting: the full draw is the epi/z reference; ez-reference
values are leave-one-out combined scores on the same samples.

Features come from ``intermediate_merged_batches_modeA.parquet`` (after-MQ
percentage + hypo/hyper z_intra + CpG counts), matching
``compute_episcore`` / ``compute_zscore`` in ``grid_search_ref40.py``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
if str(REF40_DIR) not in sys.path:
    sys.path.insert(0, str(REF40_DIR))

from grid_search_ref40 import CHR_LIST, compute_episcore, compute_zscore  # noqa: E402

console = Console()

DEFAULT_PARQUET = Path(
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet"
)
DEFAULT_META = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv")
DEFAULT_TOXIC = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
    "/expanded_pool_mad/toxic_samplesheet.tsv"
)
DEFAULT_OUT = Path("/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev")
DEFAULT_REPEATS = 10_000
DEFAULT_SEED = 42
QUANTILES = (5.0, 25.0, 50.0, 75.0, 95.0)

TOTAL_COLOR = "#0D47A1"
CLEAN_COLOR = "#FF6F00"
EVEN_COLOR = "#0D47A1"
LOO_COLOR = "#E69F00"

ARMS = ("even_total", "even_clean", "loo_total", "loo_clean")
ARM_SEED_TAG = {"even_total": 1, "even_clean": 2, "loo_total": 3, "loo_clean": 4}
_WORKER: dict = {}


def _chr_block(df: pd.DataFrame, kind: str) -> np.ndarray:
    cols = [f"{c}_{kind}" for c in CHR_LIST]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise click.ClickException(f"parquet missing columns: {missing[:3]}")
    return df[cols].to_numpy(dtype=np.float64).T


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
    spans = []
    start = 0
    for i in range(n_jobs):
        end = start + base + (1 if i < rem else 0)
        if end > start:
            spans.append((start, end))
        start = end
    return spans


def load_candidates(
    parquet: Path,
    meta_path: Path,
    toxic_path: Path,
) -> dict:
    meta = pd.read_csv(meta_path)
    meta["sample"] = meta["sample"].astype(str)
    cand = meta.loc[
        meta["set"].isin(["dev", "test"])
        & (meta["depth_qc"].astype(str) == "pass")
        & (meta["label"].astype(str) == "Normal")
    ].copy()
    if cand["sample"].duplicated().any():
        raise click.ClickException("duplicate samples in candidate pool")
    toxic = pd.read_csv(toxic_path, sep="\t")
    toxic_ids = set(toxic["sample"].astype(str))
    cand["toxic"] = cand["sample"].isin(toxic_ids)
    total = cand["sample"].tolist()
    clean = cand.loc[~cand["toxic"], "sample"].tolist()

    mat = pd.read_parquet(parquet)
    mat["sample"] = mat["sample"].astype(str)
    missing = [s for s in total if s not in set(mat["sample"])]
    if missing:
        raise click.ClickException(f"{len(missing)} candidates missing from parquet: {missing[:5]}")
    sub = mat.set_index("sample").reindex(total)
    arrays = {
        "hypo": _chr_block(sub, "hypo_z_intra_after_mq"),
        "hyper": _chr_block(sub, "hyper_z_intra_after_mq"),
        "hypo_cnt": _chr_block(sub, "hypo_cpg_count_after_mq"),
        "hyper_cnt": _chr_block(sub, "hyper_cpg_count_after_mq"),
        "pct": _chr_block(sub, "percentage_after_mq"),
    }
    n_nan = {k: int(np.isnan(v).sum()) for k, v in arrays.items()}
    sample_index = {s: i for i, s in enumerate(total)}
    clean_idx = np.array([sample_index[s] for s in clean], dtype=np.int64)
    total_idx = np.arange(len(total), dtype=np.int64)
    return {
        "total_samples": total,
        "clean_samples": clean,
        "total_idx": total_idx,
        "clean_idx": clean_idx,
        "arrays": arrays,
        "n_nan": n_nan,
        "cand": cand,
        "n_toxic": int(cand["toxic"].sum()),
    }


def _mu_sd(block: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = block[:, idx]
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(sub, axis=1)
        sd = np.nanstd(sub, axis=1, ddof=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu, sd


def combined_on_query(
    hypo: np.ndarray,
    hyper: np.ndarray,
    hypo_cnt: np.ndarray,
    hyper_cnt: np.ndarray,
    pct: np.ndarray,
    ref_idx: np.ndarray,
    query_idx: np.ndarray,
) -> np.ndarray:
    """(episcore + zscore) on query samples vs ref_idx. Shapes: (n_chr, n_query)."""
    hypo_mu, hypo_sd = _mu_sd(hypo, ref_idx)
    hyper_mu, hyper_sd = _mu_sd(hyper, ref_idx)
    pct_mu, pct_sd = _mu_sd(pct, ref_idx)
    hypo_safe = np.where(hypo_sd > 0, hypo_sd, np.nan)[:, None]
    hyper_safe = np.where(hyper_sd > 0, hyper_sd, np.nan)[:, None]
    pct_safe = np.where(pct_sd > 0, pct_sd, np.nan)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        hypo_z = (hypo[:, query_idx] - hypo_mu[:, None]) / hypo_safe
        hyper_z = (hyper[:, query_idx] - hyper_mu[:, None]) / hyper_safe
        zscore = (pct[:, query_idx] - pct_mu[:, None]) / pct_safe
    w_hypo = np.sqrt(np.nan_to_num(hypo_cnt[:, query_idx], nan=0.0))
    w_hyper = np.sqrt(np.nan_to_num(hyper_cnt[:, query_idx], nan=0.0))
    total_w = np.sqrt(w_hypo**2 + w_hyper**2)
    total_w = np.where(total_w > 0, total_w, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_inter = (hyper_z * w_hyper - hypo_z * w_hypo) / total_w
    s_inter = np.where(np.isnan(s_inter), 0.0, s_inter)
    return s_inter + zscore


def loo_combined(
    hypo: np.ndarray,
    hyper: np.ndarray,
    hypo_cnt: np.ndarray,
    hyper_cnt: np.ndarray,
    pct: np.ndarray,
) -> np.ndarray:
    """Leave-one-out (episcore + zscore) for every column. Input (n_chr, n)."""
    n = hypo.shape[1]
    if n < 2:
        raise ValueError("LOO needs n>=2")

    def _loo_mu_sd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(x)
        x0 = np.where(finite, x, 0.0)
        cnt = finite.sum(axis=1, keepdims=True).astype(np.float64)
        s = x0.sum(axis=1, keepdims=True)
        s2 = (x0 * x0).sum(axis=1, keepdims=True)
        fin_i = finite.astype(np.float64)
        cnt_i = cnt - fin_i
        s_i = s - x0
        s2_i = s2 - x0 * x0
        mu = np.divide(s_i, cnt_i, out=np.zeros_like(s_i), where=cnt_i > 0)
        var = np.divide(s2_i, cnt_i, out=np.zeros_like(s2_i), where=cnt_i > 0) - mu * mu
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var)
        mu = np.where(cnt_i > 0, mu, 0.0)
        return mu, sd

    hypo_mu, hypo_sd = _loo_mu_sd(hypo)
    hyper_mu, hyper_sd = _loo_mu_sd(hyper)
    pct_mu, pct_sd = _loo_mu_sd(pct)
    hypo_safe = np.where(hypo_sd > 0, hypo_sd, np.nan)
    hyper_safe = np.where(hyper_sd > 0, hyper_sd, np.nan)
    pct_safe = np.where(pct_sd > 0, pct_sd, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        hypo_z = (hypo - hypo_mu) / hypo_safe
        hyper_z = (hyper - hyper_mu) / hyper_safe
        zscore = (pct - pct_mu) / pct_safe
    w_hypo = np.sqrt(np.nan_to_num(hypo_cnt, nan=0.0))
    w_hyper = np.sqrt(np.nan_to_num(hyper_cnt, nan=0.0))
    total_w = np.sqrt(w_hypo**2 + w_hyper**2)
    total_w = np.where(total_w > 0, total_w, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_inter = (hyper_z * w_hyper - hypo_z * w_hypo) / total_w
    s_inter = np.where(np.isnan(s_inter), 0.0, s_inter)
    return s_inter + zscore


def _init_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)


def _run_even_chunk(span: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, end = span
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    half = int(_WORKER["half"])
    n_chr = int(_WORKER["n_chr"])
    hypo = _WORKER["hypo"]
    hyper = _WORKER["hyper"]
    hypo_cnt = _WORKER["hypo_cnt"]
    hyper_cnt = _WORKER["hyper_cnt"]
    pct = _WORKER["pct"]
    mu = np.zeros((end - start, n_chr), dtype=np.float32)
    sd = np.zeros((end - start, n_chr), dtype=np.float32)
    for i, rid in enumerate(range(start, end)):
        drawn = pool_idx[draws[rid]]
        ref_idx = drawn[:half]
        ez_idx = drawn[half:]
        comb = combined_on_query(hypo, hyper, hypo_cnt, hyper_cnt, pct, ref_idx, ez_idx)
        with np.errstate(invalid="ignore"):
            mu[i] = np.nanmean(comb, axis=1)
            sd[i] = np.nanstd(comb, axis=1, ddof=0)
    return mu, sd


def _run_loo_chunk(span: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, end = span
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    n_chr = int(_WORKER["n_chr"])
    hypo = _WORKER["hypo"]
    hyper = _WORKER["hyper"]
    hypo_cnt = _WORKER["hypo_cnt"]
    hyper_cnt = _WORKER["hyper_cnt"]
    pct = _WORKER["pct"]
    mu = np.zeros((end - start, n_chr), dtype=np.float32)
    sd = np.zeros((end - start, n_chr), dtype=np.float32)
    for i, rid in enumerate(range(start, end)):
        drawn = pool_idx[draws[rid]]
        comb = loo_combined(
            hypo[:, drawn],
            hyper[:, drawn],
            hypo_cnt[:, drawn],
            hyper_cnt[:, drawn],
            pct[:, drawn],
        )
        with np.errstate(invalid="ignore"):
            mu[i] = np.nanmean(comb, axis=1)
            sd[i] = np.nanstd(comb, axis=1, ddof=0)
    return mu, sd


def _parallel_chunks(
    fn,
    payload: dict,
    n_repeats: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_chr = int(payload["n_chr"])
    workers = _resolve_n_jobs(n_jobs)
    spans = _chunk_spans(n_repeats, workers)
    mu = np.zeros((n_repeats, n_chr), dtype=np.float32)
    sd = np.zeros((n_repeats, n_chr), dtype=np.float32)
    if workers == 1 or len(spans) == 1:
        _init_worker(payload)
        return fn((0, n_repeats))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_init_worker,
        initargs=(payload,),
    ) as pool:
        futs = {pool.submit(fn, span): span for span in spans}
        for fut in as_completed(futs):
            start, end = futs[fut]
            m, s = fut.result()
            mu[start:end] = m
            sd[start:end] = s
    return mu, sd


def run_arm(
    *,
    arrays: dict[str, np.ndarray],
    pool_idx: np.ndarray,
    pool_size: int,
    n_repeats: int,
    seed: int,
    n_jobs: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if pool_size > pool_idx.size:
        raise click.ClickException(
            f"{mode}: pool_size={pool_size} exceeds candidate n={pool_idx.size}"
        )
    if pool_size < 2 or pool_size % 2:
        raise click.ClickException(f"pool_size must be even >=2, got {pool_size}")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_repeats, pool_size), dtype=np.int64)
    n_cand = int(pool_idx.size)
    for i in range(n_repeats):
        draws[i] = rng.permutation(n_cand)[:pool_size]
    payload = {
        "pool_idx": pool_idx,
        "draws": draws,
        "half": pool_size // 2,
        "n_chr": arrays["hypo"].shape[0],
        "hypo": arrays["hypo"],
        "hyper": arrays["hyper"],
        "hypo_cnt": arrays["hypo_cnt"],
        "hyper_cnt": arrays["hyper_cnt"],
        "pct": arrays["pct"],
    }
    fn = _run_loo_chunk if mode == "loo" else _run_even_chunk
    return _parallel_chunks(fn, payload, n_repeats, n_jobs)


def summarize(arr: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore"):
        q = np.nanpercentile(arr, QUANTILES, axis=0).astype(np.float32)
        mean = np.nanmean(arr, axis=0).astype(np.float32)
    return {"q": q, "mean": mean}


def arm_seed(base: int, pool_size: int, arm: str) -> int:
    tag = ARM_SEED_TAG[arm]
    return int(base) + 100003 * int(pool_size) + tag


def expected_ez_params(mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-chr E(μ) and E(σ)=√(E[σ²] + Var(μ)) across Monte-Carlo repeats."""
    with np.errstate(invalid="ignore"):
        e_mu = np.nanmean(mu, axis=0).astype(np.float64)
        e_sd = np.sqrt(
            np.nanmean(np.square(sd.astype(np.float64)), axis=0) + np.nanvar(mu.astype(np.float64), axis=0)
        ).astype(np.float64)
    return e_mu, e_sd


def _read_json(path: Path) -> dict:
    try:
        text = path.read_text()
        if not text.strip():
            return {}
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def save_pool_summary(path: Path, pack: dict, *, merge_existing: bool = True) -> None:
    payload: dict = {}
    if merge_existing and path.is_file():
        try:
            old = np.load(path, allow_pickle=False)
            payload.update({k: old[k] for k in old.files})
        except (OSError, ValueError, zipfile.BadZipFile):
            payload = {}
    payload["pool_size"] = np.int32(pack["pool_size"])
    payload["n_repeats"] = np.int32(pack["n_repeats"])
    for arm in ARMS:
        if arm not in pack:
            continue
        for metric in ("mu", "sd"):
            stats = pack[arm][metric]
            payload[f"{arm}_{metric}_q"] = stats["q"]
            payload[f"{arm}_{metric}_mean"] = stats["mean"]
        if "e_mu" in pack[arm]:
            payload[f"{arm}_e_mu"] = pack[arm]["e_mu"]
            payload[f"{arm}_e_sd"] = pack[arm]["e_sd"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def load_all_summaries(stat_dir: Path) -> pd.DataFrame:
    rows = []
    for npz_path in sorted(stat_dir.glob("pool_*/summary.npz")):
        try:
            d = np.load(npz_path, allow_pickle=False)
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
        pool = int(d["pool_size"])
        n_repeats = int(d["n_repeats"])
        for arm in ARMS:
            for metric in ("mu", "sd"):
                q_key = f"{arm}_{metric}_q"
                m_key = f"{arm}_{metric}_mean"
                if q_key not in d.files:
                    continue
                q = d[q_key]
                mean = d[m_key]
                for ci, chr_name in enumerate(CHR_LIST):
                    rows.append(
                        {
                            "arm": arm,
                            "metric": metric,
                            "pool_size": pool,
                            "n_repeats": n_repeats,
                            "chr": chr_name,
                            "q05": float(q[0, ci]),
                            "q25": float(q[1, ci]),
                            "q50": float(q[2, ci]),
                            "q75": float(q[3, ci]),
                            "q95": float(q[4, ci]),
                            "mean": float(mean[ci]),
                        }
                    )
    if not rows:
        raise click.ClickException(f"no pool_*/summary.npz under {stat_dir}")
    return pd.DataFrame(rows)


def _arm_spread(sub: pd.DataFrame, arm: str) -> float:
    g = sub.loc[sub["arm"] == arm]
    if g.empty:
        return float("inf")
    return float((g["q95"] - g["q05"]).median())


def _band_axis(
    ax,
    sub: pd.DataFrame,
    arms: list[tuple[str, str, str]],
) -> None:
    ordered = sorted(arms, key=lambda item: _arm_spread(sub, item[0]), reverse=True)
    n = len(ordered)
    for i, (arm, color, _label) in enumerate(ordered):
        g = sub.loc[sub["arm"] == arm].sort_values("pool_size")
        if g.empty:
            continue
        x = g["pool_size"].to_numpy()
        z_fill = 1 + i
        z_line = 10 + i
        ax.fill_between(x, g["q05"], g["q95"], color=color, alpha=0.22, linewidth=0, zorder=z_fill)
        ax.plot(x, g["q50"], color=color, lw=1.8, zorder=z_line, solid_capstyle="round")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_comparison(
    df: pd.DataFrame,
    *,
    metric: str,
    arms: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
    dest: Path,
) -> None:
    fig, axes = plt.subplots(4, 6, figsize=(18.5, 12.2), sharex=True)
    axes = axes.ravel()
    for i, chr_name in enumerate(CHR_LIST):
        ax = axes[i]
        sub = df.loc[(df["metric"] == metric) & (df["chr"] == chr_name)]
        _band_axis(ax, sub, arms)
        ax.set_title(chr_name, fontsize=10)
        if i % 6 == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if i >= 18:
            ax.set_xlabel("pool size", fontsize=8)
        ax.set_xticks([20, 60, 100, 140, 180, 220])
        ax.set_xlim(18, 222)
        ax.tick_params(axis="both", labelsize=7)
    for j in range(len(CHR_LIST), len(axes)):
        axes[j].axis("off")
    handles = [
        Line2D([0], [0], color=color, lw=1.8, label=label) for _arm, color, label in arms
    ]
    handles.append(Patch(facecolor="#888888", alpha=0.22, label="5–95%"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    console.print(f"  wrote {dest}")


def write_report(df: pd.DataFrame, cfg: dict, out: Path) -> None:
    n_total = cfg["n_total"]
    n_clean = cfg["n_clean"]
    n_rep = cfg["total_repeats"]
    sizes = sorted(df["pool_size"].unique())
    lo, hi = min(sizes), max(sizes)

    def _mid_row(arm: str, metric: str, pool: int) -> str:
        g = df.loc[(df["arm"] == arm) & (df["metric"] == metric) & (df["pool_size"] == pool)]
        if g.empty:
            return "NA"
        return (
            f"median-across-chr of repeat-median = {g['q50'].median():.3f}; "
            f"median IQR = {(g['q75'] - g['q25']).median():.3f}"
        )

    mid = sizes[len(sizes) // 2] if sizes else 0
    report = f"""# Ez-ref mean/SD vs pool size (20260816)

Features: `intermediate_merged_batches_modeA.parquet` after-MQ percentage + hypo/hyper z_intra + CpG counts.  
Episcore / zscore match `scripts/ref_explore_plus_grid_search/grid_search_ref40.py` (`ddof=0`).

## Pools

- **total candidates** = {n_total} (`set ∈ {{dev, test}}`, `depth_qc=pass`, `label=Normal`)
- **clean candidates** = {n_clean} (total minus {cfg['n_toxic']} MAD-toxic)
- repeats = **{n_rep}**, seed = {cfg['seed']}
- pool size = {lo}…{hi} step {cfg['pool_step']} ({len(sizes)} sizes)

## Designs

**Even split.** Draw `n` samples, use `n/2` as the epi/z reference and the other `n/2` as the ez reference. Ez-ref μ/σ are the mean and population SD of (episcore+zscore) on that held-out half.

**LOO cross-fitting.** Draw `n` samples and use all `n` as the epi/z reference. Each sample's combined score is computed against the other `n−1`; those `n` LOO scores are the ez-reference distribution.

## Comparison 1 — even split, total vs clean

Line = median; band = 5–95% of 10k repeats (no IQR fill). The narrower band is drawn on top of the wider one.

![comp1 mu](figures/comp1_ez_mu.png)
![comp1 sd](figures/comp1_ez_sd.png)

At pool={mid}: total μ {_mid_row("even_total", "mu", mid)}; clean μ {_mid_row("even_clean", "mu", mid)}.  
At pool={mid}: total σ {_mid_row("even_total", "sd", mid)}; clean σ {_mid_row("even_clean", "sd", mid)}.

## Comparison 2 — total candidates, even split vs LOO

![comp2 mu](figures/comp2_ez_mu.png)
![comp2 sd](figures/comp2_ez_sd.png)

At pool={mid}: even μ {_mid_row("even_total", "mu", mid)}; LOO μ {_mid_row("loo_total", "mu", mid)}.  
At pool={mid}: even σ {_mid_row("even_total", "sd", mid)}; LOO σ {_mid_row("loo_total", "sd", mid)}.

## Comparison 3 — clean candidates, even split vs LOO

![comp3 mu](figures/comp3_ez_mu.png)
![comp3 sd](figures/comp3_ez_sd.png)

At pool={mid}: even μ {_mid_row("even_clean", "mu", mid)}; LOO μ {_mid_row("loo_clean", "mu", mid)}.  
At pool={mid}: even σ {_mid_row("even_clean", "sd", mid)}; LOO σ {_mid_row("loo_clean", "sd", mid)}.

## Outputs

- `stats/pool_*/summary.npz` — per-chr quantiles
- `stats/all_percentiles.tsv`
- `figures/comp{{1,2,3}}_ez_{{mu,sd}}.png`
"""
    out.write_text(report)
    console.print(f"  wrote {out}")


def plot_all(stat_dir: Path, figdir: Path, cfg: dict, report_path: Path) -> pd.DataFrame:
    df = load_all_summaries(stat_dir)
    df.to_csv(stat_dir / "all_percentiles.tsv", sep="\t", index=False, float_format="%.6f")
    n_rep = int(df["n_repeats"].iloc[0])
    n_total = cfg.get("n_total", "?")
    n_clean = cfg.get("n_clean", "?")
    plot_comparison(
        df,
        metric="mu",
        arms=[
            ("even_total", TOTAL_COLOR, f"total n={n_total}"),
            ("even_clean", CLEAN_COLOR, f"clean n={n_clean}"),
        ],
        title=f"Comparison 1: even-split ez-ref mean  (total vs clean, {n_rep} repeats)",
        ylabel="ez-ref mean",
        dest=figdir / "comp1_ez_mu.png",
    )
    plot_comparison(
        df,
        metric="sd",
        arms=[
            ("even_total", TOTAL_COLOR, f"total n={n_total}"),
            ("even_clean", CLEAN_COLOR, f"clean n={n_clean}"),
        ],
        title=f"Comparison 1: even-split ez-ref SD  (total vs clean, {n_rep} repeats)",
        ylabel="ez-ref SD",
        dest=figdir / "comp1_ez_sd.png",
    )
    plot_comparison(
        df,
        metric="mu",
        arms=[
            ("even_total", EVEN_COLOR, "even split"),
            ("loo_total", LOO_COLOR, "LOO cross-fit"),
        ],
        title=f"Comparison 2: ez-ref mean on total candidates  (even split vs LOO, {n_rep} repeats)",
        ylabel="ez-ref mean",
        dest=figdir / "comp2_ez_mu.png",
    )
    plot_comparison(
        df,
        metric="sd",
        arms=[
            ("even_total", EVEN_COLOR, "even split"),
            ("loo_total", LOO_COLOR, "LOO cross-fit"),
        ],
        title=f"Comparison 2: ez-ref SD on total candidates  (even split vs LOO, {n_rep} repeats)",
        ylabel="ez-ref SD",
        dest=figdir / "comp2_ez_sd.png",
    )
    if (df["arm"] == "loo_clean").any():
        plot_comparison(
            df,
            metric="mu",
            arms=[
                ("even_clean", EVEN_COLOR, "even split"),
                ("loo_clean", LOO_COLOR, "LOO cross-fit"),
            ],
            title=f"Comparison 3: ez-ref mean on clean candidates  (even split vs LOO, {n_rep} repeats)",
            ylabel="ez-ref mean",
            dest=figdir / "comp3_ez_mu.png",
        )
        plot_comparison(
            df,
            metric="sd",
            arms=[
                ("even_clean", EVEN_COLOR, "even split"),
                ("loo_clean", LOO_COLOR, "LOO cross-fit"),
            ],
            title=f"Comparison 3: ez-ref SD on clean candidates  (even split vs LOO, {n_rep} repeats)",
            ylabel="ez-ref SD",
            dest=figdir / "comp3_ez_sd.png",
        )
    write_report(df, cfg, report_path)
    return df


def _self_check(arrays: dict, pool_idx: np.ndarray, n: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    drawn = pool_idx[rng.permutation(pool_idx.size)[:n]]
    half = n // 2
    ref_idx = drawn[:half]
    ez_idx = drawn[half:]
    comb = combined_on_query(
        arrays["hypo"],
        arrays["hyper"],
        arrays["hypo_cnt"],
        arrays["hyper_cnt"],
        arrays["pct"],
        ref_idx,
        ez_idx,
    )
    hypo_b = arrays["hypo"][None, :, :]
    hyper_b = arrays["hyper"][None, :, :]
    hc_b = arrays["hypo_cnt"][None, :, :]
    xc_b = arrays["hyper_cnt"][None, :, :]
    pct_b = arrays["pct"][None, :, :]
    ep = compute_episcore(hypo_b, hyper_b, hc_b, xc_b, ref_idx)[0][:, ez_idx]
    zs = compute_zscore(pct_b, ref_idx)[0][:, ez_idx]
    ref = ep + zs
    max_diff = float(np.nanmax(np.abs(comb - ref)))
    if max_diff > 1e-8:
        raise click.ClickException(f"even-split combined disagrees with compute_episcore: max|{max_diff}|")
    loo = loo_combined(
        arrays["hypo"][:, drawn],
        arrays["hyper"][:, drawn],
        arrays["hypo_cnt"][:, drawn],
        arrays["hyper_cnt"][:, drawn],
        arrays["pct"][:, drawn],
    )
    loo_ref = np.empty_like(loo)
    local = np.arange(n)
    for i in range(n):
        ref_i = np.delete(local, i)
        loo_ref[:, i] = combined_on_query(
            arrays["hypo"][:, drawn],
            arrays["hyper"][:, drawn],
            arrays["hypo_cnt"][:, drawn],
            arrays["hyper_cnt"][:, drawn],
            arrays["pct"][:, drawn],
            ref_i,
            np.array([i], dtype=np.int64),
        )[:, 0]
    max_loo = float(np.nanmax(np.abs(loo - loo_ref)))
    if max_loo > 1e-8:
        raise click.ClickException(f"LOO combined disagrees with explicit leave-one-out: max|{max_loo}|")
    console.print(f"[green]self-check OK[/green] even max|{max_diff:.2e}|  loo max|{max_loo:.2e}|")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--pool-sizes", default="20,220,2", show_default=True, help="min,max,step or comma list")
@click.option("--pool-size", default=None, type=int, help="Single pool size (SLURM array)")
@click.option("--total-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option(
    "--arms",
    default="all",
    show_default=True,
    help="Comma list of arms to run, or 'all'",
)
@click.option("--plot-only", is_flag=True, default=False)
@click.option("--self-check", "do_self_check", is_flag=True, default=False)
def main(
    parquet: str,
    meta: str,
    toxic: str,
    output_dir: str,
    pool_sizes: str,
    pool_size: int | None,
    total_repeats: int,
    seed: int,
    n_jobs: int,
    arms: str,
    plot_only: bool,
    do_self_check: bool,
) -> None:
    out = Path(output_dir)
    stat_dir = out / "stats"
    figdir = out / "figures"
    parts = [p.strip() for p in pool_sizes.split(",") if p.strip()]
    if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
        lo, hi, step = map(int, parts)
        sizes = list(range(lo, hi + 1, step))
        pool_step = step
    else:
        sizes = sorted({int(p) for p in parts})
        pool_step = min(np.diff(sizes)) if len(sizes) > 1 else 2
    if pool_size is not None:
        sizes = [int(pool_size)]

    cfg_path = out / "config.json"
    cfg = {
        "parquet": str(Path(parquet).resolve()),
        "meta": str(Path(meta).resolve()),
        "toxic": str(Path(toxic).resolve()),
        "total_repeats": int(total_repeats),
        "seed": int(seed),
        "pool_sizes": f"{min(sizes) if sizes else ''},{max(sizes) if sizes else ''},{pool_step}",
        "pool_step": int(pool_step),
    }
    prev = _read_json(cfg_path)
    if prev:
        merged = {**prev, **cfg}
        if pool_size is not None and prev.get("pool_sizes"):
            merged["pool_sizes"] = prev["pool_sizes"]
            merged["pool_step"] = prev.get("pool_step", merged["pool_step"])
        cfg = merged

    if plot_only:
        plot_all(stat_dir, figdir, cfg, out / "REPORT.md")
        return

    pack = load_candidates(Path(parquet), Path(meta), Path(toxic))
    cfg["n_total"] = len(pack["total_samples"])
    cfg["n_clean"] = len(pack["clean_samples"])
    cfg["n_toxic"] = pack["n_toxic"]
    cfg["n_nan"] = pack["n_nan"]
    out.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)
    (out / "candidates_total.txt").write_text("\n".join(pack["total_samples"]) + "\n")
    (out / "candidates_clean.txt").write_text("\n".join(pack["clean_samples"]) + "\n")
    _write_json_atomic(cfg_path, cfg)
    console.print(
        f"total={cfg['n_total']} clean={cfg['n_clean']} toxic={cfg['n_toxic']} "
        f"nan={pack['n_nan']} repeats={total_repeats}"
    )
    if do_self_check:
        _self_check(pack["arrays"], pack["total_idx"])

    workers = _resolve_n_jobs(n_jobs)
    if arms.strip().lower() == "all":
        want_arms = list(ARMS)
    else:
        want_arms = [a.strip() for a in arms.split(",") if a.strip()]
        bad = [a for a in want_arms if a not in ARMS]
        if bad:
            raise click.ClickException(f"unknown arms {bad}; choose from {ARMS}")
    arm_jobs = {
        "even_total": (pack["total_idx"], "even"),
        "even_clean": (pack["clean_idx"], "even"),
        "loo_total": (pack["total_idx"], "loo"),
        "loo_clean": (pack["clean_idx"], "loo"),
    }
    for p in sizes:
        console.rule(f"[cyan]pool={p}")
        pdir = stat_dir / f"pool_{p}"
        pdir.mkdir(parents=True, exist_ok=True)
        result = {"pool_size": p, "n_repeats": total_repeats}
        for arm in want_arms:
            idx, mode = arm_jobs[arm]
            mu, sd = run_arm(
                arrays=pack["arrays"],
                pool_idx=idx,
                pool_size=p,
                n_repeats=total_repeats,
                seed=arm_seed(seed, p, arm),
                n_jobs=workers,
                mode=mode,
            )
            entry = {"mu": summarize(mu), "sd": summarize(sd)}
            if arm.startswith("loo") and p == 220:
                e_mu, e_sd = expected_ez_params(mu, sd)
                entry["e_mu"] = e_mu.astype(np.float32)
                entry["e_sd"] = e_sd.astype(np.float32)
                np.savez_compressed(
                    pdir / f"{arm}_raw.npz",
                    ez_mu=mu.astype(np.float32),
                    ez_sd=sd.astype(np.float32),
                    e_mu=entry["e_mu"],
                    e_sd=entry["e_sd"],
                )
            result[arm] = entry
            console.print(
                f"  {arm} mu={float(np.nanmean(mu)):.4f} sd={float(np.nanmean(sd)):.4f}"
            )
        save_pool_summary(pdir / "summary.npz", result, merge_existing=True)
        (pdir / "run_config.json").write_text(
            json.dumps(
                {
                    "pool_size": p,
                    "n_repeats": total_repeats,
                    "seed": seed,
                    "n_jobs": workers,
                    "arms": want_arms,
                },
                indent=2,
            )
            + "\n"
        )

    # Array tasks only write stats; plots are submitted separately (--plot-only).
    if pool_size is None:
        have_loo_clean = 0
        for npz in sorted(stat_dir.glob("pool_*/summary.npz")):
            try:
                d = np.load(npz, allow_pickle=False)
            except (OSError, ValueError, zipfile.BadZipFile):
                continue
            if "loo_clean_mu_q" in d.files:
                have_loo_clean += 1
        expected_n = len(list(range(20, 221, 2)))
        if have_loo_clean >= expected_n or (
            len(list(stat_dir.glob("pool_*/summary.npz"))) >= expected_n
            and "loo_clean" not in want_arms
        ):
            plot_all(stat_dir, figdir, cfg, out / "REPORT.md")


if __name__ == "__main__":
    main()