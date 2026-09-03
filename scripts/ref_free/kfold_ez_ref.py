#!/usr/bin/env python3
"""Fixed-pool ez-ref: even split vs K-fold (2…220=LOO).

Draw 220 Normals once from the clean pool (227; ``dev``/``test``,
``depth_qc=pass``, ``label=Normal``, MAD-toxic dropped). Then, on that
**fixed** panel, Monte-Carlo the ez-reference builder:

* **even split** — 110 epi/z refs score the other 110; ez μ/σ from those 110.
* **k-fold** — partition into *k* folds; each fold is scored against the
  complement; ez μ/σ from all 220 cross-fitted (episcore+zscore) values.
* **220-fold** is leave-one-out (LOO). On a fixed pool it is unique, so the
  10k-repeat “distribution” collapses to a point.

Features: ``intermediate_merged_batches_modeA.parquet`` after-MQ percentage
+ hypo/hyper z_intra + CpG counts, matching ``combined_on_query`` /
``loo_combined`` in ``pool_size_ez_ref_bands.py``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
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
for _p in (SCRIPT_DIR, REF40_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grid_search_ref40 import CHR_LIST  # noqa: E402
from pool_size_ez_ref_bands import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_PARQUET,
    DEFAULT_TOXIC,
    QUANTILES,
    _resolve_n_jobs,
    combined_on_query,
    expected_ez_params,
    load_candidates,
    loo_combined,
)

console = Console()

DEFAULT_OUT = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev/0824"
)
DEFAULT_POOL_N = 220
DEFAULT_REPEATS = 10_000
DEFAULT_SEED = 42

EVEN_COLOR = "#6A1B9A"
KFOLD_COLOR = "#0D47A1"
LOO_COLOR = "#E69F00"
ZERO_COLOR = "#9E9E9E"

EVEN_X = -22.0
_WORKER: dict = {}


def parse_k_values(text: str, n_pool: int) -> list[int]:
    """Parse ``2-220`` / ``2,3,5`` / ``2-10,20,220`` into unique sorted k."""
    vals: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            vals.update(range(lo, hi + 1))
        else:
            vals.add(int(part))
    out = sorted(k for k in vals if 2 <= k <= n_pool)
    if not out:
        raise click.ClickException(f"no valid k in [2, {n_pool}] from {text!r}")
    return out


def fold_ids(n: int, k: int) -> np.ndarray:
    """Contiguous fold ids of length ``n``; extra samples go to the first folds."""
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, n={n}], got {k}")
    sizes = np.full(k, n // k, dtype=np.int64)
    sizes[: n % k] += 1
    return np.repeat(np.arange(k, dtype=np.int32), sizes)


def _fold_mu_sd(x: np.ndarray, fold_id: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Leave-fold-out mean/SD (ddof=0) mapped back to each sample. ``x`` is (n_chr, n)."""
    finite = np.isfinite(x)
    x0 = np.where(finite, x, 0.0)
    n_chr = x.shape[0]
    fold_sum = np.zeros((n_chr, k), dtype=np.float64)
    fold_sum2 = np.zeros((n_chr, k), dtype=np.float64)
    fold_cnt = np.zeros((n_chr, k), dtype=np.float64)
    fin = finite.astype(np.float64)
    fid = fold_id.astype(np.int64, copy=False)
    for c in range(n_chr):
        fold_sum[c] = np.bincount(fid, weights=x0[c], minlength=k)
        fold_sum2[c] = np.bincount(fid, weights=x0[c] * x0[c], minlength=k)
        fold_cnt[c] = np.bincount(fid, weights=fin[c], minlength=k)
    total_sum = fold_sum.sum(axis=1, keepdims=True)
    total_sum2 = fold_sum2.sum(axis=1, keepdims=True)
    total_cnt = fold_cnt.sum(axis=1, keepdims=True)
    ref_cnt = total_cnt - fold_cnt
    ref_sum = total_sum - fold_sum
    ref_sum2 = total_sum2 - fold_sum2
    mu_f = np.divide(ref_sum, ref_cnt, out=np.zeros_like(ref_sum), where=ref_cnt > 0)
    var_f = np.divide(ref_sum2, ref_cnt, out=np.zeros_like(ref_sum), where=ref_cnt > 0) - mu_f * mu_f
    var_f = np.maximum(var_f, 0.0)
    sd_f = np.sqrt(var_f)
    mu_f = np.where(ref_cnt > 0, mu_f, 0.0)
    return mu_f[:, fid], sd_f[:, fid]


def _combine_tracks(
    hypo: np.ndarray,
    hyper: np.ndarray,
    hypo_cnt: np.ndarray,
    hyper_cnt: np.ndarray,
    pct: np.ndarray,
    hypo_mu: np.ndarray,
    hypo_sd: np.ndarray,
    hyper_mu: np.ndarray,
    hyper_sd: np.ndarray,
    pct_mu: np.ndarray,
    pct_sd: np.ndarray,
) -> np.ndarray:
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


def kfold_combined(
    hypo: np.ndarray,
    hyper: np.ndarray,
    hypo_cnt: np.ndarray,
    hyper_cnt: np.ndarray,
    pct: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    """(episcore + zscore) for every sample vs samples outside its fold."""
    k = int(fold_id.max()) + 1
    hypo_mu, hypo_sd = _fold_mu_sd(hypo, fold_id, k)
    hyper_mu, hyper_sd = _fold_mu_sd(hyper, fold_id, k)
    pct_mu, pct_sd = _fold_mu_sd(pct, fold_id, k)
    return _combine_tracks(
        hypo, hyper, hypo_cnt, hyper_cnt, pct,
        hypo_mu, hypo_sd, hyper_mu, hyper_sd, pct_mu, pct_sd,
    )


def make_perms(n: int, n_repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.stack([rng.permutation(n) for _ in range(n_repeats)]).astype(np.int32)


def _mu_sd_of_comb(comb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(comb, axis=1)
        sd = np.nanstd(comb, axis=1, ddof=0)
    return mu.astype(np.float32), sd.astype(np.float32)


def summarize(arr: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore"):
        q = np.nanpercentile(arr, QUANTILES, axis=0).astype(np.float32)
        mean = np.nanmean(arr, axis=0).astype(np.float32)
    return {"q": q, "mean": mean}


def select_fixed_pool(
    pack: dict,
    pool_n: int,
    seed: int,
) -> dict:
    clean_idx = pack["clean_idx"]
    clean_samples = pack["clean_samples"]
    n_clean = int(clean_idx.size)
    if n_clean < pool_n:
        raise click.ClickException(f"clean pool n={n_clean} < pool_n={pool_n}")
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(n_clean, size=pool_n, replace=False)
    pick.sort()
    pool_idx = clean_idx[pick]
    pool_samples = [clean_samples[i] for i in pick]
    held_local = np.setdiff1d(np.arange(n_clean), pick, assume_unique=False)
    held_samples = [clean_samples[i] for i in held_local]
    arrays = {k: v[:, pool_idx] for k, v in pack["arrays"].items()}
    cand = pack["cand"].set_index("sample")
    return {
        "pool_samples": pool_samples,
        "held_samples": held_samples,
        "n_clean": n_clean,
        "n_total": len(pack["total_samples"]),
        "n_toxic": pack["n_toxic"],
        "n_nan": pack["n_nan"],
        "arrays": arrays,
        "cand": cand,
        "pool_n": pool_n,
    }


def _pool_table(samples: list[str], cand: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("set", "ff_before_mq", "ff_after_mq", "depth_qc", "label") if c in cand.columns]
    rows = []
    for s in samples:
        rec = {"sample": s}
        if s in cand.index:
            for c in cols:
                rec[c] = cand.loc[s, c]
        rows.append(rec)
    return pd.DataFrame(rows)


def write_pool(out: Path, pool: dict, cfg: dict) -> None:
    pdir = out / "pool"
    pdir.mkdir(parents=True, exist_ok=True)
    _pool_table(pool["pool_samples"], pool["cand"]).to_csv(pdir / "fixed_pool_220.tsv", sep="\t", index=False)
    held = _pool_table(pool["held_samples"], pool["cand"])
    held.to_csv(pdir / "held_out.tsv", sep="\t", index=False)
    (pdir / "fixed_pool_220.txt").write_text("\n".join(pool["pool_samples"]) + "\n")
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    console.print(f"  pool n={pool['pool_n']} held_out={len(pool['held_samples'])} → {pdir}")


def _init_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)


def _run_even(perms: np.ndarray, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n_rep, n = perms.shape
    half = n // 2
    n_chr = arrays["hypo"].shape[0]
    mu = np.empty((n_rep, n_chr), dtype=np.float32)
    sd = np.empty((n_rep, n_chr), dtype=np.float32)
    hypo, hyper = arrays["hypo"], arrays["hyper"]
    hypo_cnt, hyper_cnt = arrays["hypo_cnt"], arrays["hyper_cnt"]
    pct = arrays["pct"]
    for i in range(n_rep):
        idx = perms[i]
        comb = combined_on_query(
            hypo, hyper, hypo_cnt, hyper_cnt, pct,
            idx[:half], idx[half:],
        )
        mu[i], sd[i] = _mu_sd_of_comb(comb)
    return mu, sd


def _run_k(k: int) -> tuple[int, np.ndarray, np.ndarray]:
    perms = _WORKER["perms"]
    arrays = _WORKER["arrays"]
    n_rep, n = perms.shape
    n_chr = arrays["hypo"].shape[0]
    hypo, hyper = arrays["hypo"], arrays["hyper"]
    hypo_cnt, hyper_cnt = arrays["hypo_cnt"], arrays["hyper_cnt"]
    pct = arrays["pct"]
    if k == n:
        comb = loo_combined(hypo, hyper, hypo_cnt, hyper_cnt, pct)
        m, s = _mu_sd_of_comb(comb)
        mu = np.broadcast_to(m, (n_rep, n_chr)).copy()
        sd = np.broadcast_to(s, (n_rep, n_chr)).copy()
        return k, mu, sd
    fid_template = fold_ids(n, k)
    mu = np.empty((n_rep, n_chr), dtype=np.float32)
    sd = np.empty((n_rep, n_chr), dtype=np.float32)
    fid = np.empty(n, dtype=np.int32)
    for i in range(n_rep):
        fid[perms[i]] = fid_template
        comb = kfold_combined(hypo, hyper, hypo_cnt, hyper_cnt, pct, fid)
        mu[i], sd[i] = _mu_sd_of_comb(comb)
    return k, mu, sd


def _run_even_span(span: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    a, b = span
    mu, sd = _run_even(_WORKER["perms"][a:b], _WORKER["arrays"])
    return a, b, mu, sd


def run_even_split(arrays: dict[str, np.ndarray], perms: np.ndarray, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    n_rep = int(perms.shape[0])
    workers = min(_resolve_n_jobs(n_jobs), n_rep)
    if workers <= 1:
        return _run_even(perms, arrays)
    spans = []
    base, rem = divmod(n_rep, workers)
    start = 0
    for i in range(workers):
        end = start + base + (1 if i < rem else 0)
        if end > start:
            spans.append((start, end))
        start = end
    n_chr = arrays["hypo"].shape[0]
    mu = np.empty((n_rep, n_chr), dtype=np.float32)
    sd = np.empty((n_rep, n_chr), dtype=np.float32)
    payload = {"perms": perms, "arrays": arrays}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_init_worker,
        initargs=(payload,),
    ) as pool:
        futs = [pool.submit(_run_even_span, span) for span in spans]
        for fut in as_completed(futs):
            a, b, m, s = fut.result()
            mu[a:b] = m
            sd[a:b] = s
    return mu, sd


def run_kfolds(
    arrays: dict[str, np.ndarray],
    perms: np.ndarray,
    k_values: list[int],
    n_jobs: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    payload = {"perms": perms, "arrays": arrays}
    workers = min(_resolve_n_jobs(n_jobs), max(1, len(k_values)))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if workers <= 1 or len(k_values) == 1:
        _init_worker(payload)
        for k in k_values:
            kk, mu, sd = _run_k(k)
            out[kk] = (mu, sd)
            console.print(f"  k={kk} μ={float(np.nanmean(mu)):.4f} σ={float(np.nanmean(sd)):.4f}")
        return out
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_init_worker,
        initargs=(payload,),
    ) as pool:
        futs = {pool.submit(_run_k, k): k for k in k_values}
        done = 0
        for fut in as_completed(futs):
            k, mu, sd = fut.result()
            out[k] = (mu, sd)
            done += 1
            if done == 1 or done == len(k_values) or done % 20 == 0:
                console.print(
                    f"  [{done}/{len(k_values)}] k={k} "
                    f"μ={float(np.nanmean(mu)):.4f} σ={float(np.nanmean(sd)):.4f}"
                )
    return out


def pack_quantiles(mu: np.ndarray, sd: np.ndarray) -> dict[str, np.ndarray]:
    sm, ss = summarize(mu), summarize(sd)
    e_mu, e_sd = expected_ez_params(mu, sd)
    return {
        "mu_q": sm["q"],
        "mu_mean": sm["mean"],
        "sd_q": ss["q"],
        "sd_mean": ss["mean"],
        "e_mu": e_mu.astype(np.float32),
        "e_sd": e_sd.astype(np.float32),
    }


def save_results(
    stat_dir: Path,
    even: tuple[np.ndarray, np.ndarray] | None,
    kfold: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    n_repeats: int,
    n_pool: int,
) -> None:
    stat_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "n_repeats": np.int32(n_repeats),
        "n_pool": np.int32(n_pool),
        "quantiles": np.asarray(QUANTILES, dtype=np.float32),
    }
    if even is not None:
        mu, sd = even
        np.savez_compressed(stat_dir / "even_split.npz", mu=mu, sd=sd)
        packed = pack_quantiles(mu, sd)
        for key, val in packed.items():
            payload[f"even_{key}"] = val
        console.print(f"  wrote {stat_dir / 'even_split.npz'}")
    if kfold:
        ks = np.array(sorted(kfold), dtype=np.int32)
        n_chr = next(iter(kfold.values()))[0].shape[1]
        nq = len(QUANTILES)
        mu_q = np.empty((ks.size, nq, n_chr), dtype=np.float32)
        sd_q = np.empty_like(mu_q)
        mu_mean = np.empty((ks.size, n_chr), dtype=np.float32)
        sd_mean = np.empty_like(mu_mean)
        e_mu = np.empty_like(mu_mean)
        e_sd = np.empty_like(mu_mean)
        for i, k in enumerate(ks):
            packed = pack_quantiles(*kfold[int(k)])
            mu_q[i] = packed["mu_q"]
            sd_q[i] = packed["sd_q"]
            mu_mean[i] = packed["mu_mean"]
            sd_mean[i] = packed["sd_mean"]
            e_mu[i] = packed["e_mu"]
            e_sd[i] = packed["e_sd"]
        payload["k_values"] = ks
        payload["kfold_mu_q"] = mu_q
        payload["kfold_sd_q"] = sd_q
        payload["kfold_mu_mean"] = mu_mean
        payload["kfold_sd_mean"] = sd_mean
        payload["kfold_e_mu"] = e_mu
        payload["kfold_e_sd"] = e_sd
        if n_pool in kfold:
            loo_mu, loo_sd = kfold[n_pool]
            payload["loo_mu"] = loo_mu[0]
            payload["loo_sd"] = loo_sd[0]
    tmp = stat_dir / ".summary.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, stat_dir / "summary.npz")
    console.print(f"  wrote {stat_dir / 'summary.npz'}")


def load_summary(stat_dir: Path) -> dict[str, np.ndarray]:
    path = stat_dir / "summary.npz"
    if not path.is_file():
        raise click.ClickException(f"missing {path}; run scoring first")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _iqr(q: np.ndarray) -> np.ndarray:
    """q is (5, n_chr) or (n_k, 5, n_chr); IQR = q75-q25."""
    if q.ndim == 2:
        return q[3] - q[1]
    return q[:, 3, :] - q[:, 1, :]


def _p05_95(q: np.ndarray) -> np.ndarray:
    if q.ndim == 2:
        return q[4] - q[0]
    return q[:, 4, :] - q[:, 0, :]


def ranking_frame(summary: dict[str, np.ndarray], n_pool: int) -> pd.DataFrame:
    rows = []
    if "even_mu_q" in summary:
        mu_q = summary["even_mu_q"]
        sd_q = summary["even_sd_q"]
        rows.append(
            {
                "strategy": "even_split",
                "k": np.nan,
                "x_label": "even split",
                "n_ez": n_pool // 2,
                "mean_epiz_ref": float(n_pool // 2),
                "mu_iqr_median_chr": float(np.median(_iqr(mu_q))),
                "sd_iqr_median_chr": float(np.median(_iqr(sd_q))),
                "mu_p05_95_median_chr": float(np.median(_p05_95(mu_q))),
                "sd_p05_95_median_chr": float(np.median(_p05_95(sd_q))),
                "mu_abs_mean_median_chr": float(np.median(np.abs(summary["even_mu_mean"]))),
                "sd_q50_median_chr": float(np.median(sd_q[2])),
                "e_mu_median_chr": float(np.median(summary["even_e_mu"])),
                "e_sd_median_chr": float(np.median(summary["even_e_sd"])),
            }
        )
    if "k_values" in summary:
        ks = summary["k_values"]
        mu_q = summary["kfold_mu_q"]
        sd_q = summary["kfold_sd_q"]
        for i, k in enumerate(ks):
            k = int(k)
            label = f"{k}-fold" + (" (LOO)" if k == n_pool else "")
            rows.append(
                {
                    "strategy": f"kfold_{k}" + ("_loo" if k == n_pool else ""),
                    "k": k,
                    "x_label": label,
                    "n_ez": n_pool,
                    "mean_epiz_ref": float(n_pool * (k - 1) / k),
                    "mu_iqr_median_chr": float(np.median(_iqr(mu_q[i]))),
                    "sd_iqr_median_chr": float(np.median(_iqr(sd_q[i]))),
                    "mu_p05_95_median_chr": float(np.median(_p05_95(mu_q[i]))),
                    "sd_p05_95_median_chr": float(np.median(_p05_95(sd_q[i]))),
                    "mu_abs_mean_median_chr": float(np.median(np.abs(summary["kfold_mu_mean"][i]))),
                    "sd_q50_median_chr": float(np.median(sd_q[i, 2])),
                    "e_mu_median_chr": float(np.median(summary["kfold_e_mu"][i])),
                    "e_sd_median_chr": float(np.median(summary["kfold_e_sd"][i])),
                }
            )
    if not rows:
        raise click.ClickException("summary.npz has neither even_split nor k-fold stats")
    return pd.DataFrame(rows)


def percentiles_frame(summary: dict[str, np.ndarray], n_pool: int) -> pd.DataFrame:
    rows = []
    qnames = ["q05", "q25", "q50", "q75", "q95"]

    def _add(strategy: str, k, metric: str, q: np.ndarray, mean: np.ndarray) -> None:
        for ci, chr_name in enumerate(CHR_LIST):
            rec = {
                "strategy": strategy,
                "k": k,
                "metric": metric,
                "chr": chr_name,
                "mean": float(mean[ci]),
            }
            for qi, name in enumerate(qnames):
                rec[name] = float(q[qi, ci])
            rows.append(rec)

    if "even_mu_q" in summary:
        _add("even_split", np.nan, "mu", summary["even_mu_q"], summary["even_mu_mean"])
        _add("even_split", np.nan, "sd", summary["even_sd_q"], summary["even_sd_mean"])
    if "k_values" in summary:
        for i, k in enumerate(summary["k_values"]):
            k = int(k)
            name = f"kfold_{k}" + ("_loo" if k == n_pool else "")
            _add(name, k, "mu", summary["kfold_mu_q"][i], summary["kfold_mu_mean"][i])
            _add(name, k, "sd", summary["kfold_sd_q"][i], summary["kfold_sd_mean"][i])
    return pd.DataFrame(rows)


def _elbow_k(rank: pd.DataFrame, col: str, frac: float = 0.1) -> int | None:
    """Smallest k that captures (1-frac) of the IQR drop from 2-fold to LOO."""
    kf = rank.dropna(subset=["k"]).sort_values("k")
    if kf.empty:
        return None
    v2 = float(kf.loc[kf["k"] == 2, col].iloc[0]) if (kf["k"] == 2).any() else float(kf.iloc[0][col])
    vloo = float(kf.iloc[-1][col])
    span = v2 - vloo
    if span <= 0:
        return int(kf.iloc[-1]["k"])
    thresh = vloo + frac * span
    hit = kf.loc[kf[col] <= thresh]
    if hit.empty:
        return int(kf.iloc[-1]["k"])
    return int(hit.iloc[0]["k"])


def _draw_even_box(ax, q: np.ndarray, ci: int, color: str) -> None:
    q05, q25, q50, q75, q95 = (float(q[i, ci]) for i in range(5))
    width = 5.5
    ax.fill_between(
        [EVEN_X - width / 2, EVEN_X + width / 2],
        [q25, q25],
        [q75, q75],
        color=color,
        alpha=0.35,
        linewidth=0,
        zorder=4,
    )
    ax.plot([EVEN_X, EVEN_X], [q05, q25], color=color, lw=1.4, zorder=5, solid_capstyle="round")
    ax.plot([EVEN_X, EVEN_X], [q75, q95], color=color, lw=1.4, zorder=5, solid_capstyle="round")
    cap = 2.2
    ax.plot([EVEN_X - cap, EVEN_X + cap], [q05, q05], color=color, lw=1.4, zorder=5)
    ax.plot([EVEN_X - cap, EVEN_X + cap], [q95, q95], color=color, lw=1.4, zorder=5)
    ax.plot(
        [EVEN_X - width / 2, EVEN_X + width / 2],
        [q50, q50],
        color=color,
        lw=2.0,
        zorder=6,
        solid_capstyle="round",
    )


def _style_ax(ax, n_pool: int, *, ylabel: str, show_x: bool, show_y: bool) -> None:
    ax.axvline(0.0, color="#BDBDBD", lw=0.8, ls=":", zorder=0)
    ax.axvline(n_pool, color=LOO_COLOR, lw=1.1, ls="--", alpha=0.85, zorder=3)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(EVEN_X - 12, n_pool + 8)
    ticks = [EVEN_X, 2, 50, 100, 150, n_pool]
    labels = ["even\nsplit", "2", "50", "100", "150", f"{n_pool}\n(LOO)"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if show_x else [""] * len(labels), fontsize=7)
    if show_y:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)


def plot_metric(
    summary: dict[str, np.ndarray],
    *,
    metric: str,
    ylabel: str,
    title: str,
    dest: Path,
    n_pool: int,
    zero_line: bool = False,
) -> None:
    fig, axes = plt.subplots(4, 6, figsize=(20.5, 12.6), sharex=True)
    axes = axes.ravel()
    even_q = summary.get(f"even_{metric}_q")
    ks = summary.get("k_values")
    k_q = summary.get(f"kfold_{metric}_q")
    loo_y = summary.get(f"loo_{'mu' if metric == 'mu' else 'sd'}")
    for i, chr_name in enumerate(CHR_LIST):
        ax = axes[i]
        if even_q is not None:
            _draw_even_box(ax, even_q, i, EVEN_COLOR)
        if ks is not None and k_q is not None:
            x = ks.astype(np.float64)
            q05, q25, q50, q75, q95 = (k_q[:, j, i] for j in range(5))
            ax.fill_between(x, q05, q95, color=KFOLD_COLOR, alpha=0.18, linewidth=0, zorder=1)
            ax.fill_between(x, q25, q75, color=KFOLD_COLOR, alpha=0.38, linewidth=0, zorder=2)
            ax.plot(x, q50, color=KFOLD_COLOR, lw=1.6, zorder=4, solid_capstyle="round")
            if int(ks[-1]) == n_pool:
                ax.scatter(
                    [n_pool],
                    [q50[-1]],
                    s=28,
                    color=LOO_COLOR,
                    zorder=7,
                    edgecolors="white",
                    linewidths=0.4,
                )
        if zero_line:
            ax.axhline(0.0, color=ZERO_COLOR, lw=0.7, ls="-", zorder=0)
        # μ: even-split 5–95% is ~30× k-fold; symlog keeps both visible.
        if metric == "mu":
            ax.set_yscale("symlog", linthresh=0.04, linscale=2.2)
            ylo, yhi = -0.55, 0.55
            if even_q is not None:
                ylo = min(ylo, float(even_q[0, i]) - 0.02)
                yhi = max(yhi, float(even_q[4, i]) + 0.02)
            ax.set_ylim(ylo, yhi)
        ax.set_title(chr_name, fontsize=10)
        show_x = i >= 18
        show_y = i % 6 == 0
        _style_ax(ax, n_pool, ylabel=ylabel, show_x=show_x, show_y=show_y)
        if show_x:
            ax.set_xlabel("strategy", fontsize=8)
    for j in range(len(CHR_LIST), len(axes)):
        axes[j].axis("off")
    handles = [
        Line2D([0], [0], color=EVEN_COLOR, lw=2.0, label="even split (110 ez-ref)"),
        Line2D([0], [0], color=KFOLD_COLOR, lw=1.6, label="k-fold median"),
        Patch(facecolor=KFOLD_COLOR, alpha=0.32, label="k-fold IQR"),
        Patch(facecolor=KFOLD_COLOR, alpha=0.16, label="k-fold 5–95%"),
        Line2D([0], [0], color=LOO_COLOR, lw=1.2, ls="--", label=f"{n_pool}-fold (LOO)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    console.print(f"  wrote {dest}")
    _ = loo_y


def plot_stability(rank: pd.DataFrame, dest: Path, n_pool: int) -> None:
    kf = rank.dropna(subset=["k"]).sort_values("k")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    specs = [
        ("mu_iqr_median_chr", "median-across-chr IQR of μ", axes[0]),
        ("sd_iqr_median_chr", "median-across-chr IQR of σ", axes[1]),
    ]
    even = rank.loc[rank["strategy"] == "even_split"]
    for col, title, ax in specs:
        if not kf.empty:
            ax.fill_between(kf["k"], 0.0, kf[col], color=KFOLD_COLOR, alpha=0.12, linewidth=0)
            ax.plot(kf["k"], kf[col], color=KFOLD_COLOR, lw=1.8)
            loo_val = kf.loc[kf["k"] == n_pool, col]
            if not loo_val.empty:
                ax.scatter([n_pool], [float(loo_val.iloc[0])], color=LOO_COLOR, s=36, zorder=5, label="LOO")
        if not even.empty:
            ax.scatter(
                [EVEN_X],
                even[col],
                color=EVEN_COLOR,
                s=48,
                marker="D",
                zorder=5,
                label="even split",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("k  (220 = LOO)")
        ax.set_ylabel("IQR")
        ax.axvline(n_pool, color=LOO_COLOR, ls="--", lw=1.0, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(EVEN_X - 12, n_pool + 8)
        ax.set_xticks([EVEN_X, 2, 50, 100, 150, n_pool])
        ax.set_xticklabels(["even", "2", "50", "100", "150", "220\n(LOO)"], fontsize=8)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    console.print(f"  wrote {dest}")


def write_report(
    out: Path,
    cfg: dict,
    rank: pd.DataFrame,
    summary: dict[str, np.ndarray],
) -> None:
    n_pool = int(cfg["pool_n"])
    n_rep = int(cfg["total_repeats"])
    even = rank.loc[rank["strategy"] == "even_split"]
    kf = rank.dropna(subset=["k"]).sort_values("k")
    row2 = kf.loc[kf["k"] == 2].iloc[0] if (kf["k"] == 2).any() else None
    row_loo = kf.loc[kf["k"] == n_pool].iloc[0] if (kf["k"] == n_pool).any() else kf.iloc[-1]
    elbow_mu = _elbow_k(rank, "mu_iqr_median_chr")
    elbow_sd = _elbow_k(rank, "sd_iqr_median_chr")

    def _fmt_row(row) -> str:
        return (
            f"IQR(μ)={row['mu_iqr_median_chr']:.4f}, IQR(σ)={row['sd_iqr_median_chr']:.4f}, "
            f"median σ={row['sd_q50_median_chr']:.3f}, |E(μ)|={row['mu_abs_mean_median_chr']:.4f}"
        )

    even_txt = _fmt_row(even.iloc[0]) if not even.empty else "NA"
    two_txt = _fmt_row(row2) if row2 is not None else "NA"
    loo_txt = _fmt_row(row_loo)

    best_name = "220-fold (LOO)"
    reason = (
        "On a **fixed** 220-sample panel, 220-fold is leave-one-out: every sample is "
        "scored against the other 219, and the ez-ref (μ, σ) is unique. Repeat-to-repeat "
        "IQR of μ and σ is therefore zero — the most stable ez-ref possible from this pool."
    )
    if row2 is not None and not even.empty:
        iqr_mu_ratio = float(even.iloc[0]["mu_iqr_median_chr"] / max(row2["mu_iqr_median_chr"], 1e-12))
        iqr_sd_ratio = float(even.iloc[0]["sd_iqr_median_chr"] / max(row2["sd_iqr_median_chr"], 1e-12))
        even_vs_2 = (
            f"Even split vs 2-fold (same 110/110 cuts): even-split IQR(μ) is "
            f"{iqr_mu_ratio:.2f}× 2-fold, IQR(σ) is {iqr_sd_ratio:.2f}× 2-fold. "
            f"For mean-only z-scoring on a fixed pool, 2-fold μ is identically 0 "
            f"(the two halves cancel: μ_A = −μ_B) while even-split μ = 2(x̄_Q − x̄) "
            f"moves with the cut; |I|=110 vs 220 is secondary. See DERIVATION.md."
        )
    else:
        even_vs_2 = ""

    report = f"""# Ez-ref builder: even split vs K-fold (0824)

Fixed Normal panel **n={n_pool}**, drawn once (seed={cfg['seed']}) from the clean pool
(n={cfg['n_clean']}; `set ∈ {{dev, test}}`, `depth_qc=pass`, `label=Normal`, MAD-toxic dropped).
Repeats = **{n_rep}**. Shared permutations across strategies (even split and 2-fold use the
same 110/110 cuts).

Features: `{Path(cfg['parquet']).name}` after-MQ percentage + hypo/hyper z_intra + CpG counts.

## Designs

| Strategy | epi/z reference | ez-ref scores | n for μ/σ |
|----------|-----------------|---------------|-----------|
| even split | 110 | the other 110 | 110 |
| k-fold | n × (k−1)/k per fold | all 220 cross-fitted | 220 |
| 220-fold (**LOO**) | 219 | all 220 leave-one-out | 220 |

## Figures

![mu](figures/ez_mu_by_strategy.png)
![sd](figures/ez_sd_by_strategy.png)
![stability](figures/strategy_stability.png)

Left of the dotted line: even split (box = IQR, whiskers = 5–95%). Then k = 2…{n_pool}
as median + IQR + 5–95% bands. Orange dashed line marks **{n_pool}-fold (LOO)**.
μ panels use a symlog y-axis (linear inside ±0.04) so the narrow k-fold bands stay visible
next to the much wider even-split box.

## What is best?

**{best_name}.** {reason}

- even split: {even_txt}
- 2-fold: {two_txt}
- 220-fold (LOO): {loo_txt}

{even_vs_2}

90% of the 2-fold→LOO IQR reduction is reached by **k={elbow_mu}** (μ) and **k={elbow_sd}** (σ)
(median across chromosomes). Larger k mainly buys a larger epi/z reference per fold and a
tighter Monte-Carlo distribution of the ez-ref parameters; on a frozen panel, LOO removes
that Monte-Carlo uncertainty entirely.

Full algebra (mean-only identities, LOO map \(S_i=n/(n-1)(x_i-\bar x)\), \(k\)-fold
leave-fold-out): [`DERIVATION.md`](DERIVATION.md).

## Outputs

- `pool/fixed_pool_220.tsv` — the 220-sample panel
- `pool/held_out.tsv` — the {cfg['n_clean'] - n_pool} unused clean Normals
- `stats/summary.npz` / `stats/percentiles.tsv` / `stats/ranking.tsv`
- `stats/even_split.npz` — 10k raw even-split μ/σ
"""
    dest = out / "REPORT.md"
    dest.write_text(report)
    console.print(f"  wrote {dest}")


def self_check(arrays: dict[str, np.ndarray], n: int = 40) -> None:
    rng = np.random.default_rng(0)
    idx = rng.permutation(arrays["hypo"].shape[1])[:n]
    hypo = arrays["hypo"][:, idx]
    hyper = arrays["hyper"][:, idx]
    hypo_cnt = arrays["hypo_cnt"][:, idx]
    hyper_cnt = arrays["hyper_cnt"][:, idx]
    pct = arrays["pct"][:, idx]
    local = np.arange(n, dtype=np.int32)

    loo = loo_combined(hypo, hyper, hypo_cnt, hyper_cnt, pct)
    kf_n = kfold_combined(hypo, hyper, hypo_cnt, hyper_cnt, pct, local)
    d_loo = float(np.nanmax(np.abs(loo - kf_n)))
    if d_loo > 1e-8:
        raise click.ClickException(f"k=n kfold != loo_combined: max|{d_loo}|")

    half = n // 2
    fid2 = fold_ids(n, 2)
    kf2 = kfold_combined(hypo, hyper, hypo_cnt, hyper_cnt, pct, fid2)
    a = combined_on_query(hypo, hyper, hypo_cnt, hyper_cnt, pct, local[half:], local[:half])
    b = combined_on_query(hypo, hyper, hypo_cnt, hyper_cnt, pct, local[:half], local[half:])
    ref2 = np.empty_like(kf2)
    ref2[:, :half] = a
    ref2[:, half:] = b
    d2 = float(np.nanmax(np.abs(kf2 - ref2)))
    if d2 > 1e-8:
        raise click.ClickException(f"k=2 kfold != two combined_on_query: max|{d2}|")
    console.print(f"[green]self-check OK[/green] k=n vs LOO max|{d_loo:.2e}|  k=2 max|{d2:.2e}|")


def _load_cfg(out: Path) -> dict:
    path = out / "config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _common_options(fn):
    fn = click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))(fn)
    fn = click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))(fn)
    fn = click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))(fn)
    fn = click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))(fn)
    fn = click.option("--pool-n", default=DEFAULT_POOL_N, show_default=True, type=int)(fn)
    fn = click.option("--total-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)(fn)
    fn = click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)(fn)
    return fn


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Even split vs K-fold ez-ref on a fixed 220 Normal pool."""


@cli.command()
@_common_options
def prepare(parquet: str, meta: str, toxic: str, output_dir: str, pool_n: int, total_repeats: int, seed: int) -> None:
    """Draw the fixed 220-sample pool and write config."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pack = load_candidates(Path(parquet), Path(meta), Path(toxic))
    pool = select_fixed_pool(pack, pool_n, seed)
    cfg = {
        "parquet": str(Path(parquet).resolve()),
        "meta": str(Path(meta).resolve()),
        "toxic": str(Path(toxic).resolve()),
        "pool_n": int(pool_n),
        "total_repeats": int(total_repeats),
        "seed": int(seed),
        "perm_seed": int(seed) + 1009,
        "n_clean": pool["n_clean"],
        "n_total": pool["n_total"],
        "n_toxic": pool["n_toxic"],
        "n_nan": pool["n_nan"],
        "n_held_out": len(pool["held_samples"]),
    }
    write_pool(out, pool, cfg)
    console.print(
        f"clean={cfg['n_clean']} pool={pool_n} held_out={cfg['n_held_out']} "
        f"repeats={total_repeats} seed={seed}"
    )


@cli.command()
@_common_options
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option("--k-values", default="2-220", show_default=True, help="k list or ranges, e.g. 2-220 or 2,5,10,220")
@click.option("--skip-even", is_flag=True, default=False)
@click.option("--even-only", is_flag=True, default=False)
@click.option("--self-check", "do_self_check", is_flag=True, default=False)
def run(
    parquet: str,
    meta: str,
    toxic: str,
    output_dir: str,
    pool_n: int,
    total_repeats: int,
    seed: int,
    n_jobs: int,
    k_values: str,
    skip_even: bool,
    even_only: bool,
    do_self_check: bool,
) -> None:
    """10k-repeat even split + k-fold μ/σ."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pack = load_candidates(Path(parquet), Path(meta), Path(toxic))
    pool = select_fixed_pool(pack, pool_n, seed)
    prev = _load_cfg(out)
    cfg = {
        **prev,
        "parquet": str(Path(parquet).resolve()),
        "meta": str(Path(meta).resolve()),
        "toxic": str(Path(toxic).resolve()),
        "pool_n": int(pool_n),
        "total_repeats": int(total_repeats),
        "seed": int(seed),
        "perm_seed": int(seed) + 1009,
        "n_clean": pool["n_clean"],
        "n_total": pool["n_total"],
        "n_toxic": pool["n_toxic"],
        "n_nan": pool["n_nan"],
        "n_held_out": len(pool["held_samples"]),
        "k_values": k_values if not even_only else "",
    }
    write_pool(out, pool, cfg)
    arrays = pool["arrays"]
    if do_self_check:
        self_check(arrays, n=min(40, pool_n))
    perms = make_perms(pool_n, total_repeats, cfg["perm_seed"])
    workers = _resolve_n_jobs(n_jobs)
    console.print(f"repeats={total_repeats} n_jobs={workers} k={k_values!r} even={not skip_even and not even_only or even_only}")

    even_res = None
    if not skip_even:
        console.rule("[cyan]even split")
        even_res = run_even_split(arrays, perms, workers)
        console.print(
            f"  even μ={float(np.nanmean(even_res[0])):.4f} "
            f"σ={float(np.nanmean(even_res[1])):.4f}"
        )

    kfold_res: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not even_only:
        ks = parse_k_values(k_values, pool_n)
        cfg["k_values"] = ",".join(str(k) for k in ks) if len(ks) < 30 else f"{ks[0]}-{ks[-1]}"
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        console.rule(f"[cyan]k-fold n_k={len(ks)}")
        kfold_res = run_kfolds(arrays, perms, ks, workers)

    save_results(out / "stats", even_res, kfold_res, n_repeats=total_repeats, n_pool=pool_n)


@cli.command()
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
def plot(output_dir: str) -> None:
    """22-chr μ/σ figures + ranking report."""
    out = Path(output_dir)
    cfg = _load_cfg(out)
    n_pool = int(cfg.get("pool_n", DEFAULT_POOL_N))
    summary = load_summary(out / "stats")
    rank = ranking_frame(summary, n_pool)
    perc = percentiles_frame(summary, n_pool)
    (out / "stats").mkdir(parents=True, exist_ok=True)
    rank.to_csv(out / "stats" / "ranking.tsv", sep="\t", index=False)
    perc.to_csv(out / "stats" / "percentiles.tsv", sep="\t", index=False)
    figdir = out / "figures"
    plot_metric(
        summary,
        metric="mu",
        ylabel="ez-ref μ",
        title="Ez-ref μ by builder  ·  even split, then 2-fold … 220-fold (LOO)",
        dest=figdir / "ez_mu_by_strategy.png",
        n_pool=n_pool,
        zero_line=True,
    )
    plot_metric(
        summary,
        metric="sd",
        ylabel="ez-ref σ",
        title="Ez-ref σ by builder  ·  even split, then 2-fold … 220-fold (LOO)",
        dest=figdir / "ez_sd_by_strategy.png",
        n_pool=n_pool,
        zero_line=False,
    )
    plot_stability(rank, figdir / "strategy_stability.png", n_pool)
    write_report(out, cfg, rank, summary)


@cli.command("self-check")
@_common_options
def self_check_cmd(parquet: str, meta: str, toxic: str, output_dir: str, pool_n: int, total_repeats: int, seed: int) -> None:
    """Algebraic k-fold vs explicit combined_on_query / LOO."""
    del output_dir, total_repeats
    pack = load_candidates(Path(parquet), Path(meta), Path(toxic))
    pool = select_fixed_pool(pack, pool_n, seed)
    self_check(pool["arrays"], n=min(40, pool_n))


if __name__ == "__main__":
    cli()
