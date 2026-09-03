#!/usr/bin/env python3
"""Monte-Carlo ez-ref mean/SD vs pool size (20…160), 22 chromosome panels.

For each even pool size, draw ``pool_size//2`` epi/z refs and the same for ez
refs from the 96 dev-Normal pool (plus nested test-Normal fillers when
pool_size>96). Record per-chr mean and SD of (episcore+zscore) on the ez-ref
group across ``--total-repeats`` repeats, then box-plot vs pool size.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

from common import (
    CHR_LIST,
    DEFAULT_BLACKLIST,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUT_BASE,
    DEFAULT_SEED,
    _generate_half_partitions,
    compute_episcore,
    compute_zscore,
    load_universe,
)

console = Console()
_WORKER: dict = {}
DEFAULT_FILL_SEED = 7
DEFAULT_REPEATS = 10_000


def _build_candidate_pool(
    set_arr: np.ndarray,
    label_arr: np.ndarray,
    pool_size: int,
    fill_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    is_normal = label_arr == "Normal"
    dev_idx = np.flatnonzero((set_arr == "dev") & is_normal).astype(np.int64)
    test_idx = np.flatnonzero((set_arr == "test") & is_normal).astype(np.int64)
    if pool_size <= dev_idx.size:
        return dev_idx, np.asarray([], dtype=np.int64)
    n_fill = pool_size - int(dev_idx.size)
    if n_fill > test_idx.size:
        raise click.ClickException(
            f"Need {n_fill} test Normal fillers, found {test_idx.size}"
        )
    rng = np.random.default_rng(fill_seed)
    ordered = test_idx[rng.permutation(test_idx.size)]
    fillers = np.sort(ordered[:n_fill])
    return np.concatenate([dev_idx, fillers]), fillers


def _init_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)
    ep = _WORKER["ep_arrays"]
    z = _WORKER["z_array"]
    _WORKER["ep_hypo"] = np.expand_dims(ep[0], 0)
    _WORKER["ep_hyper"] = np.expand_dims(ep[1], 0)
    _WORKER["ep_hypo_cnt"] = np.expand_dims(ep[2], 0)
    _WORKER["ep_hyper_cnt"] = np.expand_dims(ep[3], 0)
    _WORKER["z_pct"] = np.expand_dims(z, 0)


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


def _run_chunk(span: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, end = span
    candidate = _WORKER["candidate"]
    ref_draws = _WORKER["ref_draws"]
    ez_draws = _WORKER["ez_draws"]
    n_chr = int(_WORKER["n_chr"])
    mu = np.zeros((end - start, n_chr), dtype=np.float32)
    sd = np.zeros((end - start, n_chr), dtype=np.float32)
    for i, rid in enumerate(range(start, end)):
        ref_idx = candidate[ref_draws[rid]]
        ez_ref_idx = candidate[ez_draws[rid]]
        episcore = compute_episcore(
            _WORKER["ep_hypo"],
            _WORKER["ep_hyper"],
            _WORKER["ep_hypo_cnt"],
            _WORKER["ep_hyper_cnt"],
            ref_idx,
        )[0]
        zscore = compute_zscore(_WORKER["z_pct"], ref_idx)[0]
        combined = episcore + zscore
        with np.errstate(invalid="ignore"):
            mu[i] = np.nanmean(combined[:, ez_ref_idx], axis=1)
            sd[i] = np.nanstd(combined[:, ez_ref_idx], axis=1, ddof=0)
    return mu, sd


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs > 0:
        return int(n_jobs)
    import os

    for key in ("SLURM_CPUS_PER_TASK", "N_JOBS"):
        raw = os.environ.get(key)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return max(1, int(mp.cpu_count() or 1))


def run_one_pool(
    *,
    ctx: dict,
    pool_size: int,
    total_repeats: int,
    seed: int,
    fill_seed: int,
    n_jobs: int,
) -> dict:
    half = pool_size // 2
    candidate, fillers = _build_candidate_pool(
        ctx["set_arr"], ctx["label_arr"], pool_size, fill_seed
    )
    rng = np.random.default_rng(seed)
    ref_draws, ez_draws = _generate_half_partitions(
        pool_size=int(candidate.size), half=half, n_repeats=total_repeats, rng=rng
    )
    n_chr = ctx["z_array"].shape[0]
    workers = _resolve_n_jobs(n_jobs)
    spans = _chunk_spans(total_repeats, workers)
    payload = {
        "candidate": candidate,
        "ref_draws": ref_draws,
        "ez_draws": ez_draws,
        "n_chr": n_chr,
        "ep_arrays": ctx["ep_arrays"],
        "z_array": ctx["z_array"],
    }
    mu = np.zeros((total_repeats, n_chr), dtype=np.float32)
    sd = np.zeros((total_repeats, n_chr), dtype=np.float32)
    if workers == 1 or len(spans) == 1:
        _init_worker(payload)
        mu, sd = _run_chunk((0, total_repeats))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_worker,
            initargs=(payload,),
        ) as pool:
            futs = {pool.submit(_run_chunk, span): span for span in spans}
            for fut in as_completed(futs):
                start, end = futs[fut]
                m, s = fut.result()
                mu[start:end] = m
                sd[start:end] = s
    return {
        "ez_mu": mu,
        "ez_sd": sd,
        "n_candidate": int(candidate.size),
        "n_fillers": int(fillers.size),
        "filler_samples": [ctx["universe"][i] for i in fillers],
        "candidate_samples": [ctx["universe"][i] for i in candidate],
    }


def _box_figure(
    by_pool: dict[int, np.ndarray],
    *,
    ylabel: str,
    title: str,
) -> go.Figure:
    pools = sorted(by_pool)
    fig = make_subplots(
        rows=4,
        cols=6,
        subplot_titles=list(CHR_LIST),
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )
    for i, chr_name in enumerate(CHR_LIST):
        r, c = divmod(i, 6)
        q1, med, q3, lo, hi = [], [], [], [], []
        for p in pools:
            col = by_pool[p][:, i]
            qq = np.nanpercentile(col, [25, 50, 75])
            iqr = qq[2] - qq[0]
            q1.append(float(qq[0]))
            med.append(float(qq[1]))
            q3.append(float(qq[2]))
            lo.append(float(max(np.nanmin(col), qq[0] - 1.5 * iqr)))
            hi.append(float(min(np.nanmax(col), qq[2] + 1.5 * iqr)))
        fig.add_trace(
            go.Box(
                x=[str(p) for p in pools],
                q1=q1,
                median=med,
                q3=q3,
                lowerfence=lo,
                upperfence=hi,
                name=chr_name,
                showlegend=False,
                boxpoints=False,
                marker_color="#2E86AB",
                line=dict(width=1),
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1100,
        width=1400,
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text=ylabel, col=1)
    fig.update_xaxes(title_text="pool size", row=4)
    return fig


def _box_png(
    by_pool: dict[int, np.ndarray],
    *,
    ylabel: str,
    title: str,
    dest: Path,
) -> None:
    pools = sorted(by_pool)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), sharex=True)
    axes = axes.ravel()
    for i, chr_name in enumerate(CHR_LIST):
        ax = axes[i]
        data = [by_pool[p][:, i] for p in pools]
        ax.boxplot(
            data,
            labels=[str(p) for p in pools],
            showfliers=False,
            medianprops=dict(color="#C1121F", linewidth=1.2),
            boxprops=dict(color="#2E86AB"),
            whiskerprops=dict(color="#2E86AB"),
            capprops=dict(color="#2E86AB"),
        )
        ax.set_title(chr_name, fontsize=10)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        if i % 6 == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    for j in range(len(CHR_LIST), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def plot_all(stat_dir: Path, plot_dir: Path) -> None:
    by_mu: dict[int, np.ndarray] = {}
    by_sd: dict[int, np.ndarray] = {}
    for pdir in sorted(stat_dir.glob("pool_*")):
        npz = pdir / "ez_stats.npz"
        if not npz.is_file():
            continue
        try:
            pool = int(pdir.name.split("_")[1])
        except ValueError:
            continue
        d = np.load(npz)
        by_mu[pool] = d["ez_mu"]
        by_sd[pool] = d["ez_sd"]
    if not by_mu:
        raise click.ClickException(f"no pool_*/ez_stats.npz under {stat_dir}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    title_mu = (
        f"Ez-ref mean of (episcore+zscore) vs pool size "
        f"({min(by_mu)}–{max(by_mu)}, {next(iter(by_mu.values())).shape[0]} repeats)"
    )
    title_sd = (
        f"Ez-ref SD of (episcore+zscore) vs pool size "
        f"({min(by_sd)}–{max(by_sd)}, {next(iter(by_sd.values())).shape[0]} repeats)"
    )
    fig_mu = _box_figure(by_mu, ylabel="ez-ref mean", title=title_mu)
    fig_sd = _box_figure(by_sd, ylabel="ez-ref SD", title=title_sd)
    fig_mu.write_html(str(plot_dir / "ez_mu_vs_pool_size.html"), include_plotlyjs="cdn")
    fig_sd.write_html(str(plot_dir / "ez_sd_vs_pool_size.html"), include_plotlyjs="cdn")
    _box_png(by_mu, ylabel="ez-ref mean", title=title_mu, dest=plot_dir / "ez_mu_vs_pool_size.png")
    _box_png(by_sd, ylabel="ez-ref SD", title=title_sd, dest=plot_dir / "ez_sd_vs_pool_size.png")
    console.print(f"[green]plots[/green] -> {plot_dir}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", default=str(DEFAULT_INPUT_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--pool-sizes", default="20,160,10", show_default=True, help="min,max,step or comma list")
@click.option("--pool-size", default=None, type=int, help="Single pool size (SLURM array)")
@click.option("--total-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option("--fill-seed", default=DEFAULT_FILL_SEED, show_default=True, type=int)
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option("--blacklist", default=",".join(DEFAULT_BLACKLIST), show_default=True)
@click.option("--plot-only", is_flag=True, default=False)
def main(
    input_dir: str,
    output_dir: str | None,
    pool_sizes: str,
    pool_size: int | None,
    total_repeats: int,
    seed: int,
    fill_seed: int,
    n_jobs: int,
    blacklist: str,
    plot_only: bool,
) -> None:
    out = Path(output_dir) if output_dir else DEFAULT_OUT_BASE / "ref_admittance_check"
    stat_dir = out / "pool_size_ez_stats"
    plot_dir = out / "plots"
    parts = [p.strip() for p in pool_sizes.split(",") if p.strip()]
    if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
        lo, hi, step = map(int, parts)
        sizes = list(range(lo, hi + 1, step))
    else:
        sizes = sorted({int(p) for p in parts})
    if pool_size is not None:
        sizes = [int(pool_size)]

    if plot_only:
        plot_all(stat_dir, plot_dir)
        return

    bl = [s.strip() for s in blacklist.split(",") if s.strip()]
    ctx = load_universe(Path(input_dir), blacklist=bl)
    stat_dir.mkdir(parents=True, exist_ok=True)
    workers = _resolve_n_jobs(n_jobs)
    console.rule("[bold blue]pool-size ez-ref mean/SD")
    console.print(f"  pools={sizes} repeats={total_repeats} n_jobs={workers}")

    for p in sizes:
        if p < 2 or p % 2:
            raise click.ClickException(f"pool_size must be even >=2, got {p}")
        console.rule(f"[cyan]pool={p} ref {p // 2}+{p // 2}")
        pack = run_one_pool(
            ctx=ctx,
            pool_size=p,
            total_repeats=total_repeats,
            seed=seed,
            fill_seed=fill_seed,
            n_jobs=workers,
        )
        pdir = stat_dir / f"pool_{p}"
        pdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pdir / "ez_stats.npz",
            ez_mu=pack["ez_mu"],
            ez_sd=pack["ez_sd"],
            pool_size=np.int32(p),
        )
        cfg = {
            "pool_size": p,
            "ref_n": p // 2,
            "total_repeats": total_repeats,
            "seed": seed,
            "fill_seed": fill_seed,
            "n_candidate": pack["n_candidate"],
            "n_fillers": pack["n_fillers"],
            "filler_samples": pack["filler_samples"],
        }
        (pdir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        console.print(
            f"[green]OK[/green] pool={p} mu_mean={pack['ez_mu'].mean():.4f} "
            f"sd_mean={pack['ez_sd'].mean():.4f} -> {pdir / 'ez_stats.npz'}"
        )

    have = list(stat_dir.glob("pool_*/ez_stats.npz"))
    expected = list(range(20, 161, 10))
    if len(have) >= len(expected):
        plot_all(stat_dir, plot_dir)


if __name__ == "__main__":
    main()
