#!/usr/bin/env python3
"""Interactive chr22 ezscore density vs pool size for PTAY0599P8S1 and HCPT0008."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_ez3_vs_ez45_discordant import (  # noqa: E402
    DEFAULT_INPUT,
    DESIGNS,
    FILL_SEED,
    FOCUS,
    SEED,
    SWEEP_BASE,
    _build_candidate,
    load_score_ctx,
)
from grid_search_ref40 import CHR_LIST, compute_episcore, compute_zscore  # noqa: E402
from ref_free_ezscore import _generate_half_partitions  # noqa: E402

console = Console()

CHR22 = "chr22"
CHR22_I = CHR_LIST.index(CHR22)
X_GRID = np.linspace(-1.0, 12.0, 481)
X_LO, X_HI = -0.5, 10.0
EZ3, EZ45 = 3.0, 4.5
PANEL_META = {
    "PTAY0599P8S1": dict(color="#5C4B8A", subtitle="T22 · FF 5.2% · blacklisted"),
    "HCPT0008": dict(color="#0A7A7A", subtitle="T22 · FF 2.0%"),
}


def _kde(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.zeros_like(x)
    sd = float(np.std(v, ddof=1))
    if sd <= 1e-8:
        y = np.zeros_like(x)
        y[np.argmin(np.abs(x - float(np.mean(v))))] = 1.0
        return y
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(v)
        y = kde(x)
    except Exception:
        bw = 1.06 * sd * (v.size ** -0.2)
        bw = max(bw, 0.05)
        z = (x[None, :] - v[:, None]) / bw
        y = np.exp(-0.5 * z * z).mean(axis=0) / (bw * np.sqrt(2 * np.pi))
    y = np.where(np.isfinite(y), y, 0.0)
    return y


def simulate_chr22(
    ctx: dict,
    *,
    design: str,
    pool_sizes: list[int],
    n_repeats: int,
    samples: tuple[str, ...] = FOCUS,
    seed: int = SEED,
    fill_seed: int = FILL_SEED,
) -> dict[str, np.ndarray]:
    """Return {sample: array shape (n_pools, n_repeats)} of chr22 ezscore."""
    cfg = DESIGNS[design]
    idx = np.array([ctx["sample_index"][s] for s in samples], dtype=np.int64)
    out = {s: np.empty((len(pool_sizes), n_repeats), dtype=np.float32) for s in samples}
    for pi, pool_size in enumerate(pool_sizes):
        half = pool_size // 2
        cand_n = int(cfg["fixed_candidate_size"] or pool_size)
        candidate = _build_candidate(ctx["set_arr"], ctx["label_arr"], cand_n, fill_seed)
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
            combined = episcore[CHR22_I] + zscore[CHR22_I]
            ref_vals = combined[ez_ref_idx]
            with np.errstate(invalid="ignore"):
                mu = float(np.nanmean(ref_vals))
                sd = float(np.nanstd(ref_vals, ddof=0))
            if not np.isfinite(mu):
                mu = 0.0
            sd_safe = sd if sd > 0 else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                ez = (combined[idx] - mu) / sd_safe
            for j, s in enumerate(samples):
                out[s][pi, r] = ez[j]
    return out


def _vline(x0: float, ymax: float, color: str, name: str, showlegend: bool, xaxis: str, yaxis: str) -> go.Scatter:
    return go.Scatter(
        x=[x0, x0],
        y=[0, ymax],
        mode="lines",
        line=dict(color=color, width=1.6, dash="dot"),
        name=name,
        showlegend=showlegend,
        hovertemplate=f"{name}<extra></extra>",
        legendgroup=name,
        xaxis=xaxis,
        yaxis=yaxis,
    )


def _panel_traces(
    sample: str,
    density: np.ndarray,
    ymax: float,
    showlegend: bool,
    xaxis: str,
    yaxis: str,
) -> list:
    color = PANEL_META[sample]["color"]
    fill = "rgba(92,75,138,0.28)" if sample == "PTAY0599P8S1" else "rgba(10,122,122,0.28)"
    kde = go.Scatter(
        x=X_GRID,
        y=density,
        mode="lines",
        line=dict(color=color, width=2.2),
        fill="tozeroy",
        fillcolor=fill,
        name=sample,
        showlegend=showlegend,
        hovertemplate="chr22 ez=%{x:.2f}<br>density=%{y:.3f}<extra>" + sample + "</extra>",
        xaxis=xaxis,
        yaxis=yaxis,
    )
    return [
        kde,
        _vline(EZ3, ymax, "#2E6F9E", "cutoff 3", showlegend, xaxis, yaxis),
        _vline(EZ45, ymax, "#C1121F", "cutoff 4.5", showlegend, xaxis, yaxis),
    ]


def _stats(arr: np.ndarray) -> dict:
    v = arr[np.isfinite(arr)]
    return {
        "n": int(v.size),
        "median": float(np.median(v)) if v.size else float("nan"),
        "mean": float(np.mean(v)) if v.size else float("nan"),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        "p3": float((v > EZ3).mean()) if v.size else float("nan"),
        "p45": float((v > EZ45).mean()) if v.size else float("nan"),
    }


def _title(design: str, pool: int, stats: dict[str, dict]) -> str:
    bits = []
    for s in FOCUS:
        st = stats[s]
        bits.append(
            f"{s}: med={st['median']:.2f}  P(&gt;3)={st['p3']:.2f}  P(&gt;4.5)={st['p45']:.2f}"
        )
    label = DESIGNS[design]["label"]
    return (
        f"chr22 ezscore density · {label}<br>"
        f"<sup>pool={pool} (ref {pool // 2}+{pool // 2}) · "
        + " · ".join(bits)
        + "</sup>"
    )


def _play_menu() -> dict:
    return dict(
        type="buttons",
        showactive=False,
        direction="left",
        x=1.0,
        y=1.18,
        xanchor="right",
        yanchor="top",
        pad=dict(l=0, t=0),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=12),
        buttons=[
            dict(
                label="▶ Play",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=220, redraw=True),
                        fromcurrent=True,
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
            ),
            dict(
                label="❚❚ Pause",
                method="animate",
                args=[
                    [None],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=False),
                        transition=dict(duration=0),
                    ),
                ],
            ),
        ],
    )


def build_figure(
    design: str,
    pool_sizes: list[int],
    arrays: dict[str, np.ndarray],
    default_pool: int = 80,
) -> go.Figure:
    dens = {
        s: np.stack([_kde(arrays[s][i], X_GRID) for i in range(len(pool_sizes))])
        for s in FOCUS
    }
    ymax = 1.08 * max(float(d.max()) for d in dens.values())
    ymax = max(ymax, 0.4)
    stats_by_pool = []
    for i, _p in enumerate(pool_sizes):
        stats_by_pool.append({s: _stats(arrays[s][i]) for s in FOCUS})

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{s}<br><sup>{PANEL_META[s]['subtitle']}</sup>" for s in FOCUS
        ],
        horizontal_spacing=0.07,
    )
    AXES = (("x", "y"), ("x2", "y2"))
    default_i = pool_sizes.index(default_pool) if default_pool in pool_sizes else 0
    for col, sample in enumerate(FOCUS, start=1):
        xa, ya = AXES[col - 1]
        traces = _panel_traces(
            sample, dens[sample][default_i], ymax, showlegend=(col == 1), xaxis=xa, yaxis=ya
        )
        for tr in traces:
            fig.add_trace(tr, row=1, col=col)

    def _slider(active: int) -> dict:
        return dict(
            active=active,
            currentvalue=dict(prefix="pool size: ", font=dict(size=14)),
            pad=dict(t=28, b=8),
            x=0.06,
            len=0.72,
            ticklen=3,
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(p)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=str(p),
                )
                for p in pool_sizes
            ],
        )

    def _frame_layout(i: int) -> dict:
        return dict(
            title=dict(text=_title(design, pool_sizes[i], stats_by_pool[i]), x=0.02, xanchor="left"),
            updatemenus=[_play_menu()],
            sliders=[_slider(i)],
        )

    frames = []
    for i, pool in enumerate(pool_sizes):
        data = []
        for sample, (xa, ya) in zip(FOCUS, AXES):
            data.extend(
                _panel_traces(sample, dens[sample][i], ymax, showlegend=False, xaxis=xa, yaxis=ya)
            )
        frames.append(go.Frame(name=str(pool), data=data, layout=_frame_layout(i)))
    fig.frames = frames

    fig.update_layout(
        title=dict(text=_title(design, pool_sizes[default_i], stats_by_pool[default_i]), x=0.02, xanchor="left"),
        template="plotly_white",
        height=560,
        width=1180,
        margin=dict(t=110, b=90, l=60, r=40),
        plot_bgcolor="rgba(248,249,250,1)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0.5, xanchor="center"),
        updatemenus=[_play_menu()],
        sliders=[_slider(default_i)],
        bargap=0,
    )
    fig.update_xaxes(title_text="chr22 ezscore", range=[X_LO, X_HI], showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(title_text="density", range=[0, ymax], showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(title_text="", row=1, col=2)
    return fig


def save_npz(path: Path, pool_sizes: list[int], arrays: dict[str, np.ndarray], n_repeats: int, design: str) -> None:
    payload = {"pool_sizes": np.asarray(pool_sizes, dtype=np.int32), "n_repeats": np.int32(n_repeats)}
    for s, arr in arrays.items():
        payload[s] = arr
    payload["design"] = np.asarray(design)
    np.savez_compressed(path, **payload)


def load_npz(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    pools = z["pool_sizes"].tolist()
    arrays = {s: z[s] for s in FOCUS}
    return pools, arrays


@click.command()
@click.option("--input-dir", default=str(DEFAULT_INPUT), type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--n-repeats", default=2000, show_default=True, type=int)
@click.option("--pool-step", default=2, show_default=True, type=int)
@click.option("--design", default="both", type=click.Choice(["fixed160", "growing", "both"]))
@click.option("--skip-mc", is_flag=True, default=False)
def main(
    input_dir: str,
    output_dir: str | None,
    n_repeats: int,
    pool_step: int,
    design: str,
    skip_mc: bool,
) -> None:
    out = Path(output_dir) if output_dir else SWEEP_BASE / "ez3_vs_ez45_discordant"
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    pool_sizes = list(range(20, 161, int(pool_step)))
    designs = ["fixed160", "growing"] if design == "both" else [design]

    ctx = None
    for d in designs:
        npz_path = tab_dir / f"chr22_ez_{d}.npz"
        if skip_mc and npz_path.is_file():
            pools, arrays = load_npz(npz_path)
            console.print(f"reused {npz_path}")
        else:
            if ctx is None:
                ctx = load_score_ctx(Path(input_dir))
            arrays = simulate_chr22(
                ctx, design=d, pool_sizes=pool_sizes, n_repeats=n_repeats
            )
            save_npz(npz_path, pool_sizes, arrays, n_repeats, d)
            pools = pool_sizes
            console.print(f"wrote {npz_path}")
        fig = build_figure(d, pools, arrays, default_pool=80 if 80 in pools else pools[0])
        html = fig_dir / f"chr22_ez_density_{d}.html"
        fig.write_html(str(html), include_plotlyjs="cdn", full_html=True)
        console.print(f"[green]html[/green] {html}")
    console.print(f"[bold green]done[/bold green] {fig_dir}")


if __name__ == "__main__":
    main()
