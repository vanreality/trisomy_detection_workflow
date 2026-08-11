#!/usr/bin/env python3
"""FP+FN density across 40+40 fixed-combo reference draws.

Loads per-repeat abnormality flags from ``ref_free_fixed_flags`` shards, and for
each repeat counts:

    FP = Normal called abnormal
    FN = Trisomy called normal

among ff≥ff_min eval samples. Emits a density table + bar plot of repeats per
``FP+FN`` value — useful to see how often a bad reference bipartition tanks
disomy/trisomy classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.console import Console

from separation import is_trisomy_label

console = Console()

DEFAULT_BLACKLIST = (
    "PTAY0577P9S1",
    "PTAY0599P8S1",
    "PTAY0666P7S1",
    "PTAY0682P7S1",
    "PTAY0689P8H1",
)


def _load_flags(flag_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    def _start(p: Path) -> int:
        return int(p.stem.split("_")[1])

    shards = sorted(flag_dir.glob("flags_*.npz"), key=_start)
    if not shards:
        raise click.ClickException(f"No flags_*.npz under {flag_dir}")
    ep, z, ez = [], [], []
    for p in shards:
        d = np.load(p)
        ep.append(d["flags_ep"])
        z.append(d["flags_z"])
        ez.append(d["flags_ez"])
    cfg = {}
    cfg_path = flag_dir / "run_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
    return (
        np.concatenate(ep, axis=0),
        np.concatenate(z, axis=0),
        np.concatenate(ez, axis=0),
        cfg,
    )


def _fp_fn_per_repeat(flags: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """flags (n_rep, n_keep), y True=trisomy."""
    pred = flags.astype(bool)
    yb = y.astype(bool)
    fp = (pred & ~yb).sum(axis=1).astype(np.int32)
    fn = (~pred & yb).sum(axis=1).astype(np.int32)
    return fp, fn, fp + fn


def _density_table(values: np.ndarray) -> pd.DataFrame:
    vals, counts = np.unique(values, return_counts=True)
    n = int(values.size)
    return pd.DataFrame(
        {
            "fp_plus_fn": vals.astype(int),
            "n_repeats": counts.astype(int),
            "density": counts.astype(float) / float(n),
            "cum_density": np.cumsum(counts.astype(float) / float(n)),
        }
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--flag-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="…/fixed_flags with flags_*.npz + eval_samples.tsv",
)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path))
@click.option("--ff-min", default=0.01, show_default=True, type=float)
@click.option(
    "--score",
    default="ezscore",
    type=click.Choice(["ezscore", "episcore", "zscore", "all"]),
    show_default=True,
)
@click.option("--max-repeats", default=None, type=int, help="Optional cap for quick checks")
@click.option(
    "--blacklist",
    default=",".join(DEFAULT_BLACKLIST),
    show_default=True,
    help="Comma-separated samples excluded from FP/FN counts",
)
def main(
    flag_dir: Path,
    output_dir: Path | None,
    ff_min: float,
    score: str,
    max_repeats: int | None,
    blacklist: str,
) -> None:
    out = output_dir or (flag_dir.parent / "fp_fn_density")
    out.mkdir(parents=True, exist_ok=True)

    eval_info = pd.read_csv(flag_dir / "eval_samples.tsv", sep="\t")
    flags_ep, flags_z, flags_ez, cfg = _load_flags(flag_dir)
    if max_repeats is not None:
        flags_ep = flags_ep[:max_repeats]
        flags_z = flags_z[:max_repeats]
        flags_ez = flags_ez[:max_repeats]

    bl = {s.strip() for s in blacklist.split(",") if s.strip()}
    ff = pd.to_numeric(eval_info["ff_before_mq"], errors="coerce").to_numpy()
    labels = eval_info["label"].astype(str)
    samples = eval_info["sample"].astype(str)
    y_all = labels.map(is_trisomy_label).to_numpy()
    keep = (
        (ff >= ff_min)
        & (labels.eq("Normal").to_numpy() | y_all)
        & ~samples.isin(bl).to_numpy()
    )
    y = y_all[keep]
    n_bl_dropped = int(samples.isin(bl).sum())
    console.print(
        f"Loaded n_rep={flags_ez.shape[0]} n_eval={len(eval_info)} "
        f"blacklist dropped={n_bl_dropped} "
        f"ff≥{ff_min}: N={int((keep & ~y_all).sum())} T={int((keep & y_all).sum())}"
    )

    score_map = {
        "episcore": flags_ep[:, keep],
        "zscore": flags_z[:, keep],
        "ezscore": flags_ez[:, keep],
    }
    names = list(score_map) if score == "all" else [score]

    summary_rows = []
    fig = go.Figure()
    colors = {
        "episcore": "rgb(31,119,180)",
        "zscore": "rgb(44,160,44)",
        "ezscore": "rgb(214,39,40)",
    }
    for name in names:
        fp, fn, tot = _fp_fn_per_repeat(score_map[name], y)
        dens = _density_table(tot)
        dens.to_csv(out / f"fp_fn_density_{name}.tsv", sep="\t", index=False, float_format="%.6f")
        per_rep = pd.DataFrame({"fp": fp, "fn": fn, "fp_plus_fn": tot})
        # store compact histogram only by default; optional sample of worst reps
        worst = per_rep.nlargest(20, "fp_plus_fn")
        worst.to_csv(out / f"worst_repeats_{name}.tsv", sep="\t", index=False)

        fig.add_trace(
            go.Bar(
                x=dens["fp_plus_fn"],
                y=dens["density"],
                name=name,
                marker_color=colors[name],
                opacity=0.85 if len(names) == 1 else 0.65,
                hovertemplate="FP+FN=%{x}<br>density=%{y:.4f}<br>n=%{customdata}<extra>"
                + name
                + "</extra>",
                customdata=dens["n_repeats"],
            )
        )
        summary_rows.append(
            {
                "score": name,
                "n_repeats": int(tot.size),
                "mean_fp": float(fp.mean()),
                "mean_fn": float(fn.mean()),
                "mean_fp_plus_fn": float(tot.mean()),
                "median_fp_plus_fn": float(np.median(tot)),
                "p95_fp_plus_fn": float(np.quantile(tot, 0.95)),
                "p99_fp_plus_fn": float(np.quantile(tot, 0.99)),
                "max_fp_plus_fn": int(tot.max()),
                "frac_fp_plus_fn_ge_5": float((tot >= 5).mean()),
                "frac_fp_plus_fn_ge_10": float((tot >= 10).mean()),
                "frac_perfect": float((tot == 0).mean()),
                "ez_cutoff": cfg.get("ez_cutoff"),
                "ref_n": cfg.get("ref_n"),
            }
        )
        console.print(
            f"  {name}: mean FP+FN={tot.mean():.3f} max={tot.max()} "
            f"perfect={(tot == 0).mean():.4f} ≥10={(tot >= 10).mean():.4f}"
        )

    fig.update_layout(
        title=(
            "40+40 fixed-combo: repeat density of FP+FN<br>"
            f"<sup>ff≥{ff_min*100:.0f}% · ez_cutoff={cfg.get('ez_cutoff')} · "
            f"n_rep={flags_ez.shape[0]:,} · N={int((~y).sum())} T={int(y.sum())}</sup>"
        ),
        xaxis_title="FP + FN (per repeat)",
        yaxis_title="repeat density",
        barmode="group",
        template="plotly_white",
        height=520,
        width=900,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=90, b=80),
    )
    html = out / "fp_fn_density.html"
    fig.write_html(str(html), include_plotlyjs="cdn", full_html=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "fp_fn_summary.tsv", sep="\t", index=False, float_format="%.6f")
    (out / "fp_fn_summary.json").write_text(
        json.dumps(
            {
                "flag_dir": str(flag_dir),
                "ff_min": ff_min,
                "blacklist": sorted(bl),
                "n_blacklist_in_eval": n_bl_dropped,
                "n_repeats": int(flags_ez.shape[0]),
                "n_normal": int((~y).sum()),
                "n_trisomy": int(y.sum()),
                "run_config": cfg,
                "scores": summary_rows,
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"[green]OK[/green] Wrote {html}")


if __name__ == "__main__":
    main()
