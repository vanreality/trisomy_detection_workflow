#!/usr/bin/env python3
"""Plot fixed-mode pool_size → MCC on the core eval set.

Classification rule: ``ezscore_signal_ratio >= ratio_cutoff`` → trisomy, else
Normal. MCC is computed only on the intersection eval cohort (samples still
evaluable at the largest pool size), after ff≥ff_min and optional blacklist.
"""

from __future__ import annotations

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


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return float("nan")
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def _pool_dirs(mode_dir: Path) -> list[Path]:
    return sorted(
        (p for p in mode_dir.glob("pool_*") if p.is_dir() and p.name.split("_")[1].isdigit()),
        key=lambda p: int(p.name.split("_")[1]),
    )


def _core_eval_table(
    mode_dir: Path,
    *,
    ff_min: float,
    ratio_cutoff: float,
    blacklist: set[str],
    score_col: str = "ezscore_signal_ratio",
) -> pd.DataFrame:
    pool_dirs = _pool_dirs(mode_dir)
    if not pool_dirs:
        raise click.ClickException(f"No pool_* dirs under {mode_dir}")
    largest = pool_dirs[-1]
    core = pd.read_csv(largest / "abnormality_signal_ratio.tsv", sep="\t")
    core["sample"] = core["sample"].astype(str)
    core["ff_before_mq"] = pd.to_numeric(core["ff_before_mq"], errors="coerce")
    core = core[
        ~core["sample"].isin(blacklist)
        & (core["ff_before_mq"] >= ff_min)
        & (
            core["label"].astype(str).eq("Normal")
            | core["label"].map(is_trisomy_label)
        )
    ]
    core_samples = set(core["sample"])
    y_core = core.set_index("sample")["label"].map(is_trisomy_label)

    rows = []
    for pdir in pool_dirs:
        tsv = pdir / "abnormality_signal_ratio.tsv"
        if not tsv.is_file():
            continue
        df = pd.read_csv(tsv, sep="\t")
        df["sample"] = df["sample"].astype(str)
        sub = df[df["sample"].isin(core_samples)].set_index("sample")
        # align to core sample order
        scores = pd.to_numeric(sub.reindex(y_core.index)[score_col], errors="coerce")
        y_true = y_core.to_numpy()
        y_pred = (scores.to_numpy(dtype=float) >= ratio_cutoff)
        valid = np.isfinite(scores.to_numpy(dtype=float))
        mcc = matthews_corrcoef(y_true[valid], y_pred[valid])
        tp = int((y_true[valid] & y_pred[valid]).sum())
        tn = int((~y_true[valid] & ~y_pred[valid]).sum())
        fp = int((~y_true[valid] & y_pred[valid]).sum())
        fn = int((y_true[valid] & ~y_pred[valid]).sum())
        rows.append(
            {
                "pool_size": int(pdir.name.split("_")[1]),
                "ref_n": int(pdir.name.split("_")[1]) // 2,
                "mcc_ezscore": mcc,
                "ratio_cutoff": ratio_cutoff,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "fp_plus_fn": fp + fn,
                "n_normal_core": int((~y_true[valid]).sum()),
                "n_trisomy_core": int(y_true[valid].sum()),
                "core_pool_size": int(largest.name.split("_")[1]),
                "score_col": score_col,
            }
        )
    return pd.DataFrame(rows).sort_values("pool_size")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--sweep-base", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", default=None, type=click.Path(path_type=Path))
@click.option("--ff-min", default=0.01, show_default=True, type=float)
@click.option("--ratio-cutoff", default=0.5, show_default=True, type=float)
@click.option(
    "--blacklist",
    default=",".join(DEFAULT_BLACKLIST),
    show_default=True,
)
def main(
    sweep_base: Path,
    output_dir: Path | None,
    ff_min: float,
    ratio_cutoff: float,
    blacklist: str,
) -> None:
    out = output_dir or (sweep_base / "plots")
    out.mkdir(parents=True, exist_ok=True)
    bl = {s.strip() for s in blacklist.split(",") if s.strip()}

    mode_dir = sweep_base / "fixed"
    df = _core_eval_table(
        mode_dir, ff_min=ff_min, ratio_cutoff=ratio_cutoff, blacklist=bl
    )
    if df.empty:
        raise click.ClickException("No MCC rows computed")

    tsv = out / "pool_size_mcc_fixed.tsv"
    df.to_csv(tsv, sep="\t", index=False, float_format="%.6f")
    # also mirror under fixed/
    df.to_csv(mode_dir / "pool_size_mcc.tsv", sep="\t", index=False, float_format="%.6f")

    fig = go.Figure(
        go.Scatter(
            x=df["pool_size"],
            y=df["mcc_ezscore"],
            mode="lines+markers",
            name="ezscore MCC",
            line=dict(color="rgb(214,39,40)", width=2),
            marker=dict(size=7),
            customdata=np.stack(
                [df["fp"], df["fn"], df["fp_plus_fn"], df["tp"], df["tn"]], axis=1
            ),
            hovertemplate=(
                "pool=%{x}<br>MCC=%{y:.4f}<br>"
                "FP=%{customdata[0]} FN=%{customdata[1]} "
                "FP+FN=%{customdata[2]}<br>TP=%{customdata[3]} TN=%{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )
    n_n = int(df["n_normal_core"].iloc[0])
    n_t = int(df["n_trisomy_core"].iloc[0])
    fig.update_layout(
        title=(
            f"Fixed-combo pool size vs MCC "
            f"(signal_ratio≥{ratio_cutoff:g} → trisomy)"
            f"<br><sup>core eval only · N={n_n} T={n_t} · ff≥{ff_min*100:.0f}% · "
            f"blacklist n={len(bl)}</sup>"
        ),
        xaxis_title="pool_size (= 2 × ref_n)",
        yaxis_title="MCC",
        template="plotly_white",
        height=520,
        width=920,
        margin=dict(t=90, b=60),
    )
    fig.update_yaxes(range=[-0.05, 1.05])
    html = out / "pool_size_mcc.html"
    fig.write_html(str(html), include_plotlyjs="cdn", full_html=True)
    console.print(
        f"[green]OK[/green] {html}  "
        f"MCC range [{df['mcc_ezscore'].min():.4f}, {df['mcc_ezscore'].max():.4f}] "
        f"best pool={int(df.loc[df['mcc_ezscore'].idxmax(), 'pool_size'])}"
    )


if __name__ == "__main__":
    main()
