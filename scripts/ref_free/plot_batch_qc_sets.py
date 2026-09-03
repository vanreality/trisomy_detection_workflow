#!/usr/bin/env python3
"""Plot batch-QC Set A–D scatters (HTML) and Set D ezscore boxplots (PDF)."""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
if str(REF40_DIR) not in sys.path:
    sys.path.insert(0, str(REF40_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_search_ref40 import (  # noqa: E402
    CHR_LIST,
    compute_episcore,
    compute_zscore,
)
from ref_free_ezscore import (  # noqa: E402
    _compute_ezscore,
    _generate_half_partitions,
    _load_fixed_combo_arrays,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()

EZ_CUTOFFS = [round(3.0 + 0.1 * i, 1) for i in range(16)]

COLOR_GRAY = "#7f7f7f"
COLOR_RED = "#d62728"


def trisomy_chrs(label: str) -> set[str]:
    """Map label like ``T16`` / ``T21,T22`` to ``{chr16}`` / ``{chr21,chr22}``."""
    out: set[str] = set()
    for m in re.finditer(r"T(\d+)", str(label)):
        out.add(f"chr{int(m.group(1))}")
    return out


def point_color(label: str, chrom: str) -> str:
    """Normal/other → gray; T* → red only on the aneuploid chromosome(s)."""
    targets = trisomy_chrs(label)
    if targets and chrom in targets:
        return COLOR_RED
    return COLOR_GRAY


def load_set_table(cohort_dir: Path, set_name: str) -> pd.DataFrame:
    path = cohort_dir / f"set_{set_name}.csv"
    df = pd.read_csv(path)
    df["unit_id"] = df["unit_id"].astype(str)
    df["ff_before_mq"] = pd.to_numeric(df["ff_before_mq"], errors="coerce")
    df["purity"] = pd.to_numeric(df.get("purity"), errors="coerce")
    return df


def merge_stats(set_df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    s = stats.copy()
    s["sample"] = s["sample"].astype(str)
    return set_df.merge(s, left_on="unit_id", right_on="sample", how="inner", suffixes=("", "_stat"))


def _ff_split(df: pd.DataFrame, high: bool) -> pd.DataFrame:
    ff = pd.to_numeric(df["ff_before_mq"], errors="coerce")
    if high:
        return df[ff.fillna(-1) >= 0.01].copy()
    return df[ff.fillna(1) < 0.01].copy()


def make_ez_scatter_html(
    df: pd.DataFrame,
    title: str,
    out_html: Path,
    set_name: str,
) -> None:
    """Single-plot scatter: x=chr with samples ordered by FF within each chr.

    Slider over ez cutoffs (visibility toggle — works with plotly CDN).
    Color: Normal gray; T* red only on the aneuploid chr.
    """
    if df.empty:
        console.print(f"[yellow]skip empty scatter[/yellow] {title}")
        return

    value_cols = [c for c in df.columns if c.startswith("ez_signal_ratio_")]
    if not value_cols:
        console.print(f"[yellow]no ez_signal_ratio_* cols[/yellow] {title}")
        return

    id_vars = [
        c
        for c in [
            "unit_id",
            "sample",
            "batch_key",
            "label",
            "ff_before_mq",
            "purity",
            "chr",
            "n_batches",
            "preferred_batch_key",
            "is_preferred_batch",
            "set",
        ]
        if c in df.columns
    ]
    long = df.melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="ez_col",
        value_name="ez_signal_ratio",
    )
    long["ez_cutoff"] = (
        long["ez_col"].str.replace("ez_signal_ratio_", "", regex=False).astype(float)
    )
    long["chr"] = pd.Categorical(long["chr"], categories=CHR_LIST, ordered=True)
    long = long.dropna(subset=["chr", "ez_signal_ratio"]).copy()
    if long.empty:
        console.print(f"[yellow]skip empty after melt[/yellow] {title}")
        return

    # Within each (cutoff, chr), order samples by FF ascending.
    long["ff_sort"] = long["ff_before_mq"].fillna(-1.0)
    long = long.sort_values(["ez_cutoff", "chr", "ff_sort", "unit_id"])
    long["ff_rank"] = long.groupby(["ez_cutoff", "chr"], observed=True).cumcount()
    max_n = int(long.groupby(["ez_cutoff", "chr"], observed=True)["ff_rank"].max().max()) + 1
    gap = max(3, max_n // 10)
    chr_code = long["chr"].cat.codes.astype(int)
    long["x"] = chr_code * (max_n + gap) + long["ff_rank"]
    long["color"] = [
        point_color(lab, str(chrom)) for lab, chrom in zip(long["label"], long["chr"])
    ]

    tickvals = [i * (max_n + gap) + (max_n - 1) / 2.0 for i in range(len(CHR_LIST))]
    ticktext = CHR_LIST

    def _trace(
        sub: pd.DataFrame,
        color: str,
        name: str,
        size: float,
        visible: bool,
        showlegend: bool,
    ) -> go.Scatter:
        m = sub["color"] == color
        xs = sub.loc[m, "x"].to_numpy(dtype=float)
        ys = sub.loc[m, "ez_signal_ratio"].to_numpy(dtype=float)
        if m.any():
            custom = np.stack(
                [
                    sub.loc[m, "unit_id"].astype(str),
                    sub.loc[m, "label"].astype(str),
                    pd.to_numeric(sub.loc[m, "ff_before_mq"], errors="coerce").to_numpy(),
                    sub.loc[m, "chr"].astype(str),
                ],
                axis=-1,
            )
        else:
            custom = np.empty((0, 4))
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                color=color,
                size=size,
                opacity=0.7 if color == COLOR_GRAY else 0.9,
            ),
            name=name,
            legendgroup=name,
            customdata=custom,
            hovertemplate=(
                "unit=%{customdata[0]}<br>label=%{customdata[1]}"
                "<br>ff=%{customdata[2]:.4f}<br>chr=%{customdata[3]}"
                "<br>ratio=%{y:.3f}<extra></extra>"
            ),
            visible=visible,
            showlegend=showlegend,
        )

    cutoffs = sorted(long["ez_cutoff"].unique())
    traces: list[go.Scatter] = []
    # 2 traces per cutoff (gray, red); only first cutoff visible.
    for i, cut in enumerate(cutoffs):
        sub = long[long["ez_cutoff"] == cut]
        visible = i == 0
        traces.append(
            _trace(sub, COLOR_GRAY, "Normal / off-target", 6, visible, showlegend=visible)
        )
        traces.append(
            _trace(sub, COLOR_RED, "T* on target chr", 7, visible, showlegend=visible)
        )

    steps = []
    n_cut = len(cutoffs)
    for i, cut in enumerate(cutoffs):
        vis = []
        for j in range(n_cut):
            on = j == i
            vis.extend([on, on])
        steps.append(
            {
                "label": f"{cut:g}",
                "method": "update",
                "args": [
                    {
                        "visible": vis,
                        "showlegend": [
                            (j == i and k == 0) or (j == i and k == 1)
                            for j in range(n_cut)
                            for k in range(2)
                        ],
                    },
                    {"title": f"{title} (ez cutoff={cut:g})"},
                ],
            }
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} (ez cutoff={cutoffs[0]:g})",
        height=650,
        width=1600,
        yaxis=dict(title="ez signal ratio", range=[-0.05, 1.05], fixedrange=False),
        xaxis=dict(
            title="chromosome (samples ordered by ff ascending within each chr)",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[-gap, len(CHR_LIST) * (max_n + gap)],
        ),
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "ez cutoff: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 50},
                "steps": steps,
            }
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.25),
        template="plotly_white",
    )
    if set_name == "D":
        fig.add_annotation(
            text="Set D: each point is a sample×batch unit; within chr ordered by ff",
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.16,
            showarrow=False,
            font=dict(size=11),
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    # CDN is fine: slider uses visibility updates, not animation frames
    # (plotly 6 drops frames when include_plotlyjs='cdn').
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    console.print(f"[green]OK[/green] {out_html} points={len(long) // max(len(cutoffs), 1)}")


def _load_ez_repeat_matrix(
    input_dir: Path,
    result_dir: Path,
    unit_ids: list[str],
) -> tuple[np.ndarray, list[str], dict]:
    """Return ez scores shaped ``[n_repeats, n_chr, n_units]`` for requested units."""
    ref_dir = Path(result_dir) / "ref_free_ezscore"
    cfg = json.loads((ref_dir / "run_config.json").read_text())
    if cfg.get("combo_mode") != "fixed":
        raise click.ClickException("Only combo_mode=fixed supported for Set D boxplots")

    total_repeats = int(cfg["total_repeats"])
    ref_n = int(cfg["ref_n"])
    seed = int(cfg["seed"])
    ep_threshold = float(cfg["ep_threshold"])
    ep_recall = float(cfg["ep_recall"])
    z_threshold = float(cfg["z_threshold"])
    z_recall = float(cfg["z_recall"])

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
    is_dev_normal = (set_arr == "dev") & (label_arr == "Normal")
    ref_pool_idx = np.flatnonzero(is_dev_normal)
    if ref_pool_idx.size < 2 * ref_n:
        raise click.ClickException(
            f"Need >= {2 * ref_n} dev Normal, found {ref_pool_idx.size}"
        )

    keep = [u for u in unit_ids if u in sample_index]
    if not keep:
        raise click.ClickException("None of the requested unit_ids are in input universe")
    eval_idx = np.asarray([sample_index[u] for u in keep], dtype=np.int64)
    n_eval = eval_idx.size
    n_chr = len(CHR_LIST)

    ep_arrays, z_array = _load_fixed_combo_arrays(
        ep_df,
        z_df,
        ep_threshold,
        ep_recall,
        z_threshold,
        z_recall,
        sample_index,
        chr_index,
    )
    rng = np.random.default_rng(seed)
    ref_local_draws, ez_local_draws = _generate_half_partitions(
        pool_size=ref_pool_idx.size,
        half=ref_n,
        n_repeats=total_repeats,
        rng=rng,
    )

    ez_mat = np.empty((total_repeats, n_chr, n_eval), dtype=np.float64)
    console.print(
        f"SetD ez repeats: units={n_eval}/{len(unit_ids)} repeats={total_repeats}"
    )
    for r in range(total_repeats):
        ref_idx = ref_pool_idx[ref_local_draws[r]]
        ez_ref_idx = ref_pool_idx[ez_local_draws[r]]
        episcore = compute_episcore(
            np.expand_dims(ep_arrays[0], 0),
            np.expand_dims(ep_arrays[1], 0),
            np.expand_dims(ep_arrays[2], 0),
            np.expand_dims(ep_arrays[3], 0),
            ref_idx,
        )[0]
        zscore = compute_zscore(np.expand_dims(z_array, 0), ref_idx)[0]
        ez = _compute_ezscore(episcore, zscore, ez_ref_idx)
        ez_mat[r] = ez[:, eval_idx]
        if (r + 1) % 1000 == 0:
            console.print(f"  {r + 1}/{total_repeats}")
    return ez_mat, keep, cfg


def make_setD_ez_boxplots(
    set_d: pd.DataFrame,
    stats: pd.DataFrame,
    input_dir: Path,
    result_dir: Path,
    out_dir: Path,
    mode_name: str,
    signal_ratio_cutoff: float = 3.0,
) -> None:
    """One PDF per Set D unit: boxplot of ezscore across repeats per chromosome."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = set_d["unit_id"].astype(str).tolist()
    try:
        ez_mat, keep, _cfg = _load_ez_repeat_matrix(input_dir, result_dir, units)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]SetD boxplot load failed[/red] {mode_name}: {exc}")
        return

    meta = set_d.set_index("unit_id")
    ratio_col = f"ez_signal_ratio_{signal_ratio_cutoff:g}"
    stats_u = stats.copy()
    stats_u["sample"] = stats_u["sample"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_index = {u: i for i, u in enumerate(keep)}
    for uid in keep:
        row = meta.loc[uid]
        sample = str(row["sample"])
        batch = str(row.get("batch_key", ""))
        ff = row.get("ff_before_mq", np.nan)
        label = str(row.get("label", ""))
        ji = keep_index[uid]
        # [n_repeats, n_chr]
        data = [ez_mat[:, hi, ji] for hi in range(len(CHR_LIST))]

        fig, ax = plt.subplots(figsize=(16, 5))
        bp = ax.boxplot(
            data,
            positions=np.arange(1, len(CHR_LIST) + 1),
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#aec7e8")
            patch.set_alpha(0.8)

        # Signal-ratio annotations from precomputed stats
        sub_stats = stats_u[stats_u["sample"] == uid]
        ratio_by_chr = {
            str(r["chr"]): float(r[ratio_col])
            for _, r in sub_stats.iterrows()
            if ratio_col in sub_stats.columns and pd.notna(r.get(ratio_col))
        }
        ymax = max(np.nanmax(ez_mat[:, :, ji]), 1.0)
        for i, chrom in enumerate(CHR_LIST, start=1):
            ratio = ratio_by_chr.get(chrom, np.nan)
            if np.isfinite(ratio):
                y_box = np.nanpercentile(data[i - 1], 95)
                ax.text(
                    i,
                    y_box + 0.03 * (ymax - np.nanmin(ez_mat[:, :, ji]) + 1e-6),
                    f"{ratio:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        ff_txt = f"{float(ff):.4f}" if pd.notna(ff) else "NA"
        ax.set_title(
            f"{mode_name} SetD  sample={sample}  batch={batch}  "
            f"ff_before_mq={ff_txt}  label={label}"
        )
        ax.set_xticks(np.arange(1, len(CHR_LIST) + 1))
        ax.set_xticklabels(CHR_LIST, rotation=45)
        ax.set_xlabel("chromosome")
        ax.set_ylabel("ezscore (repeats)")
        ax.axhline(3.0, linestyle=":", color="black", linewidth=0.8)
        ax.axhline(4.5, linestyle=":", color="black", linewidth=0.8)
        fig.tight_layout()
        safe = uid.replace("/", "_")
        out_pdf = out_dir / f"SetD_{safe}_ez_boxplot.pdf"
        fig.savefig(out_pdf, dpi=140)
        plt.close(fig)
        console.print(f"[green]OK[/green] {out_pdf}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--cohort-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--stats-tsv", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--mode-name", required=True, type=str)
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--sets", default="A,B,C,D", show_default=True)
@click.option(
    "--input-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Mode input_fixed dir (required for Set D boxplots).",
)
@click.option(
    "--result-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Mode fixed_combo dir containing ref_free_ezscore/ (Set D boxplots).",
)
@click.option("--skip-boxplot", is_flag=True, default=False)
def main(
    cohort_dir: str,
    stats_tsv: str,
    mode_name: str,
    output_dir: str,
    sets: str,
    input_dir: str | None,
    result_dir: str | None,
    skip_boxplot: bool,
) -> None:
    cohort = Path(cohort_dir)
    stats = pd.read_csv(stats_tsv, sep="\t")
    out_root = Path(output_dir) / mode_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Drop legacy violin PDFs if present
    for p in out_root.glob("*_violin.pdf"):
        p.unlink()
        console.print(f"removed {p.name}")

    for set_name in [s.strip() for s in sets.split(",") if s.strip()]:
        set_df = load_set_table(cohort, set_name)
        merged = merge_stats(set_df, stats)
        console.print(f"{mode_name} Set{set_name}: cohort={len(set_df)} merged_rows={len(merged)}")
        for high, tag in [(True, "ff_ge_1pct"), (False, "ff_lt_1pct")]:
            sub = _ff_split(merged, high=high)
            make_ez_scatter_html(
                sub,
                title=f"{mode_name} Set{set_name} {tag}: ez abnormal signal ratio",
                out_html=out_root / f"Set{set_name}_{tag}_ez_scatter.html",
                set_name=set_name,
            )

        if set_name == "D" and not skip_boxplot:
            if not input_dir or not result_dir:
                console.print("[yellow]skip SetD boxplots: need --input-dir and --result-dir[/yellow]")
                continue
            make_setD_ez_boxplots(
                set_d=set_df,
                stats=stats,
                input_dir=Path(input_dir),
                result_dir=Path(result_dir),
                out_dir=out_root / "SetD_ez_boxplots",
                mode_name=mode_name,
            )


if __name__ == "__main__":
    main()
