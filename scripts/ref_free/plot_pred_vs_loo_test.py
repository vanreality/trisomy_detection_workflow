#!/usr/bin/env python3
"""Compare meta final_zscores (fixed ref17+ez25) vs LOO ez on 0817 test samples.

Upper row: scatter of meta ``final_zscores``; call at ez>4.5 (production Gray band
is 3–4.5, so this matches hard T#).

Lower row: 0817 clean mode4 LOO ez from ``profiles.tsv``; call at ez>3.

Cohort: test rows in ``0817/clean/mode4_loo_fixed/profiles.tsv``, minus the silent
plot blacklist (PTAY0577P9S1) → n=175.
"""

from __future__ import annotations

import json
import sys
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
SELECT_DIR = SCRIPT_DIR.parent / "select_stable_ref40"
for _p in (SCRIPT_DIR, REF40_DIR, SELECT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grid_search_ref40 import CHR_LIST  # noqa: E402
from plot_ref40_vs_fixed_profiles import (  # noqa: E402
    _cohort_frame,
    _target_mask_from_labels,
    build_universe,
    plot_fixed_ez_scatter,
)
from plot_stable_ref40_compare import (  # noqa: E402
    _plot_metric_bars,
    detection_metrics,
)
from pool_size_ez_ref_bands import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_PARQUET,
    DEFAULT_TOXIC,
    combined_on_query,
    loo_combined,
)
from pred_label_utils import parse_comma_scores  # noqa: E402

console = Console()
# Same silent drop as run_0817.write_fixed_panel (never mentioned on figures).
BLACKLIST = frozenset({"PTAY0577P9S1"})

DEFAULT_LOO_DIR = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev"
    "/0817/clean/mode4_loo_fixed"
)
DEFAULT_OUT = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev"
    "/0817/fixed_ref_17_plus_25"
)
FIXED_CUTOFF = 4.5
LOO_CUTOFF = 3.0


def select_loo_test_idx(ctx: dict, loo_dir: Path) -> np.ndarray:
    """Test samples from the 0817/0818 mode4 profiles (minus BLACKLIST)."""
    wide = pd.read_csv(loo_dir / "profiles.tsv", sep="\t")
    wide["sample"] = wide["sample"].astype(str)
    names = [
        s
        for s, set_name in zip(wide["sample"], wide["set"].astype(str))
        if set_name == "test" and s not in BLACKLIST
    ]
    missing = [s for s in names if s not in ctx["sample_index"]]
    if missing:
        raise click.ClickException(f"LOO test samples missing from universe: {missing[:8]}")
    return np.array([ctx["sample_index"][s] for s in names], dtype=np.int64)


def loo_ez_from_saved(
    ctx: dict,
    idx: np.ndarray,
    loo_dir: Path,
) -> tuple[np.ndarray, int]:
    """Load mode4 profiles.tsv; score samples missing from that eval vs clean_ref.

    Recomputing from params.npz does **not** reproduce the saved profiles (the
    Aug 18 job's fully-fixed path disagrees with profiles.tsv). Use the saved
    matrix so the lower panel matches ``fixed.png``, and fill the 16 toxic
    test Normals that were dropped from clean_eval.
    """
    samples = [ctx["ordered"][int(i)] for i in idx]
    ez = np.full((len(CHR_LIST), len(samples)), np.nan, dtype=np.float64)
    wide = pd.read_csv(loo_dir / "profiles.tsv", sep="\t")
    wide["sample"] = wide["sample"].astype(str)
    by_s = wide.set_index("sample")
    missing_pos: list[int] = []
    missing_idx: list[int] = []
    n_saved = 0
    for j, s in enumerate(samples):
        if s in by_s.index:
            ez[:, j] = pd.to_numeric(by_s.loc[s, CHR_LIST], errors="coerce").to_numpy(
                dtype=np.float64
            )
            n_saved += 1
        else:
            missing_pos.append(j)
            missing_idx.append(int(idx[j]))
    if missing_idx:
        ref_path = loo_dir.parent.parent / "pools" / "clean_ref.tsv"
        if not ref_path.is_file():
            raise click.ClickException(f"missing clean ref for fill-in scores: {ref_path}")
        ref = pd.read_csv(ref_path, sep="\t")
        ref_idx = np.array(
            [ctx["sample_index"][s] for s in ref["sample"].astype(str)], dtype=np.int64
        )
        miss = np.array(missing_idx, dtype=np.int64)
        loo_ref = loo_combined(
            ctx["arrays"]["hypo"][:, ref_idx],
            ctx["arrays"]["hyper"][:, ref_idx],
            ctx["arrays"]["hypo_cnt"][:, ref_idx],
            ctx["arrays"]["hyper_cnt"][:, ref_idx],
            ctx["arrays"]["pct"][:, ref_idx],
        )
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(loo_ref, axis=1)
            sd = np.nanstd(loo_ref, axis=1, ddof=0)
        comb = combined_on_query(
            ctx["arrays"]["hypo"],
            ctx["arrays"]["hyper"],
            ctx["arrays"]["hypo_cnt"],
            ctx["arrays"]["hyper_cnt"],
            ctx["arrays"]["pct"],
            ref_idx,
            miss,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            filled = (comb - mu[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]
        ez[:, np.array(missing_pos, dtype=np.int64)] = filled
        console.print(
            f"  LOO ez: {n_saved} from profiles.tsv, "
            f"{len(missing_idx)} filled vs clean_ref n={ref_idx.size}"
        )
    else:
        console.print(f"  LOO ez: {n_saved} from profiles.tsv")
    if not np.isfinite(ez).all():
        n_bad = int((~np.isfinite(ez)).any(axis=0).sum())
        raise click.ClickException(f"LOO ez has non-finite columns for {n_bad} samples")
    return ez, n_saved


def meta_ez_matrix(meta: pd.DataFrame, samples: list[str], score_col: str) -> np.ndarray:
    idx = meta.set_index("sample")
    missing = [s for s in samples if s not in idx.index]
    if missing:
        raise click.ClickException(f"samples missing from meta: {missing[:8]}")
    col = score_col if score_col in idx.columns else None
    if col is None:
        raise click.ClickException(
            f"meta has no '{score_col}' column; available={list(idx.columns)[-6:]}"
        )
    rows = [parse_comma_scores(idx.at[s, col]) for s in samples]
    ez = np.vstack(rows).T
    if ez.shape != (len(CHR_LIST), len(samples)):
        raise click.ClickException(f"parsed {score_col} shape {ez.shape}")
    return ez


def write_profiles(
    ctx: dict,
    idx: np.ndarray,
    ez: np.ndarray,
    pred: np.ndarray,
    call: np.ndarray,
    path: Path,
) -> None:
    df = _cohort_frame(ctx, idx, ctx["label_arr"][idx])
    df["pred_label"] = pred
    df["call"] = call.astype(int)
    for i, chr_name in enumerate(CHR_LIST):
        df[chr_name] = ez[i]
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")


def write_panel(
    *,
    ctx: dict,
    idx: np.ndarray,
    ez_meta: np.ndarray,
    ez_loo: np.ndarray,
    labels: np.ndarray,
    call_meta: np.ndarray,
    call_loo: np.ndarray,
    cutoff_fixed: float,
    cutoff_loo: float,
    out: Path,
) -> dict[str, dict]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 10.4),
        layout="constrained",
        gridspec_kw={"width_ratios": [3.55, 0.95]},
    )
    rows = (
        (
            "fixed ref17+ez25",
            ez_meta,
            call_meta,
            cutoff_fixed,
            f"ez>{cutoff_fixed:g}",
            0,
        ),
        (
            "LOO clean",
            ez_loo,
            call_loo,
            cutoff_loo,
            f"ez>{cutoff_loo:g}",
            1,
        ),
    )
    all_met: dict[str, dict] = {}
    for name, ez, call, cut, call_desc, row_i in rows:
        ax_s, ax_b = axes[row_i, 0], axes[row_i, 1]
        df = _cohort_frame(ctx, idx, labels)
        met = detection_metrics(labels, call)
        all_met[name] = {**met, "call_rule": call_desc, "ez_cutoff": cut}
        plot_fixed_ez_scatter(
            df,
            ez,
            None,
            cutoff=cut,
            seed=row_i,
            target_mask=_target_mask_from_labels(labels.tolist()),
            title=f"{name}  n={idx.size}  (T#={met['n_pos']}, Normal={met['n_neg']})  ez>{cut:g}",
            ax=ax_s,
        )
        _plot_metric_bars(ax_b, met, f"{name}  Sens / Spec / PPV")
        console.print(
            f"  {name} [{call_desc}]: Sens={met['sens']:.3f} Spec={met['spec']:.3f} "
            f"PPV={met['ppv']:.3f}  TP/FN/FP/TN="
            f"{met['tp']}/{met['fn']}/{met['fp']}/{met['tn']}"
        )
    fig.suptitle(
        f"Test only · n={idx.size} · upper: meta final_zscores (ref17+ez25) ez>{cutoff_fixed:g} · "
        f"lower: LOO clean pool=80 ez>{cutoff_loo:g}",
        fontsize=12,
    )
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    console.print(f"  wrote {out}")
    return all_met


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--loo-dir", default=str(DEFAULT_LOO_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--fixed-cutoff", default=FIXED_CUTOFF, show_default=True, type=float)
@click.option("--loo-cutoff", default=LOO_CUTOFF, show_default=True, type=float)
@click.option(
    "--score-col",
    default="final_zscores",
    show_default=True,
    help="Meta column with comma-separated 22-chr ezscores (final_zscores / final_scores).",
)
def main(
    parquet: str,
    meta: str,
    toxic: str,
    loo_dir: str,
    output_dir: str,
    fixed_cutoff: float,
    loo_cutoff: float,
    score_col: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loo = Path(loo_dir)
    if not (loo / "profiles.tsv").is_file():
        raise click.ClickException(f"missing LOO profiles: {loo / 'profiles.tsv'}")

    meta_df = pd.read_csv(meta)
    meta_df["sample"] = meta_df["sample"].astype(str)
    if score_col not in meta_df.columns and score_col == "final_zscores" and "final_scores" in meta_df.columns:
        score_col = "final_scores"
        console.print("[yellow]using meta column final_scores[/yellow]")
    elif score_col not in meta_df.columns and "final_zscores" in meta_df.columns:
        console.print(f"[yellow]'{score_col}' missing; using final_zscores[/yellow]")
        score_col = "final_zscores"

    ctx = build_universe(Path(parquet), Path(meta), Path(toxic))
    idx = select_loo_test_idx(ctx, loo)
    console.print(f"  LOO test cohort n={idx.size}")
    samples = [ctx["ordered"][int(i)] for i in idx]
    labels = ctx["label_arr"][idx]
    pred_map = meta_df.set_index("sample")["pred_label"]
    pred = np.array(
        [pred_map.at[s] if s in pred_map.index else "" for s in samples],
        dtype=object,
    )

    ez_meta = meta_ez_matrix(meta_df, samples, score_col)
    ez_loo, n_loo_saved = loo_ez_from_saved(ctx, idx, loo)
    if n_loo_saved != idx.size:
        raise click.ClickException(
            f"expected all {idx.size} LOO test samples in profiles.tsv, saved={n_loo_saved}"
        )

    call_meta = np.nanmax(ez_meta, axis=0) > fixed_cutoff
    call_loo = np.nanmax(ez_loo, axis=0) > loo_cutoff
    write_profiles(ctx, idx, ez_meta, pred, call_meta, out / "profiles_meta.tsv")
    write_profiles(ctx, idx, ez_loo, pred, call_loo, out / "profiles_loo.tsv")

    all_met = write_panel(
        ctx=ctx,
        idx=idx,
        ez_meta=ez_meta,
        ez_loo=ez_loo,
        labels=labels,
        call_meta=call_meta,
        call_loo=call_loo,
        cutoff_fixed=fixed_cutoff,
        cutoff_loo=loo_cutoff,
        out=out / "pred_vs_loo.png",
    )
    payload = {
        "n": int(idx.size),
        "n_t": int(all_met["fixed ref17+ez25"]["n_pos"]),
        "n_normal": int(all_met["fixed ref17+ez25"]["n_neg"]),
        "score_col": score_col,
        "loo_dir": str(loo),
        "n_loo_from_profiles": int(n_loo_saved),
        "fixed_cutoff": fixed_cutoff,
        "loo_cutoff": loo_cutoff,
        "blacklist": sorted(BLACKLIST),
        "panel_fixed_ref17_ez25": all_met["fixed ref17+ez25"],
        "panel_loo_clean": all_met["LOO clean"],
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    console.print(f"  wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
