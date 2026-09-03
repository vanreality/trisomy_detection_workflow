#!/usr/bin/env python3
"""Compare 40+40 ref-free vs fixed E(ez μ/σ) trisomy detection.

E(ez_mean) and E(ez_std) come from LOO cross-fitting on the clean candidate
pool at pool_size=220:

    E(μ)_c = mean_r μ_{r,c}
    E(σ)_c = sqrt( mean_r σ_{r,c}^2 + Var_r(μ_{r,c}) )

Then for each Monte-Carlo draw of 40+40 from clean candidates:
  * 40+40 mode — episcore/zscore vs first 40; ezscore vs second 40
  * fixed mode — same episcore/zscore vs first 40; ezscore uses fixed E(μ), E(σ)

Eval positives: single-trisomy labels ``T#`` in set∈{dev,test}, depth_qc=pass,
ff_before_mq>0.01. Negatives: Normal with the same set/depth/FF filters
(for FPR / MCC). Call = any-chr ez > cutoff; also report target-chr hit rate.
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REF40_DIR) not in sys.path:
    sys.path.insert(0, str(REF40_DIR))

from grid_search_ref40 import CHR_LIST  # noqa: E402
from pool_size_ez_ref_bands import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_OUT,
    DEFAULT_PARQUET,
    DEFAULT_TOXIC,
    arm_seed,
    combined_on_query,
    expected_ez_params,
    load_candidates,
    run_arm,
)

console = Console()
_WORKER: dict = {}
DEFAULT_EZ_CUTOFF = 4.5
DEFAULT_REF_N = 40
DEFAULT_REPEATS = 10_000
DEFAULT_SEED = 42

REF40_COLOR = "#0D47A1"
FIXED_COLOR = "#FF6F00"


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


def label_to_target_chr(label: str) -> str | None:
    m = re.fullmatch(r"T(\d+)", str(label).strip())
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= 22:
        return f"chr{n}"
    return None


def load_eval_indices(meta_path: Path, sample_index: dict[str, int]) -> dict:
    meta = pd.read_csv(meta_path)
    meta["sample"] = meta["sample"].astype(str)
    ff = pd.to_numeric(meta["ff_before_mq"], errors="coerce")
    base = (
        meta["set"].isin(["dev", "test"])
        & (meta["depth_qc"].astype(str) == "pass")
        & (ff > 0.01)
        & meta["sample"].isin(sample_index)
    )
    pos = meta.loc[base & meta["label"].map(lambda x: label_to_target_chr(x) is not None)].copy()
    neg = meta.loc[base & (meta["label"].astype(str) == "Normal")].copy()
    pos["target_chr"] = pos["label"].map(label_to_target_chr)
    pos_idx = np.array([sample_index[s] for s in pos["sample"]], dtype=np.int64)
    neg_idx = np.array([sample_index[s] for s in neg["sample"]], dtype=np.int64)
    target_ci = np.array([CHR_LIST.index(c) for c in pos["target_chr"]], dtype=np.int64)
    return {
        "pos": pos.reset_index(drop=True),
        "neg": neg.reset_index(drop=True),
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "target_ci": target_ci,
        "eval_idx": np.concatenate([pos_idx, neg_idx]),
        "y": np.concatenate(
            [np.ones(len(pos_idx), dtype=bool), np.zeros(len(neg_idx), dtype=bool)]
        ),
    }


def load_or_compute_e_params(
    *,
    out_dir: Path,
    arrays: dict,
    clean_idx: np.ndarray,
    total_repeats: int,
    seed: int,
    n_jobs: int,
    pool_size: int = 220,
    arm: str = "loo_clean",
) -> tuple[np.ndarray, np.ndarray, Path]:
    raw_path = out_dir / "stats" / f"pool_{pool_size}" / f"{arm}_raw.npz"
    if raw_path.is_file():
        d = np.load(raw_path)
        if "e_mu" in d.files and "e_sd" in d.files:
            console.print(f"  loaded E(μ/σ) from {raw_path}")
            return d["e_mu"].astype(np.float64), d["e_sd"].astype(np.float64), raw_path
        e_mu, e_sd = expected_ez_params(d["ez_mu"], d["ez_sd"])
        return e_mu, e_sd, raw_path
    console.print(f"  computing {arm} pool={pool_size} for E(μ/σ) ({total_repeats} repeats)")
    mu, sd = run_arm(
        arrays=arrays,
        pool_idx=clean_idx,
        pool_size=pool_size,
        n_repeats=total_repeats,
        seed=arm_seed(seed, pool_size, arm),
        n_jobs=n_jobs,
        mode="loo",
    )
    e_mu, e_sd = expected_ez_params(mu, sd)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_path,
        ez_mu=mu.astype(np.float32),
        ez_sd=sd.astype(np.float32),
        e_mu=e_mu.astype(np.float32),
        e_sd=e_sd.astype(np.float32),
    )
    return e_mu, e_sd, raw_path


def _init_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)


def _run_detect_chunk(span: tuple[int, int]) -> dict[str, np.ndarray]:
    start, end = span
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    half = int(_WORKER["half"])
    arrays = _WORKER["arrays"]
    eval_idx = _WORKER["eval_idx"]
    target_ci = _WORKER["target_ci"]
    n_pos = int(_WORKER["n_pos"])
    e_mu = _WORKER["e_mu"]
    e_sd = _WORKER["e_sd"]
    cutoff = float(_WORKER["cutoff"])
    n = end - start
    out = {
        "ref40_any": np.zeros(n, dtype=np.int16),
        "fixed_any": np.zeros(n, dtype=np.int16),
        "ref40_target": np.zeros(n, dtype=np.int16),
        "fixed_target": np.zeros(n, dtype=np.int16),
        "ref40_fp": np.zeros(n, dtype=np.int16),
        "fixed_fp": np.zeros(n, dtype=np.int16),
        "ref40_max_ez_pos": np.zeros((n, n_pos), dtype=np.float32),
        "fixed_max_ez_pos": np.zeros((n, n_pos), dtype=np.float32),
        "ref40_tgt_ez_pos": np.zeros((n, n_pos), dtype=np.float32),
        "fixed_tgt_ez_pos": np.zeros((n, n_pos), dtype=np.float32),
    }
    n_neg = int(eval_idx.size - n_pos)
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
        # 40+40: ez-ref on held-out half
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
            ez_ref40 = (comb - mu[:, None]) / sd_safe
            ez_fixed = (comb - e_mu[:, None]) / np.where(e_sd > 0, e_sd, np.nan)[:, None]
        with np.errstate(invalid="ignore"):
            max_ref = np.nanmax(ez_ref40, axis=0)
            max_fix = np.nanmax(ez_fixed, axis=0)
        pos_ref = max_ref[:n_pos]
        pos_fix = max_fix[:n_pos]
        neg_ref = max_ref[n_pos:]
        neg_fix = max_fix[n_pos:]
        out["ref40_any"][i] = int(np.sum(pos_ref > cutoff))
        out["fixed_any"][i] = int(np.sum(pos_fix > cutoff))
        out["ref40_fp"][i] = int(np.sum(neg_ref > cutoff)) if n_neg else 0
        out["fixed_fp"][i] = int(np.sum(neg_fix > cutoff)) if n_neg else 0
        tgt_ref = ez_ref40[target_ci, np.arange(n_pos)]
        tgt_fix = ez_fixed[target_ci, np.arange(n_pos)]
        out["ref40_target"][i] = int(np.sum(tgt_ref > cutoff))
        out["fixed_target"][i] = int(np.sum(tgt_fix > cutoff))
        out["ref40_max_ez_pos"][i] = pos_ref.astype(np.float32)
        out["fixed_max_ez_pos"][i] = pos_fix.astype(np.float32)
        out["ref40_tgt_ez_pos"][i] = tgt_ref.astype(np.float32)
        out["fixed_tgt_ez_pos"][i] = tgt_fix.astype(np.float32)
    return out


def run_detection(
    *,
    arrays: dict,
    clean_idx: np.ndarray,
    eval_pack: dict,
    e_mu: np.ndarray,
    e_sd: np.ndarray,
    n_repeats: int,
    ref_n: int,
    seed: int,
    cutoff: float,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    need = 2 * ref_n
    if clean_idx.size < need:
        raise click.ClickException(f"clean pool {clean_idx.size} < {need}")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_repeats, need), dtype=np.int64)
    n_cand = int(clean_idx.size)
    for i in range(n_repeats):
        draws[i] = rng.permutation(n_cand)[:need]
    payload = {
        "pool_idx": clean_idx,
        "draws": draws,
        "half": ref_n,
        "arrays": arrays,
        "eval_idx": eval_pack["eval_idx"],
        "target_ci": eval_pack["target_ci"],
        "n_pos": int(eval_pack["pos_idx"].size),
        "e_mu": e_mu,
        "e_sd": e_sd,
        "cutoff": cutoff,
    }
    workers = _resolve_n_jobs(n_jobs)
    spans = _chunk_spans(n_repeats, workers)
    keys = [
        "ref40_any",
        "fixed_any",
        "ref40_target",
        "fixed_target",
        "ref40_fp",
        "fixed_fp",
        "ref40_max_ez_pos",
        "fixed_max_ez_pos",
        "ref40_tgt_ez_pos",
        "fixed_tgt_ez_pos",
    ]
    n_pos = int(eval_pack["pos_idx"].size)
    acc: dict[str, np.ndarray] = {}
    for k in keys:
        if k.endswith("_pos"):
            acc[k] = np.zeros((n_repeats, n_pos), dtype=np.float32)
        else:
            acc[k] = np.zeros(n_repeats, dtype=np.int16)
    if workers == 1 or len(spans) == 1:
        _init_worker(payload)
        chunk = _run_detect_chunk((0, n_repeats))
        for k in keys:
            acc[k] = chunk[k]
        return acc
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_init_worker,
        initargs=(payload,),
    ) as pool:
        futs = {pool.submit(_run_detect_chunk, span): span for span in spans}
        for fut in as_completed(futs):
            start, end = futs[fut]
            chunk = fut.result()
            for k in keys:
                acc[k][start:end] = chunk[k]
    return acc


def summarize_detection(acc: dict, n_pos: int, n_neg: int, cutoff: float) -> pd.DataFrame:
    rows = []
    for mode, any_k, tgt_k, fp_k in (
        ("ref40_40", "ref40_any", "ref40_target", "ref40_fp"),
        ("fixed_E", "fixed_any", "fixed_target", "fixed_fp"),
    ):
        tp_any = acc[any_k].astype(np.float64)
        tp_tgt = acc[tgt_k].astype(np.float64)
        fp = acc[fp_k].astype(np.float64)
        fn_any = n_pos - tp_any
        tn = n_neg - fp
        tpr = tp_any / n_pos if n_pos else np.nan
        tpr_tgt = tp_tgt / n_pos if n_pos else np.nan
        fpr = fp / n_neg if n_neg else np.nan
        # MCC on counts
        mcc = []
        for i in range(len(tp_any)):
            tp, fn, fp_i, tn_i = tp_any[i], fn_any[i], fp[i], tn[i]
            den = np.sqrt((tp + fp_i) * (tp + fn) * (tn_i + fp_i) * (tn_i + fn))
            mcc.append(((tp * tn_i - fp_i * fn) / den) if den > 0 else np.nan)
        mcc = np.asarray(mcc, dtype=np.float64)
        rows.append(
            {
                "mode": mode,
                "cutoff": cutoff,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_repeats": len(tp_any),
                "tpr_any_mean": float(np.nanmean(tpr)),
                "tpr_any_sd": float(np.nanstd(tpr, ddof=0)),
                "tpr_target_mean": float(np.nanmean(tpr_tgt)),
                "tpr_target_sd": float(np.nanstd(tpr_tgt, ddof=0)),
                "fpr_mean": float(np.nanmean(fpr)) if n_neg else float("nan"),
                "fpr_sd": float(np.nanstd(fpr, ddof=0)) if n_neg else float("nan"),
                "mcc_mean": float(np.nanmean(mcc)) if n_neg else float("nan"),
                "mcc_sd": float(np.nanstd(mcc, ddof=0)) if n_neg else float("nan"),
                "mean_tp_any": float(np.nanmean(tp_any)),
                "mean_tp_target": float(np.nanmean(tp_tgt)),
                "mean_fp": float(np.nanmean(fp)) if n_neg else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_detection(acc: dict, summary: pd.DataFrame, e_df: pd.DataFrame, out: Path, n_pos: int, n_neg: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    # E(mu), E(sd) bars
    ax = axes[0, 0]
    x = np.arange(len(CHR_LIST))
    ax.bar(x, e_df["e_mu"], color=FIXED_COLOR, width=0.8)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x, [c.replace("chr", "") for c in CHR_LIST], fontsize=7)
    ax.set_title("Fixed E(ez_mean) from LOO clean pool=220")
    ax.set_ylabel("E(μ)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.bar(x, e_df["e_sd"], color=FIXED_COLOR, width=0.8)
    ax.set_xticks(x, [c.replace("chr", "") for c in CHR_LIST], fontsize=7)
    ax.set_title(r"Fixed E(ez_std)=$\sqrt{E[\sigma^2]+Var(\mu)}$")
    ax.set_ylabel("E(σ)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # TPR / FPR violin-ish via histogram of rates
    ax = axes[1, 0]
    for mode, any_k, color in (
        ("40+40", "ref40_any", REF40_COLOR),
        ("fixed E", "fixed_any", FIXED_COLOR),
    ):
        rates = acc[any_k].astype(np.float64) / max(n_pos, 1)
        ax.hist(rates, bins=30, alpha=0.45, color=color, label=mode, density=True)
    ax.set_xlabel("TPR (any-chr call)")
    ax.set_ylabel("density")
    ax.set_title(f"Sensitivity on single trisomies (n={n_pos})")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    if n_neg > 0:
        for mode, fp_k, color in (
            ("40+40", "ref40_fp", REF40_COLOR),
            ("fixed E", "fixed_fp", FIXED_COLOR),
        ):
            rates = acc[fp_k].astype(np.float64) / n_neg
            ax.hist(rates, bins=30, alpha=0.45, color=color, label=mode, density=True)
        ax.set_xlabel("FPR (any-chr call on Normals)")
        ax.set_title(f"False-positive rate (n_neg={n_neg})")
    else:
        ax.text(0.5, 0.5, "no negatives", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("40+40 ref-free vs fixed E(ez μ/σ) trisomy detection", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    console.print(f"  wrote {out}")


def write_report(
    summary: pd.DataFrame,
    e_df: pd.DataFrame,
    eval_pack: dict,
    cfg: dict,
    out: Path,
) -> None:
    s40 = summary.loc[summary["mode"] == "ref40_40"].iloc[0]
    sf = summary.loc[summary["mode"] == "fixed_E"].iloc[0]
    report = f"""# Fixed E(ez μ/σ) vs 40+40 ref-free detection

## Fixed parameters (LOO clean, pool={cfg['e_pool_size']}, M={cfg['e_repeats']})

$$E(\\mu)_c = \\mathrm{{mean}}_r\\,\\mu_{{r,c}}$$

$$E(\\sigma)_c = \\sqrt{{\\mathrm{{mean}}_r\\,\\sigma_{{r,c}}^2 + \\mathrm{{Var}}_r(\\mu_{{r,c}})}}$$

| chr | E(μ) | E(σ) |
| --- | --- | --- |
"""
    for _, r in e_df.iterrows():
        report += f"| {r['chr']} | {r['e_mu']:.4f} | {r['e_sd']:.4f} |\n"
    report += f"""
## Eval set

- Positives: single `T#`, set∈{{dev,test}}, depth_qc=pass, ff_before_mq>0.01 → **{len(eval_pack['pos'])}**
- Negatives: Normal, same filters → **{len(eval_pack['neg'])}**
- Draw pool: clean candidates n={cfg['n_clean']}
- Repeats: {cfg['detect_repeats']}; ref_n={cfg['ref_n']}; cutoff={cfg['cutoff']}

## Results (mean ± SD over repeats)

| mode | TPR any-chr | TPR target-chr | FPR | MCC |
| --- | --- | --- | --- | --- |
| 40+40 | {s40['tpr_any_mean']:.3f}±{s40['tpr_any_sd']:.3f} | {s40['tpr_target_mean']:.3f}±{s40['tpr_target_sd']:.3f} | {s40['fpr_mean']:.3f}±{s40['fpr_sd']:.3f} | {s40['mcc_mean']:.3f}±{s40['mcc_sd']:.3f} |
| fixed E | {sf['tpr_any_mean']:.3f}±{sf['tpr_any_sd']:.3f} | {sf['tpr_target_mean']:.3f}±{sf['tpr_target_sd']:.3f} | {sf['fpr_mean']:.3f}±{sf['fpr_sd']:.3f} | {sf['mcc_mean']:.3f}±{sf['mcc_sd']:.3f} |

![detection](figures/fixed_vs_ref40_detection.png)

Both modes share the same 40 epi/z reference draws; only the ez normalization differs.
"""
    out.write_text(report)
    console.print(f"  wrote {out}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--e-pool-size", default=220, show_default=True, type=int)
@click.option("--e-arm", default="loo_clean", show_default=True, type=click.Choice(["loo_clean", "loo_total"]))
@click.option("--e-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--detect-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--ref-n", default=DEFAULT_REF_N, show_default=True, type=int)
@click.option("--cutoff", default=DEFAULT_EZ_CUTOFF, show_default=True, type=float)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option("--n-jobs", default=0, show_default=True, type=int)
def main(
    parquet: str,
    meta: str,
    toxic: str,
    output_dir: str,
    e_pool_size: int,
    e_arm: str,
    e_repeats: int,
    detect_repeats: int,
    ref_n: int,
    cutoff: float,
    seed: int,
    n_jobs: int,
) -> None:
    out = Path(output_dir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    det_dir = out / "detection"
    det_dir.mkdir(parents=True, exist_ok=True)

    pack = load_candidates(Path(parquet), Path(meta), Path(toxic))
    from pool_size_ez_ref_bands import _chr_block

    mat = pd.read_parquet(parquet)
    mat["sample"] = mat["sample"].astype(str)
    meta_df = pd.read_csv(meta)
    meta_df["sample"] = meta_df["sample"].astype(str)
    # Full universe: every meta sample present in the parquet (candidates + eval).
    want = [s for s in meta_df["sample"].tolist() if s in set(mat["sample"])]
    # Keep candidate order first, then remaining samples (trisomies etc.).
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
    clean_idx = np.array([sample_index[s] for s in pack["clean_samples"]], dtype=np.int64)

    eval_pack = load_eval_indices(Path(meta), sample_index)
    console.print(
        f"clean={len(pack['clean_samples'])} pos={len(eval_pack['pos'])} "
        f"neg={len(eval_pack['neg'])} universe={len(ordered)}"
    )

    workers = _resolve_n_jobs(n_jobs)
    # E(μ/σ) LOO uses candidate-feature arrays only (same order as load_candidates).
    e_mu, e_sd, raw_path = load_or_compute_e_params(
        out_dir=out,
        arrays=pack["arrays"],
        clean_idx=pack["clean_idx"],
        total_repeats=e_repeats,
        seed=seed,
        n_jobs=workers,
        pool_size=e_pool_size,
        arm=e_arm,
    )
    e_df = pd.DataFrame({"chr": CHR_LIST, "e_mu": e_mu, "e_sd": e_sd})
    e_df.to_csv(det_dir / "fixed_ez_params.tsv", sep="\t", index=False, float_format="%.8f")

    console.rule("[cyan]detection 40+40 vs fixed E")
    acc = run_detection(
        arrays=arrays,
        clean_idx=clean_idx,
        eval_pack=eval_pack,
        e_mu=e_mu,
        e_sd=e_sd,
        n_repeats=detect_repeats,
        ref_n=ref_n,
        seed=seed + 17,
        cutoff=cutoff,
        n_jobs=workers,
    )
    np.savez_compressed(det_dir / "detection_repeats.npz", **acc)
    summary = summarize_detection(
        acc, n_pos=len(eval_pack["pos"]), n_neg=len(eval_pack["neg"]), cutoff=cutoff
    )
    summary.to_csv(det_dir / "detection_summary.tsv", sep="\t", index=False, float_format="%.6f")
    eval_pack["pos"].to_csv(det_dir / "eval_positives.tsv", sep="\t", index=False)
    eval_pack["neg"][["sample", "set", "label", "ff_before_mq", "depth_qc"]].to_csv(
        det_dir / "eval_negatives.tsv", sep="\t", index=False
    )

    cfg = {
        "n_clean": len(pack["clean_samples"]),
        "e_pool_size": e_pool_size,
        "e_arm": e_arm,
        "e_repeats": e_repeats,
        "e_source": str(raw_path),
        "detect_repeats": detect_repeats,
        "ref_n": ref_n,
        "cutoff": cutoff,
        "seed": seed,
    }
    (det_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    plot_detection(
        acc,
        summary,
        e_df,
        figdir / "fixed_vs_ref40_detection.png",
        n_pos=len(eval_pack["pos"]),
        n_neg=len(eval_pack["neg"]),
    )
    write_report(summary, e_df, eval_pack, cfg, det_dir / "REPORT.md")
    console.print(summary.to_string(index=False))
    console.print(f"[green]OK[/green] -> {det_dir}")


if __name__ == "__main__":
    main()
