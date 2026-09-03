#!/usr/bin/env python3
"""Shared paths, constants, and loaders for reference-pool admittance analysis."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REF_FREE_DIR = SCRIPT_DIR.parent / "ref_free"
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
for _p in (REF_FREE_DIR, REF40_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grid_search_ref40 import CHR_LIST, _build_dense, compute_episcore, compute_zscore  # noqa: E402
from ref_free_ezscore import (  # noqa: E402
    _compute_ezscore,
    _flag_abnormal,
    _generate_half_partitions,
    _load_fixed_combo_arrays,
)
from separation import is_trisomy_label  # noqa: E402

DEFAULT_INPUT_DIR = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng"
)
DEFAULT_OUT_BASE = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
)
DEFAULT_DENSITY_TSV = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260810-ref_free_pool_size"
    "/fp_fn_density/fp_fn_density_ezscore.tsv"
)
UPDATED_SHEET = Path("/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet.csv")

DEFAULT_BLACKLIST = (
    "PTAY0577P9S1",
    "PTAY0599P8S1",
    "PTAY0666P7S1",
    "PTAY0682P7S1",
    "PTAY0689P8H1",
)

DEFAULT_EP_THRESHOLD = 0.5
DEFAULT_EP_RECALL = 0.65
DEFAULT_Z_THRESHOLD = 0.85
DEFAULT_Z_RECALL = 0.95
DEFAULT_CUTOFF = 3.0
DEFAULT_EZ_CUTOFF = 4.5
DEFAULT_REF_N = 40
DEFAULT_SEED = 42
DEFAULT_FF_MIN = 0.01
BAD_K = 5
WORST_K = 8
MAD_K = 0.6744897501960817  # 1 / sqrt(2) * erf-scale; ≈ Φ^{-1}(0.75)

FP_COLOR = "#F18F01"
FN_COLOR = "#2E86AB"
PERFECT_COLOR = "#A8DADC"
BAD_COLOR = "#C1121F"
OK_COLOR = "#6C757D"


def shard_start(path: Path) -> int:
    return int(path.stem.split("_")[1])


def parse_sample_list(path: Path) -> list[str]:
    """Read sample IDs from a txt/tsv (header ``sample`` optional)."""
    p = Path(path)
    if p.suffix.lower() in {".tsv", ".csv"}:
        df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        if "sample" in df.columns:
            return df["sample"].astype(str).tolist()
    raw = p.read_text().strip().splitlines()
    out: list[str] = []
    for line in raw:
        if not line.strip() or line.startswith("#"):
            continue
        out.append(line.split("\t")[0].strip())
    if out and out[0].lower() == "sample":
        out = out[1:]
    return out


def class_mask(fp_plus_fn: np.ndarray, name: str) -> np.ndarray:
    k = np.asarray(fp_plus_fn)
    if name == "perfect":
        return k == 0
    if name == "ok":
        return (k >= 1) & (k <= 4)
    if name == "bad":
        return k >= BAD_K
    if name == "worst":
        return k >= WORST_K
    raise ValueError(f"unknown class {name}")


def mad_z(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Robust z-score: 0.6745 * (x - median) / MAD. MAD=0 → 0."""
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = MAD_K * (x - med) / mad
    z = np.where(mad > 0, z, 0.0)
    return np.where(np.isfinite(z), z, 0.0)


def mad_z_vs_ref(values: np.ndarray, ref: np.ndarray, axis: int = 1) -> np.ndarray:
    """MAD-z of ``values`` using median/MAD estimated on ``ref`` only.

    Both arrays are chr × samples. Returns an array with the same shape as
    ``values``. Use this to freeze fences on a training pool and score new
    samples without letting the test pool shift the median/MAD.
    """
    x = np.asarray(values, dtype=float)
    r = np.asarray(ref, dtype=float)
    med = np.nanmedian(r, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(r - med), axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = MAD_K * (x - med) / mad
    z = np.where(mad > 0, z, 0.0)
    return np.where(np.isfinite(z), z, 0.0)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x>y) - P(x<y). Positive ⇒ x stochastically larger."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    # chunked broadcast to keep memory bounded
    n_gt = 0.0
    n_lt = 0.0
    step = max(1, 2_000_000 // max(y.size, 1))
    for i0 in range(0, x.size, step):
        chunk = x[i0 : i0 + step][:, None]
        n_gt += float((chunk > y).sum())
        n_lt += float((chunk < y).sum())
    denom = float(x.size * y.size)
    return (n_gt - n_lt) / denom if denom else float("nan")


def try_fisher(table: np.ndarray) -> tuple[float, float]:
    """Return (odds_ratio, p_value) for a 2x2 table. NaN if degenerate."""
    t = np.asarray(table, dtype=int)
    if t.shape != (2, 2) or (t < 0).any() or t.sum() == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import fisher_exact

        oddsr, p = fisher_exact(t, alternative="two-sided")
        return float(oddsr), float(p)
    except Exception:
        return float("nan"), float("nan")


def try_mannwhitney(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def overlay_ff(meta: pd.DataFrame, sheet: Path = UPDATED_SHEET) -> pd.DataFrame:
    out = meta.copy()
    out["sample"] = out["sample"].astype(str)
    out["ff_before_mq"] = pd.to_numeric(out["ff_before_mq"], errors="coerce")
    if sheet.is_file():
        upd = pd.read_csv(sheet).drop_duplicates("sample", keep="first")
        upd["sample"] = upd["sample"].astype(str)
        ff = dict(
            zip(upd["sample"], pd.to_numeric(upd["ff_before_mq"], errors="coerce"))
        )
        mapped = out["sample"].map(ff)
        out["ff_before_mq"] = mapped.where(mapped.notna(), out["ff_before_mq"])
        if "purity" in upd.columns:
            pur = dict(zip(upd["sample"], pd.to_numeric(upd["purity"], errors="coerce")))
            out["purity"] = out["sample"].map(pur)
    return out


def _concat_extra_grids(
    meta: pd.DataFrame,
    ep_df: pd.DataFrame,
    z_df: pd.DataFrame,
    extra_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append extra meta/ep/z; main ``sample`` IDs win on collision."""
    extra_meta = pd.read_csv(extra_dir / "meta.csv")
    extra_meta["sample"] = extra_meta["sample"].astype(str)
    extra_ep = pd.read_parquet(extra_dir / "episcore_grid_search.parquet")
    extra_z = pd.read_parquet(extra_dir / "zscore_grid_search.parquet")
    extra_ep["sample"] = extra_ep["sample"].astype(str)
    extra_z["sample"] = extra_z["sample"].astype(str)
    have = set(meta["sample"].astype(str))
    extra_meta = extra_meta.loc[~extra_meta["sample"].isin(have)].copy()
    extra_ep = extra_ep.loc[~extra_ep["sample"].isin(have)].copy()
    extra_z = extra_z.loc[~extra_z["sample"].isin(have)].copy()
    if extra_meta.empty:
        return meta, ep_df, z_df
    meta = pd.concat([meta, extra_meta], ignore_index=True, sort=False)
    ep_df = pd.concat([ep_df, extra_ep], ignore_index=True)
    z_df = pd.concat([z_df, extra_z], ignore_index=True)
    return meta, ep_df, z_df


def load_universe(
    input_dir: Path,
    *,
    ep_threshold: float = DEFAULT_EP_THRESHOLD,
    ep_recall: float = DEFAULT_EP_RECALL,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    z_recall: float = DEFAULT_Z_RECALL,
    blacklist: Iterable[str] = DEFAULT_BLACKLIST,
    ff_min: float = DEFAULT_FF_MIN,
    pool_samples: list[str] | None = None,
    pool_source: str = "dev_normal",
    exclude_eval_samples: Iterable[str] | None = None,
    eval_samples: Iterable[str] | None = None,
    extra_input_dir: Path | None = None,
) -> dict:
    """Mirror ``ref_free_fixed_flags`` universe / pool / eval construction.

    ``pool_source``:
      * ``dev_normal`` — 96 dev Normals, optionally filtered by ``pool_samples``
      * ``listed`` — ``pool_samples`` is the pool, in that order (any set)
    ``exclude_eval_samples`` are dropped from eval (use the original 96 test
    candidates so admitted/random-80 redraws share the same eval).
    ``eval_samples`` if set is an exact eval whitelist (shared across pools).
    ``extra_input_dir`` appends extra meta/ep/z (new sample IDs only).
    """
    meta = overlay_ff(
        pd.read_csv(input_dir / "meta.csv").drop_duplicates("sample", keep="first")
    )
    meta["sample"] = meta["sample"].astype(str)
    ep_df = pd.read_parquet(input_dir / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_dir / "zscore_grid_search.parquet")
    ep_df["sample"] = ep_df["sample"].astype(str)
    z_df["sample"] = z_df["sample"].astype(str)
    if extra_input_dir is not None:
        meta, ep_df, z_df = _concat_extra_grids(meta, ep_df, z_df, Path(extra_input_dir))
        meta = overlay_ff(meta.drop_duplicates("sample", keep="first"))
    ep_samples = set(ep_df["sample"].astype(str).unique())
    z_samples = set(z_df["sample"].astype(str).unique())
    universe = sorted(set(meta["sample"]) & ep_samples & z_samples)
    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}

    meta_idx = meta.set_index("sample").reindex(universe)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    ff_arr = pd.to_numeric(meta_idx["ff_before_mq"], errors="coerce").to_numpy()
    if "cpg_mean_coverage" in meta_idx.columns:
        cov = pd.to_numeric(meta_idx["cpg_mean_coverage"], errors="coerce").to_numpy()
    else:
        cov = np.full(len(universe), np.nan)
    if "final_zscores" in meta_idx.columns:
        max_abs_final = meta_idx["final_zscores"].map(_max_abs_csv)
    else:
        max_abs_final = pd.Series(np.nan, index=meta_idx.index)

    is_trisomy = np.array([bool(re.match(r"^T\d", s)) for s in label_arr])
    is_normal = label_arr == "Normal"
    is_dev_normal = (set_arr == "dev") & is_normal
    is_dev_trisomy = (set_arr == "dev") & is_trisomy
    is_test = set_arr == "test"
    samples = np.array(universe, dtype=object)
    eval_mask = is_dev_trisomy | is_test
    src = str(pool_source or "dev_normal")
    if src == "listed":
        if not pool_samples:
            raise ValueError("pool_source='listed' requires pool_samples")
        missing = [s for s in pool_samples if str(s) not in sample_index]
        if missing:
            raise ValueError(f"{len(missing)} pool samples missing from universe: {missing[:5]}")
        ref_pool_idx = np.array([sample_index[str(s)] for s in pool_samples], dtype=np.int64)
    else:
        ref_pool_idx = np.flatnonzero(is_dev_normal)
        if pool_samples is not None:
            allowed = {str(s) for s in pool_samples}
            keep = np.array([universe[i] in allowed for i in ref_pool_idx], dtype=bool)
            ref_pool_idx = ref_pool_idx[keep]
    if eval_samples is not None:
        wanted = [str(s).strip() for s in eval_samples if str(s).strip()]
        missing = [s for s in wanted if s not in sample_index]
        if missing:
            raise ValueError(f"{len(missing)} eval samples missing from universe: {missing[:5]}")
        eval_mask = np.isin(samples, wanted)
    elif exclude_eval_samples:
        drop = {str(s) for s in exclude_eval_samples if str(s).strip()}
        eval_mask = eval_mask & ~np.isin(samples, list(drop))
    eval_idx = np.flatnonzero(eval_mask)

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

    bl = {s.strip() for s in blacklist if str(s).strip()}
    y_all = np.array([is_trisomy_label(s) for s in label_arr], dtype=bool)
    if eval_samples is not None:
        keep_eval = eval_mask & ~np.isin(samples, list(bl))
    else:
        keep_eval = (
            (ff_arr >= ff_min)
            & (is_normal | is_trisomy)
            & ~np.isin(samples, list(bl))
            & eval_mask
        )
    eval_keep_idx = np.flatnonzero(keep_eval)

    return {
        "meta": meta,
        "universe": universe,
        "sample_index": sample_index,
        "chr_index": chr_index,
        "set_arr": set_arr,
        "label_arr": label_arr,
        "ff_arr": ff_arr,
        "coverage_arr": cov,
        "max_abs_final": max_abs_final.to_numpy(dtype=float),
        "is_normal": is_normal,
        "is_trisomy": is_trisomy,
        "ref_pool_idx": ref_pool_idx,
        "eval_idx": eval_idx,
        "eval_keep_idx": eval_keep_idx,
        "y_keep": y_all[eval_keep_idx],
        "ep_arrays": ep_arrays,
        "z_array": z_array,
        "ep_df": ep_df,
        "z_df": z_df,
        "pool_source": src,
    }


def _max_abs_csv(val: object) -> float:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return float("nan")
    parts = str(val).split(",")
    nums = []
    for p in parts:
        try:
            nums.append(abs(float(p)))
        except ValueError:
            continue
    return float(max(nums)) if nums else float("nan")


def score_one_repeat(
    ctx: dict,
    ref_local: np.ndarray,
    ez_local: np.ndarray,
    *,
    ez_cutoff: float = DEFAULT_EZ_CUTOFF,
) -> dict:
    """Score one 40+40 draw. ``ref_local`` / ``ez_local`` index into the pool array."""
    pool = ctx["ref_pool_idx"]
    ref_idx = pool[ref_local]
    ez_ref_idx = pool[ez_local]
    ep_arrays = ctx["ep_arrays"]
    z_array = ctx["z_array"]
    episcore = compute_episcore(
        np.expand_dims(ep_arrays[0], 0),
        np.expand_dims(ep_arrays[1], 0),
        np.expand_dims(ep_arrays[2], 0),
        np.expand_dims(ep_arrays[3], 0),
        ref_idx,
    )[0]
    zscore = compute_zscore(np.expand_dims(z_array, 0), ref_idx)[0]
    combined = episcore + zscore
    ez = _compute_ezscore(episcore, zscore, ez_ref_idx)
    flags = _flag_abnormal(ez, ctx["eval_keep_idx"], ez_cutoff).astype(bool)
    y = ctx["y_keep"]
    fp = int((flags & ~y).sum())
    fn = int((~flags & y).sum())

    ff = ctx["ff_arr"]
    ff_epz = ff[ref_idx]
    ff_ez = ff[ez_ref_idx]
    ff_80 = np.concatenate([ff_epz, ff_ez])

    with np.errstate(invalid="ignore"):
        ez_mu = np.nanmean(combined[:, ez_ref_idx], axis=1)
        ez_sd = np.nanstd(combined[:, ez_ref_idx], axis=1, ddof=0)
        pct_mu = np.nanmean(z_array[:, ref_idx], axis=1)
        pct_sd = np.nanstd(z_array[:, ref_idx], axis=1, ddof=0)

    def _ff_stats(a: np.ndarray) -> tuple[float, float, float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return (float(a.mean()), float(a.std(ddof=0)), float(a.min()), float(a.max()))

    return {
        "fp": fp,
        "fn": fn,
        "fp_plus_fn": fp + fn,
        "ff_epz": _ff_stats(ff_epz),
        "ff_ez": _ff_stats(ff_ez),
        "ff_80": _ff_stats(ff_80),
        "ez_mu": ez_mu.astype(np.float32),
        "ez_sd": ez_sd.astype(np.float32),
        "pct_mu": pct_mu.astype(np.float32),
        "pct_sd": pct_sd.astype(np.float32),
    }


def load_repeat_shards(score_dir: Path) -> dict:
    shards = sorted(score_dir.glob("repeats_*.npz"), key=shard_start)
    if not shards:
        raise FileNotFoundError(f"No repeats_*.npz under {score_dir}")
    parts = [np.load(p, allow_pickle=False) for p in shards]
    keys = [
        "repeat_id",
        "fp",
        "fn",
        "fp_plus_fn",
        "mem_epz",
        "mem_ez",
        "ff_epz_mean",
        "ff_epz_std",
        "ff_epz_min",
        "ff_epz_max",
        "ff_ez_mean",
        "ff_ez_std",
        "ff_ez_min",
        "ff_ez_max",
        "ff_80_mean",
        "ff_80_std",
        "ff_80_min",
        "ff_80_max",
        "ez_mu",
        "ez_sd",
        "pct_mu",
        "pct_sd",
    ]
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    cfg = {}
    cfg_path = score_dir / "run_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
    pool = pd.read_csv(score_dir / "pool_samples.tsv", sep="\t")
    out["cfg"] = cfg
    out["pool"] = pool
    return out


def density_table(fp: np.ndarray, fn: np.ndarray, tot: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"fp": fp, "fn": fn, "fp_plus_fn": tot})
    g = (
        df.groupby("fp_plus_fn", as_index=False)
        .agg(
            n_repeats=("fp_plus_fn", "size"),
            mean_fp=("fp", "mean"),
            mean_fn=("fn", "mean"),
        )
        .sort_values("fp_plus_fn")
        .reset_index(drop=True)
    )
    n = float(tot.size)
    g["density"] = g["n_repeats"].astype(float) / n
    g["cum_density"] = g["density"].cumsum()
    k = g["fp_plus_fn"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g["fp_share"] = np.where(k > 0, g["mean_fp"] / k, 0.0)
        g["fn_share"] = np.where(k > 0, g["mean_fn"] / k, 0.0)
    g["fp_density"] = g["density"] * g["fp_share"]
    g["fn_density"] = g["density"] * g["fn_share"]
    g["perfect_density"] = np.where(k == 0, g["density"], 0.0)
    return g


def fp_fn_summary(fp: np.ndarray, fn: np.ndarray, tot: np.ndarray) -> dict:
    tot = np.asarray(tot)
    fp = np.asarray(fp)
    fn = np.asarray(fn)
    return {
        "n_repeats": int(tot.size),
        "mean_fp": float(fp.mean()),
        "mean_fn": float(fn.mean()),
        "mean_fp_plus_fn": float(tot.mean()),
        "median_fp_plus_fn": float(np.median(tot)),
        "p95_fp_plus_fn": float(np.quantile(tot, 0.95)),
        "max_fp_plus_fn": int(tot.max()) if tot.size else 0,
        "frac_perfect": float((tot == 0).mean()),
        "frac_fp_plus_fn_ge_5": float((tot >= BAD_K).mean()),
        "frac_fp_plus_fn_ge_8": float((tot >= WORST_K).mean()),
    }
