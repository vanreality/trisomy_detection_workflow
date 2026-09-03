#!/usr/bin/env python3
"""0817 raw / clean / ref60: even-split and LOO, ratio + fixed plots.

Arms
----
raw   : ref = all dev Normal depth_qc=pass; pool_size=80
clean : raw minus top MAD until n=80; pool_size=80
ref60 : random 60 from raw (seed); pool_size=40

Eval (raw/clean)
----------------
raw_eval  : FF≥0.01, depth pass, (dev single T#) ∪ (test single T# ∪ test Normal)
clean_eval: raw_eval minus toxic sheet samples

Eval (ref60)
------------
ff≥0.01, depth pass,
  (dev single-T# + remaining Normal NOT in the 60) ∪ (test single-T# ∪ test Normal)

Global eval (every arm/mode)
----------------------------
ff_lt_1   : ff_before_mq < 0.01, depth pass, label in {Normal, single-T#} (any set)
emergency : set==emergency, depth pass; pred_label → effective label (Gray_* → Normal)

Modes
-----
1 even-split ratio · 2 even-split fixed E(μ)/E(σ) · 3 LOO ratio · 4 LOO fixed
"""

from __future__ import annotations

import json
import multiprocessing as mp
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
for _p in (SCRIPT_DIR, REF40_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grid_search_ref40 import CHR_LIST  # noqa: E402
from plot_ref40_vs_fixed_profiles import (  # noqa: E402
    DEFAULT_CUTOFF,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    PURITY_LO,
    _chunk_spans,
    _cohort_frame,
    _resolve_n_jobs,
    _target_mask_from_labels,
    _track_mu_sd,
    build_universe,
    fixed_ez_profiles_fully_fixed,
    label_to_target_chr,
    labels_to_target_chrs,
    plot_fixed_ez_scatter,
)
from plot_stable_ref40_compare import (  # noqa: E402
    BAR_LO,
    BAR_OK,
    _plot_metric_bars,
    detection_metrics,
)
from pool_size_ez_ref_bands import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_PARQUET,
    DEFAULT_TOXIC,
    combined_on_query,
    expected_ez_params,
    loo_combined,
)

console = Console()
_WORKER: dict = {}

DEFAULT_OUT = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260816-ref_free_dev/0817"
)
DEFAULT_MAD = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
    "/expanded_pool_mad/candidate_mad_scores.tsv"
)

ARM_TAG = {"raw": 0, "clean": 17, "ref60": 31}
DEFAULT_POOL_SIZE = {"raw": 80, "clean": 80, "ref60": 40}
REF60_N = 60
CLEAN_KEEP_N = 80
# Dropped from ratio/fixed plots and on-plot metrics (never mentioned on figures).
BLACKLIST = frozenset({"PTAY0577P9S1"})

MODE_DIR = {
    1: "mode1_even_ratio",
    2: "mode2_even_fixed",
    3: "mode3_loo_ratio",
    4: "mode4_loo_fixed",
}
MODE_TITLE = {
    1: "even split  ·  signal_ratio",
    2: "even split  ·  fixed E(μ)/E(σ) epi/z/ez",
    3: "LOO  ·  signal_ratio",
    4: "LOO  ·  fixed E(μ)/E(σ) epi/z/ez",
}

_HARD_T_RE = re.compile(r"^T(\d+)$")


def mode_dir(root: Path, arm: str, mode: int) -> Path:
    return root / arm / MODE_DIR[int(mode)]


def default_pool_size(arm: str) -> int:
    return int(DEFAULT_POOL_SIZE[arm])


def emergency_effective_label(pred: str) -> str:
    """Map emergency pred_label → effective label for scoring/metrics.

    Split on comma; skip Gray_* and Normal; keep tokens matching T(1..22).
    If any hard T# remain, join them ("T16" or "T15,T21"); else "Normal".
    """
    hard: list[str] = []
    seen: set[str] = set()
    for tok in str(pred or "").split(","):
        tok = tok.strip()
        if not tok or tok == "Normal" or tok.startswith("Gray_"):
            continue
        m = _HARD_T_RE.fullmatch(tok)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 22:
            lab = f"T{n}"
            if lab not in seen:
                seen.add(lab)
                hard.append(lab)
    return ",".join(hard) if hard else "Normal"


def _is_single_t(lab: str) -> bool:
    return label_to_target_chr(lab) is not None


def _is_any_t(lab: str) -> bool:
    return bool(labels_to_target_chrs(lab))


def _detection_metrics_any_t(labels: np.ndarray, call: np.ndarray) -> dict[str, float | int]:
    """Like detection_metrics, but multi-T labels (e.g. T15,T21) count as positive."""
    is_pos = np.array([_is_any_t(x) for x in labels])
    is_neg = np.asarray(labels) == "Normal"
    keep = is_pos | is_neg
    y = is_pos[keep]
    p = np.asarray(call, dtype=bool)[keep]
    tp = int((y & p).sum())
    fn = int((y & ~p).sum())
    fp = int((~y & p).sum())
    tn = int((~y & ~p).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
    }


def _metrics_for_labels(labels: np.ndarray, call: np.ndarray) -> dict[str, float | int]:
    """Use stock detection_metrics when all T# are single-token; else multi-T aware."""
    if all((not _is_any_t(x)) or _is_single_t(x) for x in labels):
        return detection_metrics(labels, call)
    return _detection_metrics_any_t(labels, call)


def _samples_to_idx(names: list[str], sample_index: dict[str, int]) -> np.ndarray:
    missing = [s for s in names if s not in sample_index]
    if missing:
        raise click.ClickException(f"samples missing from universe: {missing[:8]}")
    return np.array([sample_index[s] for s in names], dtype=np.int64)


def _eval_row(ctx: dict, i: int, toxic_set: set[str], *, label: str | None = None) -> dict:
    lab = ctx["label_arr"][i] if label is None else label
    s = ctx["ordered"][i]
    return {
        "sample": s,
        "set": ctx["set_arr"][i],
        "label": lab,
        "ff_before_mq": ctx["ff_arr"][i],
        "purity": ctx["purity_arr"][i],
        "target_chr": ",".join(labels_to_target_chrs(lab)),
        "in_toxic_sheet": s in toxic_set,
    }


def prepare_pools(
    ctx: dict,
    mad_path: Path,
    toxic_path: Path,
    out: Path,
    *,
    keep_n: int = CLEAN_KEEP_N,
    ref60_n: int = REF60_N,
    seed: int = DEFAULT_SEED,
) -> dict:
    ordered = ctx["ordered"]
    set_arr = ctx["set_arr"]
    label_arr = ctx["label_arr"]
    depth_arr = ctx["depth_arr"]
    ff_arr = ctx["ff_arr"]
    purity_arr = ctx["purity_arr"]
    pred_arr = ctx["pred_arr"]
    is_t = np.array([_is_single_t(x) for x in label_arr])
    is_norm = label_arr == "Normal"
    depth_ok = depth_arr == "pass"

    raw_ref_mask = (set_arr == "dev") & depth_ok & is_norm
    raw_ref_idx = np.flatnonzero(raw_ref_mask)
    mad = pd.read_csv(mad_path, sep="\t")
    mad["sample"] = mad["sample"].astype(str)
    mad_map = mad.set_index("sample")["mad_score"]
    toxic = pd.read_csv(toxic_path, sep="\t")
    toxic_set = set(toxic["sample"].astype(str))

    ref_rows = []
    for i in raw_ref_idx:
        s = ordered[i]
        ref_rows.append(
            {
                "sample": s,
                "set": set_arr[i],
                "label": label_arr[i],
                "ff_before_mq": ff_arr[i],
                "purity": purity_arr[i],
                "mad_score": float(mad_map.get(s, np.nan)),
                "in_toxic_sheet": s in toxic_set,
            }
        )
    raw_ref = pd.DataFrame(ref_rows)
    if len(raw_ref) < keep_n:
        raise click.ClickException(f"raw ref n={len(raw_ref)} < keep_n={keep_n}")
    if len(raw_ref) < ref60_n:
        raise click.ClickException(f"raw ref n={len(raw_ref)} < ref60_n={ref60_n}")

    ranked = raw_ref.sort_values(
        ["mad_score", "sample"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    n_drop = len(ranked) - keep_n
    dropped = ranked.iloc[:n_drop].copy()
    dropped["drop_rank"] = np.arange(1, n_drop + 1)
    clean_ref = ranked.iloc[n_drop:].sort_values("sample").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(raw_ref), size=ref60_n, replace=False)
    ref60_ref = (
        raw_ref.iloc[pick]
        .sort_values("sample")
        .reset_index(drop=True)
        .drop(columns=["mad_score", "in_toxic_sheet"], errors="ignore")
    )
    ref60_set = set(ref60_ref["sample"].astype(str))

    # --- raw / clean eval ---
    eval_mask = (
        depth_ok
        & (ff_arr >= 0.01)
        & (
            ((set_arr == "dev") & is_t)
            | ((set_arr == "test") & (is_t | is_norm))
        )
    )
    eval_idx = np.flatnonzero(eval_mask)
    raw_eval = pd.DataFrame([_eval_row(ctx, i, toxic_set) for i in eval_idx])
    clean_eval = raw_eval.loc[~raw_eval["in_toxic_sheet"]].reset_index(drop=True)
    tox_eval = raw_eval.loc[raw_eval["in_toxic_sheet"]].reset_index(drop=True)

    # --- ref60 eval ---
    remaining_dev_norm = (
        (set_arr == "dev") & depth_ok & is_norm
        & np.array([ordered[i] not in ref60_set for i in range(len(ordered))])
    )
    ref60_eval_mask = (
        depth_ok
        & (ff_arr >= 0.01)
        & (
            ((set_arr == "dev") & is_t)
            | remaining_dev_norm
            | ((set_arr == "test") & (is_t | is_norm))
        )
    )
    ref60_eval = pd.DataFrame(
        [_eval_row(ctx, i, toxic_set) for i in np.flatnonzero(ref60_eval_mask)]
    )

    # --- global evals ---
    ff_lt_mask = (
        depth_ok
        & (ff_arr < 0.01)
        & (is_norm | is_t)
    )
    global_ff_lt_1 = pd.DataFrame(
        [_eval_row(ctx, i, toxic_set) for i in np.flatnonzero(ff_lt_mask)]
    )

    emerg_mask = (set_arr == "emergency") & depth_ok
    emerg_rows = []
    for i in np.flatnonzero(emerg_mask):
        eff = emergency_effective_label(pred_arr[i])
        row = _eval_row(ctx, i, toxic_set, label=eff)
        row["pred_label"] = str(pred_arr[i])
        emerg_rows.append(row)
    global_emergency = pd.DataFrame(emerg_rows)

    pools = out / "pools"
    pools.mkdir(parents=True, exist_ok=True)
    raw_ref.sort_values("sample").to_csv(pools / "raw_ref.tsv", sep="\t", index=False)
    clean_ref.to_csv(pools / "clean_ref.tsv", sep="\t", index=False)
    dropped.to_csv(pools / "dropped_top_toxic_ref.tsv", sep="\t", index=False)
    ref60_ref.to_csv(pools / "ref60_ref.tsv", sep="\t", index=False)
    raw_eval.to_csv(pools / "raw_eval.tsv", sep="\t", index=False)
    clean_eval.to_csv(pools / "clean_eval.tsv", sep="\t", index=False)
    tox_eval.to_csv(pools / "toxic_removed_from_eval.tsv", sep="\t", index=False)
    ref60_eval.to_csv(pools / "ref60_eval.tsv", sep="\t", index=False)
    global_ff_lt_1.to_csv(pools / "global_ff_lt_1.tsv", sep="\t", index=False)
    global_emergency.to_csv(pools / "global_emergency.tsv", sep="\t", index=False)

    def _n_t(df: pd.DataFrame) -> int:
        return int(df["label"].map(_is_any_t).sum()) if len(df) else 0

    def _n_norm(df: pd.DataFrame) -> int:
        return int((df["label"] == "Normal").sum()) if len(df) else 0

    cfg = {
        "task": "0817",
        "seed": int(seed),
        "n_raw_ref": int(len(raw_ref)),
        "n_clean_ref": int(len(clean_ref)),
        "n_dropped_ref": int(len(dropped)),
        "n_ref60_ref": int(len(ref60_ref)),
        "n_raw_eval": int(len(raw_eval)),
        "n_clean_eval": int(len(clean_eval)),
        "n_toxic_eval": int(len(tox_eval)),
        "n_ref60_eval": int(len(ref60_eval)),
        "n_global_ff_lt_1": int(len(global_ff_lt_1)),
        "n_global_emergency": int(len(global_emergency)),
        "pool_size_raw_clean": keep_n,
        "pool_size_ref60": default_pool_size("ref60"),
        "ref60_n": ref60_n,
        "repeats": DEFAULT_REPEATS,
        "ez_cutoff": DEFAULT_CUTOFF,
        "raw_eval_n_t": _n_t(raw_eval),
        "raw_eval_n_normal": _n_norm(raw_eval),
        "clean_eval_n_t": _n_t(clean_eval),
        "clean_eval_n_normal": _n_norm(clean_eval),
        "ref60_eval_n_t": _n_t(ref60_eval),
        "ref60_eval_n_normal": _n_norm(ref60_eval),
        "ff_lt_1_n_t": _n_t(global_ff_lt_1),
        "ff_lt_1_n_normal": _n_norm(global_ff_lt_1),
        "emergency_n_t": _n_t(global_emergency),
        "emergency_n_normal": _n_norm(global_emergency),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "README.md").write_text(_readme_text(cfg) + "\n")
    console.print(
        f"raw_ref={cfg['n_raw_ref']} clean_ref={cfg['n_clean_ref']} "
        f"ref60_ref={cfg['n_ref60_ref']} dropped={cfg['n_dropped_ref']}"
    )
    console.print(
        f"raw_eval={cfg['n_raw_eval']} clean_eval={cfg['n_clean_eval']} "
        f"ref60_eval={cfg['n_ref60_eval']}  "
        f"ff_lt_1={cfg['n_global_ff_lt_1']} emergency={cfg['n_global_emergency']}"
    )
    return cfg


def _readme_text(cfg: dict) -> str:
    return f"""# 0817 raw / clean / ref60  ·  even-split / LOO

## Pools

| table | n | definition |
|---|---:|---|
| `pools/raw_ref.tsv` | {cfg['n_raw_ref']} | dev, depth pass, Normal |
| `pools/clean_ref.tsv` | {cfg['n_clean_ref']} | raw ref minus top-{cfg['n_dropped_ref']} MAD |
| `pools/ref60_ref.tsv` | {cfg['n_ref60_ref']} | random {cfg['ref60_n']} from raw_ref (seed={cfg['seed']}) |
| `pools/raw_eval.tsv` | {cfg['n_raw_eval']} | FF≥0.01, depth pass; dev T# + test T# + test Normal |
| `pools/clean_eval.tsv` | {cfg['n_clean_eval']} | raw eval minus toxic sheet |
| `pools/ref60_eval.tsv` | {cfg['n_ref60_eval']} | FF≥0.01, depth pass; dev T# + remaining dev Normal + test T#/Normal |
| `pools/global_ff_lt_1.tsv` | {cfg['n_global_ff_lt_1']} | FF<0.01, depth pass; Normal + single T# (any set) |
| `pools/global_emergency.tsv` | {cfg['n_global_emergency']} | set=emergency, depth pass; effective pred_label |

## Layout

```
0817/
  INDEX.html
  config.json
  pools/
  raw/    mode1_even_ratio | mode2_even_fixed | mode3_loo_ratio | mode4_loo_fixed
  clean/  (same four modes)
  ref60/  (same four modes; pool_size=40)
```

Repeats={cfg['repeats']}, pool_size raw/clean={cfg['pool_size_raw_clean']},
pool_size ref60={cfg['pool_size_ref60']}, ez cutoff={cfg['ez_cutoff']}.
"""


def _init_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)


def _run_even_ratio_chunk(span: tuple[int, int]) -> np.ndarray:
    start, end = span
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    half = int(_WORKER["half"])
    arrays = _WORKER["arrays"]
    eval_idx = _WORKER["eval_idx"]
    cutoff = float(_WORKER["cutoff"])
    n_pos = int(eval_idx.size)
    flags = np.zeros((end - start, n_pos), dtype=bool)
    for i, rid in enumerate(range(start, end)):
        drawn = pool_idx[draws[rid]]
        ref_idx = drawn[:half]
        ez_idx = drawn[half:]
        comb = combined_on_query(
            arrays["hypo"], arrays["hyper"], arrays["hypo_cnt"], arrays["hyper_cnt"],
            arrays["pct"], ref_idx, eval_idx,
        )
        comb_ez = combined_on_query(
            arrays["hypo"], arrays["hyper"], arrays["hypo_cnt"], arrays["hyper_cnt"],
            arrays["pct"], ref_idx, ez_idx,
        )
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(comb_ez, axis=1)
            sd = np.nanstd(comb_ez, axis=1, ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ez = (comb - mu[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]
            flags[i] = np.nanmax(ez, axis=0) > cutoff
    return flags


def _run_loo_ratio_chunk(span: tuple[int, int]) -> np.ndarray:
    start, end = span
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    arrays = _WORKER["arrays"]
    eval_idx = _WORKER["eval_idx"]
    cutoff = float(_WORKER["cutoff"])
    n_pos = int(eval_idx.size)
    flags = np.zeros((end - start, n_pos), dtype=bool)
    for i, rid in enumerate(range(start, end)):
        drawn = pool_idx[draws[rid]]
        comb = combined_on_query(
            arrays["hypo"], arrays["hyper"], arrays["hypo_cnt"], arrays["hyper_cnt"],
            arrays["pct"], drawn, eval_idx,
        )
        loo = loo_combined(
            arrays["hypo"][:, drawn],
            arrays["hyper"][:, drawn],
            arrays["hypo_cnt"][:, drawn],
            arrays["hyper_cnt"][:, drawn],
            arrays["pct"][:, drawn],
        )
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(loo, axis=1)
            sd = np.nanstd(loo, axis=1, ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ez = (comb - mu[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]
            flags[i] = np.nanmax(ez, axis=0) > cutoff
    return flags


def _run_even_params_chunk(span: tuple[int, int]) -> dict[str, np.ndarray]:
    start, end = span
    n = end - start
    n_chr = int(_WORKER["n_chr"])
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    half = int(_WORKER["half"])
    arrays = _WORKER["arrays"]
    out = {
        k: np.zeros((n, n_chr), dtype=np.float64)
        for k in (
            "hypo_mu", "hypo_sd", "hyper_mu", "hyper_sd",
            "pct_mu", "pct_sd", "ez_mu", "ez_sd",
        )
    }
    for i, rid in enumerate(range(start, end)):
        drawn = pool_idx[draws[rid]]
        epz = drawn[:half]
        ez_idx = drawn[half:]
        out["hypo_mu"][i], out["hypo_sd"][i] = _track_mu_sd(arrays["hypo"], epz)
        out["hyper_mu"][i], out["hyper_sd"][i] = _track_mu_sd(arrays["hyper"], epz)
        out["pct_mu"][i], out["pct_sd"][i] = _track_mu_sd(arrays["pct"], epz)
        comb = combined_on_query(
            arrays["hypo"], arrays["hyper"], arrays["hypo_cnt"], arrays["hyper_cnt"],
            arrays["pct"], epz, ez_idx,
        )
        with np.errstate(invalid="ignore"):
            out["ez_mu"][i] = np.nanmean(comb, axis=1)
            out["ez_sd"][i] = np.nanstd(comb, axis=1, ddof=0)
    return out


def _run_loo_params_chunk(span: tuple[int, int]) -> dict[str, np.ndarray]:
    start, end = span
    n = end - start
    n_chr = int(_WORKER["n_chr"])
    pool_idx = _WORKER["pool_idx"]
    draws = _WORKER["draws"]
    arrays = _WORKER["arrays"]
    out = {
        k: np.zeros((n, n_chr), dtype=np.float64)
        for k in (
            "hypo_mu", "hypo_sd", "hyper_mu", "hyper_sd",
            "pct_mu", "pct_sd", "ez_mu", "ez_sd",
        )
    }
    for i, rid in enumerate(range(start, end)):
        drawn = pool_idx[draws[rid]]
        out["hypo_mu"][i], out["hypo_sd"][i] = _track_mu_sd(arrays["hypo"], drawn)
        out["hyper_mu"][i], out["hyper_sd"][i] = _track_mu_sd(arrays["hyper"], drawn)
        out["pct_mu"][i], out["pct_sd"][i] = _track_mu_sd(arrays["pct"], drawn)
        loo = loo_combined(
            arrays["hypo"][:, drawn],
            arrays["hyper"][:, drawn],
            arrays["hypo_cnt"][:, drawn],
            arrays["hyper_cnt"][:, drawn],
            arrays["pct"][:, drawn],
        )
        with np.errstate(invalid="ignore"):
            out["ez_mu"][i] = np.nanmean(loo, axis=1)
            out["ez_sd"][i] = np.nanstd(loo, axis=1, ddof=0)
    return out


def _parallel_flags(fn, payload: dict, n_repeats: int, n_jobs: int, n_eval: int) -> np.ndarray:
    workers = _resolve_n_jobs(n_jobs)
    spans = _chunk_spans(n_repeats, workers)
    flags = np.zeros((n_repeats, n_eval), dtype=bool)
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
            flags[start:end] = fut.result()
    return flags


def _parallel_params(fn, payload: dict, n_repeats: int, n_jobs: int, n_chr: int) -> dict[str, np.ndarray]:
    keys = (
        "hypo_mu", "hypo_sd", "hyper_mu", "hyper_sd",
        "pct_mu", "pct_sd", "ez_mu", "ez_sd",
    )
    workers = _resolve_n_jobs(n_jobs)
    spans = _chunk_spans(n_repeats, workers)
    stacked = {k: np.zeros((n_repeats, n_chr), dtype=np.float64) for k in keys}
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
            part = fut.result()
            for k in keys:
                stacked[k][start:end] = part[k]
    return stacked


def _make_draws(pool_idx: np.ndarray, pool_size: int, n_repeats: int, seed: int) -> np.ndarray:
    if pool_idx.size < pool_size:
        raise click.ClickException(f"pool {pool_idx.size} < pool_size {pool_size}")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_repeats, pool_size), dtype=np.int64)
    n_cand = int(pool_idx.size)
    for i in range(n_repeats):
        draws[i] = rng.permutation(n_cand)[:pool_size]
    return draws


def score_ratio(
    *,
    arrays: dict,
    pool_idx: np.ndarray,
    eval_idx: np.ndarray,
    mode: int,
    n_repeats: int,
    pool_size: int,
    seed: int,
    cutoff: float,
    n_jobs: int,
) -> np.ndarray:
    draws = _make_draws(pool_idx, pool_size, n_repeats, seed)
    payload = {
        "pool_idx": pool_idx,
        "draws": draws,
        "half": pool_size // 2,
        "arrays": arrays,
        "eval_idx": eval_idx,
        "cutoff": cutoff,
    }
    fn = _run_even_ratio_chunk if mode == 1 else _run_loo_ratio_chunk
    return _parallel_flags(fn, payload, n_repeats, n_jobs, eval_idx.size)


def score_fixed_params(
    *,
    arrays: dict,
    pool_idx: np.ndarray,
    mode: int,
    n_repeats: int,
    pool_size: int,
    seed: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    draws = _make_draws(pool_idx, pool_size, n_repeats, seed)
    n_chr = arrays["hypo"].shape[0]
    payload = {
        "pool_idx": pool_idx,
        "draws": draws,
        "half": pool_size // 2,
        "arrays": arrays,
        "n_chr": n_chr,
    }
    fn = _run_even_params_chunk if mode == 2 else _run_loo_params_chunk
    stacked = _parallel_params(fn, payload, n_repeats, n_jobs, n_chr)
    params = {}
    for track in ("hypo", "hyper", "pct", "ez"):
        e_mu, e_sd = expected_ez_params(stacked[f"{track}_mu"], stacked[f"{track}_sd"])
        params[f"{track}_mu"] = e_mu.astype(np.float32)
        params[f"{track}_sd"] = e_sd.astype(np.float32)
    return params


def write_ratio_tsv(
    ctx: dict,
    eval_idx: np.ndarray,
    ratio: np.ndarray,
    path: Path,
    *,
    labels: np.ndarray | None = None,
    cohort: str | None = None,
) -> pd.DataFrame:
    labs = ctx["label_arr"][eval_idx] if labels is None else np.asarray(labels)
    df = pd.DataFrame(
        {
            "sample": [ctx["ordered"][i] for i in eval_idx],
            "set": ctx["set_arr"][eval_idx],
            "label": labs,
            "ff_before_mq": ctx["ff_arr"][eval_idx],
            "purity": ctx["purity_arr"][eval_idx],
            "y_true": np.array([_is_any_t(x) for x in labs], dtype=int),
            "signal_ratio": ratio,
        }
    )
    if cohort is not None:
        df.insert(0, "cohort", cohort)
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")
    return df


def write_profiles_tsv(
    ctx: dict,
    eval_idx: np.ndarray,
    ez: np.ndarray,
    path: Path,
    *,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    labs = ctx["label_arr"][eval_idx] if labels is None else np.asarray(labels)
    wide = pd.DataFrame(
        {
            "sample": [ctx["ordered"][i] for i in eval_idx],
            "set": ctx["set_arr"][eval_idx],
            "label": labs,
            "ff_before_mq": ctx["ff_arr"][eval_idx],
            "purity": ctx["purity_arr"][eval_idx],
        }
    )
    for i, chr_name in enumerate(CHR_LIST):
        wide[chr_name] = ez[i]
    wide.to_csv(path, sep="\t", index=False, float_format="%.6f")
    return wide


def _filter_blacklist_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sample" not in df.columns:
        return df
    return df.loc[~df["sample"].astype(str).isin(BLACKLIST)].reset_index(drop=True)


def _filter_blacklist_arrays(
    ctx: dict,
    idx: np.ndarray,
    labels: np.ndarray,
    ez: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (idx, ez, labels) with BLACKLIST samples removed."""
    if idx.size == 0:
        return idx, ez, labels
    keep = np.array([ctx["ordered"][int(i)] not in BLACKLIST for i in idx], dtype=bool)
    return idx[keep], ez[:, keep], np.asarray(labels)[keep]


def write_interactive_ratio_html(
    df: pd.DataFrame,
    out: Path,
    *,
    title: str,
    n_repeats: int,
    pool_n: int,
    arm: str,
    default_cut: float = 0.5,
) -> None:
    """Interactive scatter of all cohorts; metrics slider uses cohort==main only."""
    if "cohort" not in df.columns:
        df = df.assign(cohort="main")
    df = _filter_blacklist_df(df)

    purity = pd.to_numeric(df["purity"], errors="coerce").to_numpy()
    lo = purity < PURITY_LO
    is_main = df["cohort"].to_numpy() == "main"
    is_ff = df["cohort"].to_numpy() == "ff_lt_1"
    is_em = df["cohort"].to_numpy() == "emergency"
    is_dev = df["set"].to_numpy() == "dev"
    is_test = df["set"].to_numpy() == "test"
    is_t = df["y_true"].to_numpy() == 1
    is_norm = df["label"].to_numpy() == "Normal"

    groups: list[tuple[str, np.ndarray, str, str]] = [
        ("main · dev T# (purity≥0.8)", is_main & is_dev & is_t & ~lo, "#C1121F", "circle"),
        ("main · dev T# (purity<0.8)", is_main & is_dev & is_t & lo, "#1D4ED8", "diamond"),
        ("main · test T# (purity≥0.8)", is_main & is_test & is_t & ~lo, "#E07A3D", "circle"),
        ("main · test T# (purity<0.8)", is_main & is_test & is_t & lo, "#7C3AED", "diamond"),
        ("main · test Normal", is_main & is_test & is_norm, "#9E9E9E", "square"),
    ]
    if arm == "ref60":
        groups.append(
            ("main · dev Normal", is_main & is_dev & is_norm, "#6B7280", "square"),
        )
    groups.extend(
        [
            ("ff_lt_1 · T#", is_ff & is_t, "#0D9488", "circle"),
            ("ff_lt_1 · Normal", is_ff & is_norm, "#99F6E4", "square"),
            ("emergency · T#", is_em & is_t, "#BE185D", "circle"),
            ("emergency · Normal", is_em & is_norm, "#F9A8D4", "square"),
        ]
    )

    payload = []
    for _, row in df.iterrows():
        payload.append(
            {
                "sample": str(row["sample"]),
                "set": str(row["set"]),
                "label": str(row["label"]),
                "cohort": str(row["cohort"]),
                "ff": None if pd.isna(row["ff_before_mq"]) else float(row["ff_before_mq"]),
                "purity": None if pd.isna(row["purity"]) else float(row["purity"]),
                "ratio": float(row["signal_ratio"]),
                "y": int(row["y_true"]),
            }
        )
    traces_js = []
    for name, mask, color, symbol in groups:
        sub = df.loc[mask]
        if sub.empty:
            continue
        traces_js.append(
            {
                "name": name,
                "x": sub["signal_ratio"].astype(float).tolist(),
                "y": pd.to_numeric(sub["ff_before_mq"], errors="coerce").tolist(),
                "text": sub["sample"].astype(str).tolist(),
                "color": color,
                "symbol": symbol,
            }
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #1f2933; }}
  body {{ margin: 0; background: #f4f6f8; }}
  .wrap {{ max-width: 1180px; margin: 24px auto 40px; padding: 0 20px; }}
  h1 {{ font-size: 1.35rem; font-weight: 650; margin: 0 0 6px; }}
  .sub {{ color: #5b6770; margin-bottom: 18px; font-size: 0.92rem; }}
  .card {{ background: #fff; border-radius: 12px; box-shadow: 0 1px 4px rgba(16,24,40,.08); padding: 16px 18px 10px; }}
  .ctrl {{ display: flex; align-items: center; gap: 14px; margin: 8px 0 4px; }}
  input[type=range] {{ flex: 1; accent-color: #1f3a5f; }}
  .cutval {{ font-variant-numeric: tabular-nums; font-weight: 650; min-width: 3.6rem; }}
  #bars {{ height: 280px; }}
  #scatter {{ height: 560px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <div class="sub">pool n={pool_n} · {n_repeats} repeats · call trisomy if signal_ratio ≥ cutoff · metrics on <b>main</b> only · hover for sample id</div>
  <div class="card">
    <div class="ctrl">
      <label for="cut">ratio cutoff</label>
      <input id="cut" type="range" min="0" max="100" value="{int(round(default_cut * 100))}"/>
      <span class="cutval" id="cutlab">{default_cut:.2f}</span>
    </div>
    <div id="scatter"></div>
    <div id="bars"></div>
  </div>
</div>
<script>
const ROWS = {json.dumps(payload)};
const TRACES = {json.dumps(traces_js)};
const DEFAULT_CUT = {default_cut};
const BAR_OK = "{BAR_OK}";
const BAR_LO = "{BAR_LO}";

function metrics(cut) {{
  let tp=0, fn=0, fp=0, tn=0;
  for (const r of ROWS) {{
    if (r.cohort !== "main") continue;
    const pred = r.ratio >= cut;
    if (r.y === 1) {{ if (pred) tp++; else fn++; }}
    else {{ if (pred) fp++; else tn++; }}
  }}
  const sens = (tp+fn) ? tp/(tp+fn) : null;
  const spec = (tn+fp) ? tn/(tn+fp) : null;
  const ppv  = (tp+fp) ? tp/(tp+fp) : null;
  return {{tp, fn, fp, tn, sens, spec, ppv}};
}}

function scatterTraces(cut, ymax) {{
  const tr = TRACES.map(t => ({{
    type: "scatter",
    mode: "markers",
    name: t.name,
    x: t.x, y: t.y, text: t.text,
    hovertemplate: "%{{text}}<br>ratio=%{{x:.3f}}<br>FF=%{{y:.4f}}<extra>%{{fullData.name}}</extra>",
    marker: {{
      size: t.symbol === "diamond" ? 11 : 9,
      color: t.color,
      symbol: t.symbol,
      line: {{color: "#333", width: 0.4}},
      opacity: 0.9
    }}
  }}));
  tr.push({{
    type: "scatter", mode: "lines", name: "ratio_cutoff",
    x: [cut, cut], y: [-0.005, ymax],
    line: {{color: "#111", width: 2, dash: "dash"}},
    hoverinfo: "skip"
  }});
  return tr;
}}

function barTrace(m) {{
  const vals = [m.sens, m.spec, m.ppv].map(v => v === null ? 0 : v);
  const colors = [m.sens, m.spec, m.ppv].map(v => (v !== null && Math.abs(v-1)<1e-12) ? BAR_OK : BAR_LO);
  const text = [m.sens, m.spec, m.ppv].map(v => v === null ? "n/a" : v.toFixed(3));
  return [{{
    type: "bar", x: ["Sens", "Spec", "PPV"], y: vals,
    marker: {{color: colors, line: {{color: "#fff", width: 1}}}},
    width: 0.42,
    text: text, textposition: "outside",
    hovertemplate: "%{{x}}=%{{text}}<extra></extra>",
    showlegend: false
  }}];
}}

function render(cut) {{
  const m = metrics(cut);
  document.getElementById("cutlab").textContent = cut.toFixed(2);
  const ymax = Math.max(0.08, ...ROWS.map(r => r.ff || 0)) * 1.08;
  Plotly.react("scatter", scatterTraces(cut, ymax), {{
    template: "plotly_white",
    margin: {{t: 36, r: 20, b: 48, l: 56}},
    xaxis: {{title: "signal_ratio", range: [-0.03, 1.03], zeroline: false}},
    yaxis: {{title: "ff_before_mq", range: [-0.005, ymax], zeroline: false}},
    legend: {{orientation: "h", y: 1.12, x: 0, font: {{size: 10}}}},
    shapes: [],
    annotations: [{{
      text: "over cutoff → trisomy  ·  metrics = main only", xref: "paper", yref: "paper",
      x: 1, y: 1.02, showarrow: false, font: {{size: 11, color: "#5b6770"}}, xanchor: "right"
    }}]
  }}, {{displayModeBar: true, responsive: true}});
  Plotly.react("bars", barTrace(m), {{
    template: "plotly_white",
    margin: {{t: 48, r: 20, b: 40, l: 40}},
    yaxis: {{range: [0, 1.18], title: ""}},
    title: {{text: `Sens/Spec/PPV (main)  ·  TP ${{m.tp}}  FN ${{m.fn}}  FP ${{m.fp}}  TN ${{m.tn}}`, font: {{size: 14}}}},
    bargap: 0.45
  }}, {{displayModeBar: false, responsive: true}});
}}

const slider = document.getElementById("cut");
slider.addEventListener("input", () => render(slider.value / 100));
render(DEFAULT_CUT);
</script>
</body>
</html>
"""
    out.write_text(html)
    console.print(f"  wrote {out}")


def _empty_metrics() -> dict[str, float | int]:
    return {
        "tp": 0, "fn": 0, "fp": 0, "tn": 0,
        "n_pos": 0, "n_neg": 0,
        "sens": float("nan"), "spec": float("nan"), "ppv": float("nan"),
    }


def write_fixed_panel(
    *,
    ctx: dict,
    out: Path,
    cutoff: float,
    suptitle: str,
    main_idx: np.ndarray,
    main_ez: np.ndarray,
    main_labels: np.ndarray,
    ff_idx: np.ndarray,
    ff_ez: np.ndarray,
    ff_labels: np.ndarray,
    em_idx: np.ndarray,
    em_ez: np.ndarray,
    em_labels: np.ndarray,
) -> dict:
    """4-row × 2-col: main-dev, main-test, ff_lt_1, emergency (scatter | bars)."""
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13.6, 18),
        layout="constrained",
        gridspec_kw={"width_ratios": [3.55, 0.95]},
    )

    is_dev = ctx["set_arr"][main_idx] == "dev"
    is_test = ctx["set_arr"][main_idx] == "test"
    rows = [
        (
            "main-dev",
            *_filter_blacklist_arrays(
                ctx, main_idx[is_dev], main_labels[is_dev], main_ez[:, np.flatnonzero(is_dev)]
            ),
            0,
        ),
        (
            "main-test",
            *_filter_blacklist_arrays(
                ctx, main_idx[is_test], main_labels[is_test], main_ez[:, np.flatnonzero(is_test)]
            ),
            1,
        ),
        ("ff_lt_1", *_filter_blacklist_arrays(ctx, ff_idx, ff_labels, ff_ez), 2),
        ("emergency", *_filter_blacklist_arrays(ctx, em_idx, em_labels, em_ez), 3),
    ]
    all_met: dict[str, dict] = {}
    for name, idx, ez, labels, row_i in rows:
        ax_s, ax_b = axes[row_i, 0], axes[row_i, 1]
        if idx.size == 0:
            ax_s.set_axis_off()
            ax_b.set_axis_off()
            ax_s.set_title(f"{name}  n=0 (empty)", fontsize=10)
            all_met[name] = _empty_metrics()
            continue
        df = _cohort_frame(ctx, idx, labels)
        call = np.nanmax(ez, axis=0) > cutoff
        met = _metrics_for_labels(labels, call)
        all_met[name] = met
        plot_fixed_ez_scatter(
            df,
            ez,
            None,
            cutoff=cutoff,
            seed=row_i,
            target_mask=_target_mask_from_labels(labels.tolist()),
            title=f"{name}  n={idx.size}  (T#={met['n_pos']}, Normal={met['n_neg']})",
            ax=ax_s,
        )
        _plot_metric_bars(ax_b, met, f"{name}  Sens / Spec / PPV")
        console.print(
            f"  {name}: Sens={met['sens']:.3f} Spec={met['spec']:.3f} "
            f"PPV={met['ppv']:.3f}  TP/FN/FP/TN="
            f"{met['tp']}/{met['fn']}/{met['fp']}/{met['tn']}"
        )

    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    console.print(f"  wrote {out}")
    return all_met


def write_index(root: Path) -> None:
    cards = []
    for arm in ("raw", "clean", "ref60"):
        for mode in (1, 2, 3, 4):
            d = mode_dir(root, arm, mode)
            rel = d.relative_to(root)
            html = d / "ratio.html"
            png = d / "fixed.png"
            tsv = d / "signal_ratio.tsv" if mode in (1, 3) else d / "profiles.tsv"
            links = []
            if html.is_file():
                links.append(f'<a href="{rel}/ratio.html">ratio.html</a>')
            if png.is_file():
                links.append(f'<a href="{rel}/fixed.png">fixed.png</a>')
            if tsv.is_file():
                links.append(f'<a href="{rel}/{tsv.name}">{tsv.name}</a>')
            thumb = f'<img src="{rel}/fixed.png" alt="fixed"/>' if png.is_file() else ""
            cards.append(
                f'<div class="card"><h3>{arm} · {MODE_DIR[mode]}</h3>'
                f'<p class="muted">{MODE_TITLE[mode]} · pool={default_pool_size(arm)}</p>'
                f'{thumb}<p>{" · ".join(links) if links else "<span class=muted>pending</span>"}</p></div>'
            )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>0817 index</title>
<style>
body {{ font-family: "Segoe UI", Arial, sans-serif; background:#f4f6f8; margin:0; color:#1f2933; }}
.wrap {{ max-width: 1100px; margin: 28px auto; padding: 0 20px 40px; }}
h1 {{ font-size: 1.4rem; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.card {{ background:#fff; border-radius:12px; padding:14px 16px; box-shadow:0 1px 4px rgba(16,24,40,.08); }}
.card img {{ width:100%; border-radius:8px; margin:8px 0; }}
.muted {{ color:#5b6770; font-size:.9rem; }}
a {{ color:#1d4ed8; }}
</style></head>
<body><div class="wrap">
<h1>0817 raw / clean / ref60 · even-split / LOO</h1>
<p class="muted"><a href="README.md">README</a> · <a href="config.json">config.json</a> · <a href="pools/">pools/</a></p>
<div class="grid">{''.join(cards)}</div>
</div></body></html>
"""
    (root / "INDEX.html").write_text(html)
    console.print(f"  wrote {root / 'INDEX.html'}")


def _load_arm_indices(ctx: dict, root: Path, arm: str) -> tuple[np.ndarray, np.ndarray]:
    ref = pd.read_csv(root / "pools" / f"{arm}_ref.tsv", sep="\t")
    ev = pd.read_csv(root / "pools" / f"{arm}_eval.tsv", sep="\t")
    pool_idx = _samples_to_idx(ref["sample"].astype(str).tolist(), ctx["sample_index"])
    eval_idx = _samples_to_idx(ev["sample"].astype(str).tolist(), ctx["sample_index"])
    return pool_idx, eval_idx


def _load_global_indices(
    ctx: dict, root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ff_lt_idx, ff_lt_labels, emerg_idx, emerg_labels)."""
    ff = pd.read_csv(root / "pools" / "global_ff_lt_1.tsv", sep="\t")
    em = pd.read_csv(root / "pools" / "global_emergency.tsv", sep="\t")
    ff_idx = _samples_to_idx(ff["sample"].astype(str).tolist(), ctx["sample_index"])
    em_idx = _samples_to_idx(em["sample"].astype(str).tolist(), ctx["sample_index"])
    ff_labels = ff["label"].astype(str).to_numpy()
    em_labels = em["label"].astype(str).to_numpy()
    return ff_idx, ff_labels, em_idx, em_labels


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """0817 raw/clean/ref60 even-split and LOO scoring + plots."""


@cli.command()
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--mad", default=str(DEFAULT_MAD), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
def prepare(parquet: str, meta: str, toxic: str, mad: str, output_dir: str, seed: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = build_universe(Path(parquet), Path(meta), Path(toxic))
    prepare_pools(ctx, Path(mad), Path(toxic), out, seed=seed)
    write_index(out)


@cli.command()
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--arm", type=click.Choice(["raw", "clean", "ref60"]), required=True)
@click.option("--mode", type=click.Choice(["1", "2", "3", "4"]), required=True)
@click.option("--total-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--cutoff", default=DEFAULT_CUTOFF, show_default=True, type=float)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option(
    "--pool-size",
    default=None,
    type=int,
    help="Default: 40 for ref60, else 80",
)
def score(
    parquet: str,
    meta: str,
    toxic: str,
    output_dir: str,
    arm: str,
    mode: str,
    total_repeats: int,
    cutoff: float,
    seed: int,
    n_jobs: int,
    pool_size: int | None,
) -> None:
    out = Path(output_dir)
    mode_i = int(mode)
    dest = mode_dir(out, arm, mode_i)
    dest.mkdir(parents=True, exist_ok=True)
    if pool_size is None:
        pool_size = default_pool_size(arm)
    ctx = build_universe(Path(parquet), Path(meta), Path(toxic))
    pool_idx, eval_idx = _load_arm_indices(ctx, out, arm)
    ff_idx, ff_labels, em_idx, em_labels = _load_global_indices(ctx, out)
    arm_seed = seed + ARM_TAG[arm] * 1009 + mode_i * 97
    console.rule(
        f"[cyan]{arm} mode{mode_i}  pool={pool_idx.size} "
        f"eval={eval_idx.size} ff_lt_1={ff_idx.size} emerg={em_idx.size} "
        f"pool_size={pool_size} repeats={total_repeats}"
    )

    if mode_i in (1, 3):
        # One MC over concatenated query indices, then split.
        parts = [eval_idx, ff_idx, em_idx]
        sizes = [int(p.size) for p in parts]
        all_idx = np.concatenate(parts) if sum(sizes) else eval_idx
        flags_all = score_ratio(
            arrays=ctx["arrays"],
            pool_idx=pool_idx,
            eval_idx=all_idx,
            mode=mode_i,
            n_repeats=total_repeats,
            pool_size=pool_size,
            seed=arm_seed,
            cutoff=cutoff,
            n_jobs=n_jobs,
        )
        ratio_all = flags_all.mean(axis=0)
        n0, n1, n2 = sizes
        ratio_main = ratio_all[:n0]
        ratio_ff = ratio_all[n0 : n0 + n1]
        ratio_em = ratio_all[n0 + n1 : n0 + n1 + n2]
        flags_main = flags_all[:, :n0]

        np.savez_compressed(
            dest / "flags.npz",
            flags=flags_main.astype(np.uint8),
            eval_idx=eval_idx,
            ff_lt_1_idx=ff_idx,
            emergency_idx=em_idx,
        )
        df_main = write_ratio_tsv(
            ctx, eval_idx, ratio_main, dest / "signal_ratio.tsv", cohort="main"
        )
        df_ff = write_ratio_tsv(
            ctx, ff_idx, ratio_ff, dest / "signal_ratio_ff_lt_1.tsv",
            labels=ff_labels, cohort="ff_lt_1",
        )
        df_em = write_ratio_tsv(
            ctx, em_idx, ratio_em, dest / "signal_ratio_emergency.tsv",
            labels=em_labels, cohort="emergency",
        )
        df_all = pd.concat([df_main, df_ff, df_em], ignore_index=True)
        write_interactive_ratio_html(
            df_all,
            dest / "ratio.html",
            title=f"0817 {arm} · {MODE_TITLE[mode_i]} · pool_size={pool_size}",
            n_repeats=total_repeats,
            pool_n=int(pool_idx.size),
            arm=arm,
        )
        df_met = _filter_blacklist_df(df_main)
        met = _metrics_for_labels(
            df_met["label"].to_numpy(),
            (df_met["signal_ratio"].to_numpy() >= 0.5),
        )
        (dest / "metrics.json").write_text(
            json.dumps({"main_ratio_cut_0.5": met}, indent=2) + "\n"
        )
    else:
        params = score_fixed_params(
            arrays=ctx["arrays"],
            pool_idx=pool_idx,
            mode=mode_i,
            n_repeats=total_repeats,
            pool_size=pool_size,
            seed=arm_seed,
            n_jobs=n_jobs,
        )
        np.savez_compressed(dest / "params.npz", **params)

        ez_main = fixed_ez_profiles_fully_fixed(ctx["arrays"], eval_idx, params)
        ez_ff = (
            fixed_ez_profiles_fully_fixed(ctx["arrays"], ff_idx, params)
            if ff_idx.size
            else np.zeros((len(CHR_LIST), 0), dtype=np.float64)
        )
        ez_em = (
            fixed_ez_profiles_fully_fixed(ctx["arrays"], em_idx, params)
            if em_idx.size
            else np.zeros((len(CHR_LIST), 0), dtype=np.float64)
        )
        main_labels = ctx["label_arr"][eval_idx]
        write_profiles_tsv(ctx, eval_idx, ez_main, dest / "profiles.tsv")
        write_profiles_tsv(
            ctx, ff_idx, ez_ff, dest / "profiles_ff_lt_1.tsv", labels=ff_labels
        )
        write_profiles_tsv(
            ctx, em_idx, ez_em, dest / "profiles_emergency.tsv", labels=em_labels
        )

        all_met = write_fixed_panel(
            ctx=ctx,
            out=dest / "fixed.png",
            cutoff=cutoff,
            suptitle=f"0817 {arm} · {MODE_TITLE[mode_i]} · pool_size={pool_size} · ez>{cutoff:g}",
            main_idx=eval_idx,
            main_ez=ez_main,
            main_labels=main_labels,
            ff_idx=ff_idx,
            ff_ez=ez_ff,
            ff_labels=ff_labels,
            em_idx=em_idx,
            em_ez=ez_em,
            em_labels=em_labels,
        )
        call = np.nanmax(ez_main, axis=0) > cutoff
        keep = np.array([ctx["ordered"][int(i)] not in BLACKLIST for i in eval_idx], dtype=bool)
        met = _metrics_for_labels(main_labels[keep], call[keep])
        payload = {"overall_main": met, **{f"panel_{k}": v for k, v in all_met.items()}}
        (dest / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        console.print(
            f"  overall Sens={met['sens']:.3f} Spec={met['spec']:.3f} PPV={met['ppv']:.3f}"
        )

    meta_out = {
        "arm": arm,
        "mode": mode_i,
        "n_pool": int(pool_idx.size),
        "n_eval": int(eval_idx.size),
        "n_ff_lt_1": int(ff_idx.size),
        "n_emergency": int(em_idx.size),
        "total_repeats": total_repeats,
        "pool_size": pool_size,
        "ez_cutoff": cutoff,
        "seed": arm_seed,
    }
    (dest / "run.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    write_index(out)


@cli.command("index")
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
def index_cmd(output_dir: str) -> None:
    write_index(Path(output_dir))


@cli.command("replot")
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--toxic", default=str(DEFAULT_TOXIC), type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--cutoff", default=DEFAULT_CUTOFF, show_default=True, type=float)
@click.option(
    "--arm",
    multiple=True,
    type=click.Choice(["raw", "clean", "ref60"]),
    default=None,
    help="Repeatable; default all arms",
)
def replot(
    parquet: str,
    meta: str,
    toxic: str,
    output_dir: str,
    cutoff: float,
    arm: tuple[str, ...] | None,
) -> None:
    """Regenerate ratio.html / fixed.png from existing TSVs (blacklist applied silently)."""
    out = Path(output_dir)
    arms = arm if arm else ("raw", "clean", "ref60")
    ctx = build_universe(Path(parquet), Path(meta), Path(toxic))
    for a in arms:
        for mode_i in (1, 2, 3, 4):
            dest = mode_dir(out, a, mode_i)
            run_path = dest / "run.json"
            if not run_path.is_file():
                console.print(f"[yellow]skip[/yellow] missing {run_path}")
                continue
            run = json.loads(run_path.read_text())
            n_repeats = int(run.get("total_repeats", DEFAULT_REPEATS))
            pool_n = int(run.get("n_pool", default_pool_size(a)))
            pool_size = int(run.get("pool_size", default_pool_size(a)))
            title = f"0817 {a} · {MODE_TITLE[mode_i]} · pool_size={pool_size}"
            console.rule(f"[cyan]replot {a} mode{mode_i}")

            if mode_i in (1, 3):
                parts = []
                for name in ("signal_ratio.tsv", "signal_ratio_ff_lt_1.tsv", "signal_ratio_emergency.tsv"):
                    p = dest / name
                    if p.is_file():
                        parts.append(pd.read_csv(p, sep="\t"))
                if not parts:
                    console.print(f"[yellow]skip[/yellow] no ratio TSVs in {dest}")
                    continue
                df_all = pd.concat(parts, ignore_index=True)
                write_interactive_ratio_html(
                    df_all,
                    dest / "ratio.html",
                    title=title,
                    n_repeats=n_repeats,
                    pool_n=pool_n,
                    arm=a,
                )
            else:
                main_p = dest / "profiles.tsv"
                if not main_p.is_file() or not (dest / "params.npz").is_file():
                    console.print(f"[yellow]skip[/yellow] missing profiles/params in {dest}")
                    continue
                # Rebuild indices/ez from saved profiles (order matches sample column).
                def _load_profiles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    if not path.is_file():
                        return (
                            np.zeros(0, dtype=np.int64),
                            np.zeros((len(CHR_LIST), 0), dtype=np.float64),
                            np.array([], dtype=object),
                        )
                    wide = pd.read_csv(path, sep="\t")
                    wide = _filter_blacklist_df(wide)
                    idx = _samples_to_idx(wide["sample"].astype(str).tolist(), ctx["sample_index"])
                    labs = wide["label"].astype(str).to_numpy()
                    ez = np.vstack([wide[c].to_numpy(dtype=float) for c in CHR_LIST])
                    return idx, ez, labs

                main_idx, main_ez, main_labels = _load_profiles(dest / "profiles.tsv")
                ff_idx, ff_ez, ff_labels = _load_profiles(dest / "profiles_ff_lt_1.tsv")
                em_idx, em_ez, em_labels = _load_profiles(dest / "profiles_emergency.tsv")
                # profiles already blacklist-filtered; write_fixed_panel filters again (no-op).
                write_fixed_panel(
                    ctx=ctx,
                    out=dest / "fixed.png",
                    cutoff=cutoff,
                    suptitle=f"{title} · ez>{cutoff:g}",
                    main_idx=main_idx,
                    main_ez=main_ez,
                    main_labels=main_labels,
                    ff_idx=ff_idx,
                    ff_ez=ff_ez,
                    ff_labels=ff_labels,
                    em_idx=em_idx,
                    em_ez=em_ez,
                    em_labels=em_labels,
                )
    write_index(out)


if __name__ == "__main__":
    cli()
