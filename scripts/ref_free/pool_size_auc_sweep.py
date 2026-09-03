#!/usr/bin/env python3
"""Pool-size sweep for reference-free fixed / filtered modes.

For each even ``pool_size`` in [20, 160] step 2:
  * ``ref_n = pool_size // 2`` for episcore/zscore refs and the same for ez refs
  * Candidate Normal pool = all 96 dev Normals; if ``pool_size > 96``, fill with
    randomly chosen test Normals (seeded). Fillers are excluded from eval.
  * ``--fixed-candidate-size N``: build the candidate of size N once (same filler
    rule); every ``pool_size`` draws ``pool_size`` refs from that fixed set.
    ``--exclude-candidate`` drops the whole candidate from eval (not just fillers).
  * Each repeat draws ``pool_size`` Normals from the candidate pool and splits
    them evenly into epi/z vs ez reference groups.
  * After ``total-repeats``, compute signal-ratio ROC-AUC (ff≥ff_min).

Writes per-pool outputs under ``{output_base}/{mode}/pool_{P}/`` and appends a
row to ``{output_base}/{mode}/pool_size_auc.tsv``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
if str(REF40_DIR) not in sys.path:
    sys.path.insert(0, str(REF40_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_search_ref40 import (  # noqa: E402
    CHR_LIST,
    _build_dense,
    compute_episcore,
    compute_zscore,
)
from ref_free_ezscore import (  # noqa: E402
    _accumulate_combo_flags,
    _accumulate_ez_pairs_multi,
    _build_ez_pairs,
    _compute_ezscore,
    _filter_combo_df,
    _flag_abnormal,
    _flag_abnormal_multi,
    _generate_half_partitions,
    _load_fixed_combo_arrays,
)
from separation import is_trisomy_label, separation_index  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()

DEFAULT_POOL_SIZES = ",".join(str(p) for p in range(20, 161, 2))
DEFAULT_BLACKLIST = (
    "PTAY0577P9S1",
    "PTAY0599P8S1",
    "PTAY0666P7S1",
    "PTAY0682P7S1",
    "PTAY0689P8H1",
)
Combo = Tuple[float, float]
EzPair = Tuple[int, int]

# Process-pool worker context (fork CoW on Linux; set via initializer).
_WORKER: Dict[str, object] = {}


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
    return max(1, int(os.cpu_count() or 1))


def _init_repeat_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)
    # Pre-expand fixed-combo arrays once per worker (avoids per-repeat expand_dims).
    if _WORKER.get("use_fixed"):
        ep_arrays = _WORKER["ep_arrays"]
        z_array = _WORKER["z_array_or_all"]
        _WORKER["ep_hypo"] = np.expand_dims(ep_arrays[0], 0)
        _WORKER["ep_hyper"] = np.expand_dims(ep_arrays[1], 0)
        _WORKER["ep_hypo_cnt"] = np.expand_dims(ep_arrays[2], 0)
        _WORKER["ep_hyper_cnt"] = np.expand_dims(ep_arrays[3], 0)
        _WORKER["z_pct"] = np.expand_dims(z_array, 0)


def _run_repeat_chunk(span: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate abnormality flags for repeats ``[start, end)``."""
    start, end = span
    candidate = _WORKER["candidate"]
    ref_draws = _WORKER["ref_draws"]
    ez_draws = _WORKER["ez_draws"]
    eval_idx = _WORKER["eval_idx"]
    cutoff = _WORKER["cutoff"]
    ez_cutoff = _WORKER["ez_cutoff"]
    use_fixed = _WORKER["use_fixed"]
    ez_pairs = _WORKER["ez_pairs"]

    n_eval = int(eval_idx.size)
    ep_counts = np.zeros(n_eval, dtype=np.int64)
    z_counts = np.zeros(n_eval, dtype=np.int64)
    ez_counts = np.zeros(n_eval, dtype=np.int64)

    if use_fixed:
        ep_hypo = _WORKER["ep_hypo"]
        ep_hyper = _WORKER["ep_hyper"]
        ep_hypo_cnt = _WORKER["ep_hypo_cnt"]
        ep_hyper_cnt = _WORKER["ep_hyper_cnt"]
        z_pct = _WORKER["z_pct"]
        for repeat_index in range(start, end):
            ref_idx = candidate[ref_draws[repeat_index]]
            ez_ref_idx = candidate[ez_draws[repeat_index]]
            episcore = compute_episcore(
                ep_hypo, ep_hyper, ep_hypo_cnt, ep_hyper_cnt, ref_idx
            )[0]
            zscore = compute_zscore(z_pct, ref_idx)[0]
            ep_counts += _flag_abnormal(episcore, eval_idx, cutoff)
            z_counts += _flag_abnormal(zscore, eval_idx, cutoff)
            ez = _compute_ezscore(episcore, zscore, ez_ref_idx)
            ez_counts += _flag_abnormal(ez, eval_idx, ez_cutoff)
    else:
        ep_arrays = _WORKER["ep_arrays"]
        z_array_or_all = _WORKER["z_array_or_all"]
        for repeat_index in range(start, end):
            ref_idx = candidate[ref_draws[repeat_index]]
            ez_ref_idx = candidate[ez_draws[repeat_index]]
            episcore_all = compute_episcore(
                ep_arrays[0], ep_arrays[1], ep_arrays[2], ep_arrays[3], ref_idx
            )
            zscore_all = compute_zscore(z_array_or_all, ref_idx)
            ep_counts += _accumulate_combo_flags(episcore_all, eval_idx, cutoff)
            z_counts += _accumulate_combo_flags(zscore_all, eval_idx, cutoff)
            ez_step = _accumulate_ez_pairs_multi(
                episcore_all,
                zscore_all,
                eval_idx,
                ez_ref_idx,
                [ez_cutoff],
                ez_pairs,
            )
            ez_counts += ez_step[0]
    return ep_counts, z_counts, ez_counts


def _chunk_spans(n: int, n_jobs: int) -> List[Tuple[int, int]]:
    n_jobs = max(1, min(int(n_jobs), int(n)))
    base, rem = divmod(n, n_jobs)
    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_jobs):
        end = start + base + (1 if i < rem else 0)
        if end > start:
            spans.append((start, end))
        start = end
    return spans


def _parse_pool_sizes(text: str) -> List[int]:
    sizes = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    for p in sizes:
        if p < 2 or p % 2 != 0:
            raise click.ClickException(f"pool_size must be even and >=2, got {p}")
    return sizes


def _build_candidate_pool(
    set_arr: np.ndarray,
    label_arr: np.ndarray,
    pool_size: int,
    fill_seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (candidate_idx, filler_idx, notes).

    When filling beyond 96, randomly permute test Normals once (``fill_seed``)
    and take a nested prefix ``[:n_fill]`` so larger pools contain smaller ones.
    """
    is_normal = label_arr == "Normal"
    dev_idx = np.flatnonzero((set_arr == "dev") & is_normal)
    test_idx = np.flatnonzero((set_arr == "test") & is_normal)
    notes: List[str] = []
    if pool_size <= dev_idx.size:
        notes.append(f"candidate=all {dev_idx.size} dev Normal; draw {pool_size}/repeat")
        return dev_idx.astype(np.int64), np.asarray([], dtype=np.int64), notes
    n_fill = pool_size - int(dev_idx.size)
    if n_fill > test_idx.size:
        raise click.ClickException(
            f"Need {n_fill} test Normal fillers for pool_size={pool_size}, "
            f"found only {test_idx.size}"
        )
    rng = np.random.default_rng(fill_seed)
    ordered = test_idx[rng.permutation(test_idx.size)].astype(np.int64, copy=False)
    fillers = np.sort(ordered[:n_fill])
    candidate = np.concatenate([dev_idx.astype(np.int64), fillers])
    notes.append(
        f"candidate={dev_idx.size} dev + {n_fill} nested random test Normal "
        f"fillers (fill_seed={fill_seed}); use all {pool_size}/repeat"
    )
    return candidate, fillers, notes


def _run_one_pool(
    *,
    pool_size: int,
    total_repeats: int,
    seed: int,
    fill_seed: int,
    cutoff: float,
    ez_cutoff: float,
    ff_min: float,
    combo_mode: str,
    ep_arrays,
    z_array_or_all,
    ep_combos: List[Combo],
    z_combos: List[Combo],
    ez_pairs: List[EzPair],
    set_arr: np.ndarray,
    label_arr: np.ndarray,
    ff_arr: np.ndarray,
    universe: List[str],
    use_fixed: bool,
    blacklist: Sequence[str],
    n_jobs: int = 1,
    fixed_candidate_size: Optional[int] = None,
    exclude_candidate: bool = False,
) -> dict:
    half = pool_size // 2
    cand_n = int(fixed_candidate_size) if fixed_candidate_size else int(pool_size)
    if cand_n < pool_size:
        raise click.ClickException(
            f"fixed_candidate_size={cand_n} must be >= pool_size={pool_size}"
        )
    candidate, fillers, notes = _build_candidate_pool(
        set_arr, label_arr, cand_n, fill_seed
    )
    if fixed_candidate_size:
        notes.append(
            f"draw {pool_size}/repeat from fixed candidate {cand_n} "
            f"(ref {half}+{half})"
        )
    is_trisomy = np.array([bool(re.match(r"^T\d", s)) for s in label_arr])
    is_normal = label_arr == "Normal"
    is_dev_trisomy = (set_arr == "dev") & is_trisomy
    is_test = set_arr == "test"
    # Eval = dev trisomy + test. Default: drop fillers only. With
    # --exclude-candidate, drop the entire candidate (e.g. the fixed 160).
    # Blacklist samples stay in the TSV when they are not in the dropped set.
    drop_mask = np.zeros(len(universe), dtype=bool)
    if exclude_candidate:
        drop_mask[candidate] = True
        notes.append("eval excludes entire candidate pool")
    else:
        drop_mask[fillers] = True
    eval_mask = (is_dev_trisomy | is_test) & ~drop_mask
    eval_idx = np.flatnonzero(eval_mask)
    if eval_idx.size == 0:
        raise click.ClickException(
            f"pool_size={pool_size}: empty eval after candidate/filler exclusion"
        )

    rng = np.random.default_rng(seed)
    ref_draws, ez_draws = _generate_half_partitions(
        pool_size=candidate.size,
        half=half,
        n_repeats=total_repeats,
        rng=rng,
    )

    n_eval = eval_idx.size
    workers = _resolve_n_jobs(n_jobs)
    spans = _chunk_spans(total_repeats, workers)
    console.print(
        f"  parallel: n_jobs={workers} chunks={len(spans)} "
        f"repeats={total_repeats}"
    )

    payload = {
        "candidate": candidate,
        "ref_draws": ref_draws,
        "ez_draws": ez_draws,
        "eval_idx": eval_idx,
        "cutoff": cutoff,
        "ez_cutoff": ez_cutoff,
        "use_fixed": use_fixed,
        "ep_arrays": ep_arrays,
        "z_array_or_all": z_array_or_all,
        "ez_pairs": ez_pairs,
    }

    ep_counts = np.zeros(n_eval, dtype=np.int64)
    z_counts = np.zeros(n_eval, dtype=np.int64)
    ez_counts = np.zeros(n_eval, dtype=np.int64)

    if workers == 1 or len(spans) == 1:
        _init_repeat_worker(payload)
        ep_counts, z_counts, ez_counts = _run_repeat_chunk((0, total_repeats))
        console.print(f"  pool={pool_size} repeat {total_repeats}/{total_repeats}")
    else:
        done = 0
        # fork: share large draw/score arrays via CoW instead of pickling per worker.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_repeat_worker,
            initargs=(payload,),
        ) as pool:
            futures = {pool.submit(_run_repeat_chunk, span): span for span in spans}
            for fut in as_completed(futures):
                ep_c, z_c, ez_c = fut.result()
                ep_counts += ep_c
                z_counts += z_c
                ez_counts += ez_c
                start, end = futures[fut]
                done += end - start
                console.print(
                    f"  pool={pool_size} repeat {done}/{total_repeats} "
                    f"(chunk {start}:{end})"
                )

    n_ep = len(ep_combos)
    n_z = len(z_combos)
    n_ez = len(ez_pairs)
    result = pd.DataFrame(
        {
            "sample": [universe[i] for i in eval_idx],
            "set": set_arr[eval_idx],
            "label": label_arr[eval_idx],
            "ff_before_mq": ff_arr[eval_idx],
            "episcore_signal_ratio": ep_counts / float(n_ep * total_repeats),
            "zscore_signal_ratio": z_counts / float(n_z * total_repeats),
            "ezscore_signal_ratio": ez_counts / float(n_ez * total_repeats),
        }
    )
    # AUC/separation ignore analysis blacklist (still present in result TSV for plots)
    bl_set = {str(s) for s in blacklist}
    result_scored = result[~result["sample"].astype(str).isin(bl_set)].copy()
    sep = {
        name: separation_index(result_scored, col, ff_min=ff_min)
        for name, col in [
            ("episcore", "episcore_signal_ratio"),
            ("zscore", "zscore_signal_ratio"),
            ("ezscore", "ezscore_signal_ratio"),
        ]
    }
    filler_samples = [universe[i] for i in fillers]
    row = {
        "pool_size": pool_size,
        "ref_n": half,
        "ez_ref_n": half,
        "candidate_pool_size": int(candidate.size),
        "n_fillers": int(fillers.size),
        "total_repeats": total_repeats,
        "ez_cutoff": ez_cutoff,
        "ff_min": ff_min,
        "n_eval": int(n_eval),
        "n_ep_combos": n_ep,
        "n_z_combos": n_z,
        "n_ez_combos": n_ez,
        "auc_episcore": sep["episcore"]["sep_auc"],
        "auc_zscore": sep["zscore"]["sep_auc"],
        "auc_ezscore": sep["ezscore"]["sep_auc"],
        "youden_ezscore": sep["ezscore"]["sep_youden"],
        "n_normal_auc": sep["ezscore"]["n_normal"],
        "n_trisomy_auc": sep["ezscore"]["n_trisomy"],
        "notes": "; ".join(notes),
        "fixed_candidate_size": int(fixed_candidate_size) if fixed_candidate_size else None,
        "exclude_candidate": bool(exclude_candidate),
    }
    return {
        "row": row,
        "result": result,
        "sep": sep,
        "filler_samples": filler_samples,
        "candidate_samples": [universe[i] for i in candidate],
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-base", required=True, type=click.Path(file_okay=False))
@click.option("--combo-mode", required=True, type=click.Choice(["fixed", "all"]))
@click.option("--pool-sizes", default=DEFAULT_POOL_SIZES, show_default=True)
@click.option("--pool-size", default=None, type=int, help="Run a single pool size (SLURM array)")
@click.option("--total-repeats", default=5000, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--fill-seed", default=7, show_default=True, type=int,
              help="Seed for choosing test-Normal fillers when pool_size>96")
@click.option("--cutoff", default=3.0, show_default=True, type=float)
@click.option("--ez-cutoff", default=None, type=float,
              help="Default: 4.5 for fixed, 3.0 for filtered")
@click.option("--ff-min", default=0.01, show_default=True, type=float)
@click.option("--ep-threshold", default=0.5, show_default=True, type=float)
@click.option("--ep-recall", default=0.65, show_default=True, type=float)
@click.option("--z-threshold", default=0.85, show_default=True, type=float)
@click.option("--z-recall", default=0.95, show_default=True, type=float)
@click.option("--ep-threshold-min", default=0.1, show_default=True, type=float)
@click.option("--ep-threshold-max", default=0.5, show_default=True, type=float)
@click.option("--ep-recall-min", default=0.5, show_default=True, type=float)
@click.option("--ep-recall-max", default=0.75, show_default=True, type=float)
@click.option("--z-threshold-min", default=0.8, show_default=True, type=float)
@click.option("--z-threshold-max", default=0.95, show_default=True, type=float)
@click.option("--z-recall-min", default=0.9, show_default=True, type=float)
@click.option("--z-recall-max", default=0.99, show_default=True, type=float)
@click.option(
    "--blacklist",
    default=",".join(DEFAULT_BLACKLIST),
    show_default=True,
    help="Comma-separated samples excluded from eval/AUC",
)
@click.option(
    "--n-jobs",
    default=0,
    show_default=True,
    type=int,
    help="Worker processes for repeat loop (0 → SLURM_CPUS_PER_TASK / N_JOBS / cpu_count)",
)
@click.option(
    "--fixed-candidate-size",
    default=None,
    type=int,
    help="Build this candidate size once; every pool_size draws from it (e.g. 160)",
)
@click.option(
    "--exclude-candidate",
    is_flag=True,
    default=False,
    help="Drop the entire candidate pool from eval (default: drop fillers only)",
)
def main(
    input_dir: str,
    output_base: str,
    combo_mode: str,
    pool_sizes: str,
    pool_size: Optional[int],
    total_repeats: int,
    seed: int,
    fill_seed: int,
    cutoff: float,
    ez_cutoff: Optional[float],
    ff_min: float,
    ep_threshold: float,
    ep_recall: float,
    z_threshold: float,
    z_recall: float,
    ep_threshold_min: float,
    ep_threshold_max: float,
    ep_recall_min: float,
    ep_recall_max: float,
    z_threshold_min: float,
    z_threshold_max: float,
    z_recall_min: float,
    z_recall_max: float,
    blacklist: str,
    n_jobs: int,
    fixed_candidate_size: Optional[int],
    exclude_candidate: bool,
) -> None:
    sizes = _parse_pool_sizes(pool_sizes)
    if pool_size is not None:
        if pool_size not in sizes:
            sizes = _parse_pool_sizes(str(pool_size))
        else:
            sizes = [pool_size]
    use_fixed = combo_mode == "fixed"
    if ez_cutoff is None:
        ez_cutoff = 4.5 if use_fixed else 3.0
    bl = tuple(s.strip() for s in blacklist.split(",") if s.strip())

    input_path = Path(input_dir)
    mode_name = "fixed" if use_fixed else "filtered"
    out_root = Path(output_base) / mode_name
    out_root.mkdir(parents=True, exist_ok=True)

    workers = _resolve_n_jobs(n_jobs)
    console.rule(f"[bold blue]Pool-size AUC sweep ({mode_name})")
    console.print(f"  pools   : {sizes}")
    console.print(f"  repeats : {total_repeats}")
    console.print(f"  n_jobs  : {workers}")
    console.print(f"  ez cut  : {ez_cutoff:g}")
    console.print(f"  blacklist: {list(bl)}")
    if fixed_candidate_size:
        console.print(
            f"  candidate: fixed {fixed_candidate_size} "
            f"(exclude_candidate={exclude_candidate})"
        )

    meta = pd.read_csv(input_path / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    meta["ff_before_mq"] = pd.to_numeric(meta["ff_before_mq"], errors="coerce")
    ep_df = pd.read_parquet(input_path / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_path / "zscore_grid_search.parquet")

    if not use_fixed:
        ep_df = _filter_combo_df(
            ep_df, ep_threshold_min, ep_threshold_max, ep_recall_min, ep_recall_max
        )
        z_df = _filter_combo_df(
            z_df, z_threshold_min, z_threshold_max, z_recall_min, z_recall_max
        )
        if ep_df.empty or z_df.empty:
            raise click.ClickException("Combo filters removed all rows")

    ep_samples = set(ep_df["sample"].astype(str).unique())
    z_samples = set(z_df["sample"].astype(str).unique())
    universe = sorted(set(meta["sample"]) & ep_samples & z_samples)
    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}

    meta_idx = meta.set_index("sample").reindex(universe)
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    ff_arr = pd.to_numeric(meta_idx["ff_before_mq"], errors="coerce").to_numpy()

    if use_fixed:
        ep_arrays, z_array = _load_fixed_combo_arrays(
            ep_df, z_df, ep_threshold, ep_recall, z_threshold, z_recall,
            sample_index, chr_index,
        )
        ep_combos: List[Combo] = [(ep_threshold, ep_recall)]
        z_combos = [(z_threshold, z_recall)]
        ez_pairs: List[EzPair] = [(0, 0)]
        z_array_or_all = z_array
    else:
        ep_combos, ep_arrays = _build_dense(
            ep_df,
            ["hypo_z_intra", "hyper_z_intra", "hypo_cpgs_count", "hyper_cpgs_count"],
            sample_index,
            chr_index,
        )
        z_combos, z_arrays = _build_dense(z_df, ["percentage"], sample_index, chr_index)
        z_array_or_all = z_arrays[0]
        ez_pairs, ez_pair_mode = _build_ez_pairs(ep_combos, z_combos)
        console.print(f"  ez pairs: {len(ez_pairs)} ({ez_pair_mode})")

    rows = []
    for p in sizes:
        console.rule(f"[cyan]pool_size={p} (ref {p//2}+{p//2})")
        pack = _run_one_pool(
            pool_size=p,
            total_repeats=total_repeats,
            seed=seed,
            fill_seed=fill_seed,
            cutoff=cutoff,
            ez_cutoff=float(ez_cutoff),
            ff_min=ff_min,
            combo_mode=combo_mode,
            ep_arrays=ep_arrays,
            z_array_or_all=z_array_or_all,
            ep_combos=ep_combos,
            z_combos=z_combos,
            ez_pairs=ez_pairs,
            set_arr=set_arr,
            label_arr=label_arr,
            ff_arr=ff_arr,
            universe=universe,
            use_fixed=use_fixed,
            blacklist=bl,
            n_jobs=workers,
            fixed_candidate_size=fixed_candidate_size,
            exclude_candidate=exclude_candidate,
        )
        pool_dir = out_root / f"pool_{p}"
        pool_dir.mkdir(parents=True, exist_ok=True)
        pack["result"].to_csv(
            pool_dir / "abnormality_signal_ratio.tsv", sep="\t", index=False, float_format="%.6f"
        )
        if fixed_candidate_size:
            cand_path = out_root / "candidate_samples.tsv"
            if not cand_path.is_file():
                cand_df = (
                    meta.loc[meta["sample"].isin(pack["candidate_samples"]), ["sample", "set", "label"]]
                    .drop_duplicates("sample")
                    .sort_values("sample")
                )
                cand_df.to_csv(cand_path, sep="\t", index=False)
        cfg = {
            "combo_mode": combo_mode,
            "pool_size": p,
            "ref_n": p // 2,
            "total_repeats": total_repeats,
            "seed": seed,
            "fill_seed": fill_seed,
            "cutoff": cutoff,
            "ez_cutoff": float(ez_cutoff),
            "ff_min": ff_min,
            "blacklist": list(bl),
            "filler_samples": pack["filler_samples"],
            "candidate_samples": pack["candidate_samples"],
            "n_candidate": len(pack["candidate_samples"]),
            "fixed_candidate_size": fixed_candidate_size,
            "exclude_candidate": bool(exclude_candidate),
            "separation": pack["sep"],
            "row": pack["row"],
        }
        if use_fixed:
            cfg.update(
                {
                    "ep_threshold": ep_threshold,
                    "ep_recall": ep_recall,
                    "z_threshold": z_threshold,
                    "z_recall": z_recall,
                }
            )
        (pool_dir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        # Atomic per-pool row so concurrent array tasks never read a partial TSV.
        row_tmp = pool_dir / "pool_size_auc_row.tsv.tmp"
        row_path = pool_dir / "pool_size_auc_row.tsv"
        pd.DataFrame([pack["row"]]).to_csv(
            row_tmp, sep="\t", index=False, float_format="%.6f"
        )
        row_tmp.replace(row_path)
        rows.append(pack["row"])
        console.print(
            f"[green]OK[/green] pool={p} ez AUC={pack['row']['auc_ezscore']:.4f} "
            f"(N={pack['row']['n_normal_auc']} T={pack['row']['n_trisomy_auc']})"
        )

    # Array workers (single pool_size) only write per-pool rows. Do NOT merge the
    # shared summary here — concurrent .tsv.tmp replaces race on lustre and mark
    # successful tasks FAILED, which blocks afterok plot jobs.
    # plot_pool_size_auc.py merges pool_*/pool_size_auc_row.tsv.
    if len(sizes) > 1:
        summary = pd.DataFrame(rows).drop_duplicates("pool_size", keep="last").sort_values(
            "pool_size"
        )
        summary_path = out_root / "pool_size_auc.tsv"
        summary.to_csv(summary_path, sep="\t", index=False, float_format="%.6f")
        console.print(f"[green]Done[/green] summary -> {summary_path}")
    else:
        console.print(
            f"[green]Done[/green] wrote "
            f"{out_root / f'pool_{sizes[0]}' / 'pool_size_auc_row.tsv'}"
        )

    (out_root / "sweep_config.json").write_text(
        json.dumps(
            {
                "combo_mode": combo_mode,
                "mode_name": mode_name,
                "pool_sizes_requested": sizes,
                "total_repeats": total_repeats,
                "n_jobs": workers,
                "seed": seed,
                "fill_seed": fill_seed,
                "ez_cutoff": float(ez_cutoff),
                "ff_min": ff_min,
                "blacklist": list(bl),
                "fixed_candidate_size": fixed_candidate_size,
                "exclude_candidate": bool(exclude_candidate),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
