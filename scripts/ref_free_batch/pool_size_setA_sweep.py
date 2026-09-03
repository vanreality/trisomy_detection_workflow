#!/usr/bin/env python3
"""Fixed-combo pool-size sweep for Set A, dual ez cutoffs (3.0 and 4.5).

Each even pool_size draws ``pool_size`` Normals (96 dev + nested test fillers
when pool>96), splits them into epi/z vs ez halves, and accumulates abnormality
flags on Set A eval units (excluding units whose orig_sample is in the current
candidate pool).

Writes ``{output_base}/pool_{P}/abnormality_signal_ratio.tsv`` with both
``ezscore_signal_ratio_3`` and ``ezscore_signal_ratio_4.5``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
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
REF_FREE_DIR = SCRIPT_DIR.parent / "ref_free"
REF40_DIR = SCRIPT_DIR.parent / "ref_explore_plus_grid_search"
for p in (SCRIPT_DIR, REF_FREE_DIR, REF40_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import (  # noqa: E402
    DEFAULT_REPEATS,
    EP_CUTOFF,
    EZ_CUTOFFS,
    FILL_SEED,
    MODES,
    SEED,
    ez_ratio_col,
    pool_sizes as default_pool_sizes,
)
from grid_search_ref40 import CHR_LIST, compute_episcore, compute_zscore  # noqa: E402
from ref_free_ezscore import (  # noqa: E402
    _flag_abnormal,
    _flag_abnormal_multi,
    _generate_half_partitions,
    _load_fixed_combo_arrays,
    _compute_ezscore,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()

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


def _init_repeat_worker(payload: dict) -> None:
    _WORKER.clear()
    _WORKER.update(payload)
    ep_arrays = _WORKER["ep_arrays"]
    z_array = _WORKER["z_array"]
    _WORKER["ep_hypo"] = np.expand_dims(ep_arrays[0], 0)
    _WORKER["ep_hyper"] = np.expand_dims(ep_arrays[1], 0)
    _WORKER["ep_hypo_cnt"] = np.expand_dims(ep_arrays[2], 0)
    _WORKER["ep_hyper_cnt"] = np.expand_dims(ep_arrays[3], 0)
    _WORKER["z_pct"] = np.expand_dims(z_array, 0)


def _run_repeat_chunk(
    span: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, end = span
    candidate = _WORKER["candidate"]
    ref_draws = _WORKER["ref_draws"]
    ez_draws = _WORKER["ez_draws"]
    eval_idx = _WORKER["eval_idx"]
    cutoff = _WORKER["cutoff"]
    ez_cutoffs = _WORKER["ez_cutoffs"]
    ep_hypo = _WORKER["ep_hypo"]
    ep_hyper = _WORKER["ep_hyper"]
    ep_hypo_cnt = _WORKER["ep_hypo_cnt"]
    ep_hyper_cnt = _WORKER["ep_hyper_cnt"]
    z_pct = _WORKER["z_pct"]

    n_eval = int(eval_idx.size)
    n_cut = len(ez_cutoffs)
    ep_counts = np.zeros(n_eval, dtype=np.int64)
    z_counts = np.zeros(n_eval, dtype=np.int64)
    ez_counts = np.zeros((n_cut, n_eval), dtype=np.int64)

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
        ez_counts += _flag_abnormal_multi(ez, eval_idx, ez_cutoffs)
    return ep_counts, z_counts, ez_counts


def _build_candidate(
    role: np.ndarray,
    universe: Sequence[str],
    pool_size: int,
    fill_seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ref_idx = np.flatnonzero(role == "ref_pool").astype(np.int64)
    filler_idx = np.flatnonzero(role == "filler_pool").astype(np.int64)
    notes: List[str] = []
    if pool_size <= ref_idx.size:
        notes.append(f"candidate=all {ref_idx.size} ref_pool; draw {pool_size}/repeat")
        return ref_idx, np.asarray([], dtype=np.int64), notes
    n_fill = pool_size - int(ref_idx.size)
    if n_fill > filler_idx.size:
        raise click.ClickException(
            f"Need {n_fill} fillers for pool_size={pool_size}, "
            f"found only {filler_idx.size}"
        )
    rng = np.random.default_rng(fill_seed)
    ordered = filler_idx[rng.permutation(filler_idx.size)]
    fillers = np.sort(ordered[:n_fill])
    candidate = np.concatenate([ref_idx, fillers])
    notes.append(
        f"candidate={ref_idx.size} ref_pool + {n_fill} nested filler_pool "
        f"(fill_seed={fill_seed})"
    )
    return candidate, fillers, notes


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--mode", required=True, type=click.Choice(sorted(MODES)))
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-base", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--pool-sizes",
    default=",".join(str(p) for p in default_pool_sizes()),
    show_default=True,
)
@click.option("--pool-size", default=None, type=int, help="Run a single pool size (SLURM array)")
@click.option("--total-repeats", default=DEFAULT_REPEATS, show_default=True, type=int)
@click.option("--seed", default=SEED, show_default=True, type=int)
@click.option("--fill-seed", default=FILL_SEED, show_default=True, type=int)
@click.option("--cutoff", default=EP_CUTOFF, show_default=True, type=float)
@click.option("--n-jobs", default=0, show_default=True, type=int)
def main(
    mode: str,
    input_dir: Path,
    output_base: Path,
    pool_sizes: str,
    pool_size: Optional[int],
    total_repeats: int,
    seed: int,
    fill_seed: int,
    cutoff: float,
    n_jobs: int,
) -> None:
    cfg = MODES[mode]
    sizes = _parse_pool_sizes(pool_sizes)
    if pool_size is not None:
        sizes = [pool_size] if pool_size not in sizes else [pool_size]
    ez_cutoffs = list(EZ_CUTOFFS)
    workers = _resolve_n_jobs(n_jobs)
    output_base.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold blue]Set A pool sweep ({mode})")
    console.print(f"  pools   : {sizes}")
    console.print(f"  repeats : {total_repeats}")
    console.print(f"  n_jobs  : {workers}")
    console.print(f"  ez cuts : {ez_cutoffs}")

    meta = pd.read_csv(input_dir / "meta.csv").drop_duplicates("sample", keep="first")
    meta["sample"] = meta["sample"].astype(str)
    meta["orig_sample"] = meta["orig_sample"].astype(str)
    meta["role"] = meta["role"].astype(str)
    meta["ff_before_mq"] = pd.to_numeric(meta["ff_before_mq"], errors="coerce")
    meta["purity"] = pd.to_numeric(meta["purity"], errors="coerce")
    ep_df = pd.read_parquet(input_dir / "episcore_grid_search.parquet")
    z_df = pd.read_parquet(input_dir / "zscore_grid_search.parquet")

    ep_samples = set(ep_df["sample"].astype(str).unique())
    z_samples = set(z_df["sample"].astype(str).unique())
    universe = [s for s in meta["sample"] if s in ep_samples and s in z_samples]
    # Keep meta order of roles: ref, filler, eval (prepare concatenates that way),
    # but intersect can scramble — reindex meta.
    meta_idx = meta.set_index("sample").reindex(universe)
    sample_index = {s: i for i, s in enumerate(universe)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}

    role = meta_idx["role"].astype(str).to_numpy()
    orig = meta_idx["orig_sample"].astype(str).to_numpy()
    set_arr = meta_idx["set"].astype(str).to_numpy()
    label_arr = meta_idx["label"].astype(str).to_numpy()
    ff_arr = pd.to_numeric(meta_idx["ff_before_mq"], errors="coerce").to_numpy()
    pur_arr = pd.to_numeric(meta_idx["purity"], errors="coerce").to_numpy()
    unit_arr = meta_idx["unit_id"].astype(str).to_numpy() if "unit_id" in meta_idx.columns else orig

    ep_arrays, z_array = _load_fixed_combo_arrays(
        ep_df,
        z_df,
        cfg["ep_threshold"],
        cfg["ep_recall"],
        cfg["z_threshold"],
        cfg["z_recall"],
        sample_index,
        chr_index,
    )

    for p in sizes:
        console.rule(f"[cyan]pool_size={p} (ref {p // 2}+{p // 2})")
        candidate, fillers, notes = _build_candidate(role, universe, p, fill_seed)
        cand_orig = {orig[i] for i in candidate}
        eval_mask = (role == "eval") & ~np.array([o in cand_orig for o in orig])
        eval_idx = np.flatnonzero(eval_mask)
        if eval_idx.size == 0:
            raise click.ClickException(f"pool_size={p}: empty eval after candidate exclusion")

        rng = np.random.default_rng(seed)
        ref_draws, ez_draws = _generate_half_partitions(
            pool_size=int(candidate.size),
            half=p // 2,
            n_repeats=total_repeats,
            rng=rng,
        )
        # When pool_size < candidate.size (pool<=96, candidate=96), draws must
        # sample from the 96 then use only pool_size of them. half-partitions
        # require pool_size >= 2*half, so pass candidate.size and draw halves
        # of size p//2 from the full candidate (same as 20260810).
        payload = {
            "candidate": candidate,
            "ref_draws": ref_draws,
            "ez_draws": ez_draws,
            "eval_idx": eval_idx,
            "cutoff": float(cutoff),
            "ez_cutoffs": ez_cutoffs,
            "ep_arrays": ep_arrays,
            "z_array": z_array,
        }
        spans = _chunk_spans(total_repeats, workers)
        console.print(
            f"  eval={eval_idx.size} candidate={candidate.size} "
            f"fillers={fillers.size} chunks={len(spans)}"
        )

        n_eval = int(eval_idx.size)
        n_cut = len(ez_cutoffs)
        ep_counts = np.zeros(n_eval, dtype=np.int64)
        z_counts = np.zeros(n_eval, dtype=np.int64)
        ez_counts = np.zeros((n_cut, n_eval), dtype=np.int64)

        if workers == 1 or len(spans) == 1:
            _init_repeat_worker(payload)
            ep_counts, z_counts, ez_counts = _run_repeat_chunk((0, total_repeats))
        else:
            done = 0
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
                        f"  pool={p} repeat {done}/{total_repeats} (chunk {start}:{end})"
                    )

        result = pd.DataFrame(
            {
                "sample": [universe[i] for i in eval_idx],
                "orig_sample": orig[eval_idx],
                "unit_id": unit_arr[eval_idx],
                "set": set_arr[eval_idx],
                "label": label_arr[eval_idx],
                "ff_before_mq": ff_arr[eval_idx],
                "purity": pur_arr[eval_idx],
                "episcore_signal_ratio": ep_counts / float(total_repeats),
                "zscore_signal_ratio": z_counts / float(total_repeats),
            }
        )
        for ci, cut in enumerate(ez_cutoffs):
            result[ez_ratio_col(cut)] = ez_counts[ci] / float(total_repeats)

        pool_dir = output_base / f"pool_{p}"
        pool_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(
            pool_dir / "abnormality_signal_ratio.tsv",
            sep="\t",
            index=False,
            float_format="%.6f",
        )
        cfg_json = {
            "mode": mode,
            "pool_size": p,
            "ref_n": p // 2,
            "total_repeats": total_repeats,
            "seed": seed,
            "fill_seed": fill_seed,
            "cutoff": cutoff,
            "ez_cutoffs": ez_cutoffs,
            "n_eval": int(n_eval),
            "n_candidate": int(candidate.size),
            "n_fillers": int(fillers.size),
            "filler_samples": [universe[i] for i in fillers],
            "notes": notes,
            "ep_threshold": cfg["ep_threshold"],
            "ep_recall": cfg["ep_recall"],
            "z_threshold": cfg["z_threshold"],
            "z_recall": cfg["z_recall"],
        }
        (pool_dir / "run_config.json").write_text(json.dumps(cfg_json, indent=2) + "\n")
        console.print(f"[green]OK[/green] pool={p} n_eval={n_eval} -> {pool_dir}")

    (output_base / "sweep_config.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "pool_sizes": sizes,
                "total_repeats": total_repeats,
                "seed": seed,
                "fill_seed": fill_seed,
                "ez_cutoffs": ez_cutoffs,
                "input_dir": str(input_dir),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
