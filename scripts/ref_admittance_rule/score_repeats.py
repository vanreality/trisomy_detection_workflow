#!/usr/bin/env python3
"""Replay 40+40 ref_free draws and store compact FP/FN + membership.

Same seed / combo / eval mask as the 20260810 ``fixed_flags_ez45`` run, but
writes per-repeat membership and set-level summaries instead of full flag
matrices.

Outputs under ``--output-dir``:
  run_config.json
  pool_samples.tsv
  repeats_{start}_{end}.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from common import (
    DEFAULT_BLACKLIST,
    DEFAULT_CUTOFF,
    DEFAULT_EP_RECALL,
    DEFAULT_EP_THRESHOLD,
    DEFAULT_EZ_CUTOFF,
    DEFAULT_FF_MIN,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUT_BASE,
    DEFAULT_REF_N,
    DEFAULT_SEED,
    DEFAULT_Z_RECALL,
    DEFAULT_Z_THRESHOLD,
    _generate_half_partitions,
    load_universe,
    parse_sample_list,
    score_one_repeat,
)

console = Console()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", default=str(DEFAULT_INPUT_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--total-repeats", default=100_000, show_default=True, type=int)
@click.option("--repeat-start", default=0, show_default=True, type=int)
@click.option("--repeat-end", default=None, type=int)
@click.option("--ref-n", default=DEFAULT_REF_N, show_default=True, type=int)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option("--ez-cutoff", default=DEFAULT_EZ_CUTOFF, show_default=True, type=float)
@click.option("--cutoff", default=DEFAULT_CUTOFF, show_default=True, type=float)
@click.option("--ep-threshold", default=DEFAULT_EP_THRESHOLD, show_default=True, type=float)
@click.option("--ep-recall", default=DEFAULT_EP_RECALL, show_default=True, type=float)
@click.option("--z-threshold", default=DEFAULT_Z_THRESHOLD, show_default=True, type=float)
@click.option("--z-recall", default=DEFAULT_Z_RECALL, show_default=True, type=float)
@click.option("--ff-min", default=DEFAULT_FF_MIN, show_default=True, type=float)
@click.option("--blacklist", default=",".join(DEFAULT_BLACKLIST), show_default=True)
@click.option(
    "--pool-samples",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional TSV/TXT of sample IDs restricting / defining the pool.",
)
@click.option(
    "--pool-source",
    default="dev_normal",
    type=click.Choice(["dev_normal", "listed"]),
    show_default=True,
    help="dev_normal: subset the 96 dev Normals. listed: pool_samples is the pool.",
)
@click.option(
    "--exclude-eval-samples",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Samples dropped from eval (keep admitted/random redraws on the same eval).",
)
@click.option(
    "--eval-samples",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Exact eval whitelist (shared across pools). Overrides exclude-eval-samples.",
)
@click.option(
    "--extra-input-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Optional extra meta/ep/z grids (new sample IDs only), e.g. 20260811 extras.",
)
@click.option("--tag", default="baseline96", show_default=True, help="Subdir name under output-dir.")
def main(
    input_dir: str,
    output_dir: str | None,
    total_repeats: int,
    repeat_start: int,
    repeat_end: int | None,
    ref_n: int,
    seed: int,
    ez_cutoff: float,
    cutoff: float,
    ep_threshold: float,
    ep_recall: float,
    z_threshold: float,
    z_recall: float,
    ff_min: float,
    blacklist: str,
    pool_samples: str | None,
    pool_source: str,
    exclude_eval_samples: str | None,
    eval_samples: str | None,
    extra_input_dir: str | None,
    tag: str,
) -> None:
    del cutoff  # episcore/zscore cutoff unused; ez_cutoff drives FP/FN
    out_root = Path(output_dir) if output_dir else DEFAULT_OUT_BASE
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    if repeat_end is None:
        repeat_end = total_repeats
    if repeat_start < 0 or repeat_end > total_repeats or repeat_end <= repeat_start:
        raise click.ClickException(
            f"Repeat slice [{repeat_start}, {repeat_end}) invalid for total={total_repeats}"
        )

    bl = [s.strip() for s in blacklist.split(",") if s.strip()]
    pool_list = parse_sample_list(Path(pool_samples)) if pool_samples else None
    excl_list = (
        parse_sample_list(Path(exclude_eval_samples)) if exclude_eval_samples else None
    )
    eval_list = parse_sample_list(Path(eval_samples)) if eval_samples else None

    console.rule("[bold blue]score_repeats 40+40")
    ctx = load_universe(
        Path(input_dir),
        ep_threshold=ep_threshold,
        ep_recall=ep_recall,
        z_threshold=z_threshold,
        z_recall=z_recall,
        blacklist=bl,
        ff_min=ff_min,
        pool_samples=pool_list,
        pool_source=pool_source,
        exclude_eval_samples=None if eval_list else excl_list,
        eval_samples=eval_list,
        extra_input_dir=Path(extra_input_dir) if extra_input_dir else None,
    )
    n_pool = int(ctx["ref_pool_idx"].size)
    if n_pool < 2 * ref_n:
        raise click.ClickException(f"Need >= {2 * ref_n} pool samples, found {n_pool}")

    pool_names = [ctx["universe"][i] for i in ctx["ref_pool_idx"]]
    n_shard = repeat_end - repeat_start
    console.print(
        f"  pool={n_pool} eval_keep={ctx['eval_keep_idx'].size} "
        f"N={int((~ctx['y_keep']).sum())} T={int(ctx['y_keep'].sum())}"
    )
    console.print(f"  repeats [{repeat_start}, {repeat_end}) / {total_repeats} seed={seed}")

    if repeat_start == 0:
        pd.DataFrame(
            {
                "pool_index": np.arange(n_pool),
                "sample": pool_names,
                "ff_before_mq": ctx["ff_arr"][ctx["ref_pool_idx"]],
                "label": ctx["label_arr"][ctx["ref_pool_idx"]],
                "set": ctx["set_arr"][ctx["ref_pool_idx"]],
            }
        ).to_csv(out / "pool_samples.tsv", sep="\t", index=False)
        cfg = {
            "tag": tag,
            "total_repeats": total_repeats,
            "ref_n": ref_n,
            "seed": seed,
            "ez_cutoff": ez_cutoff,
            "ep_threshold": ep_threshold,
            "ep_recall": ep_recall,
            "z_threshold": z_threshold,
            "z_recall": z_recall,
            "ff_min": ff_min,
            "blacklist": bl,
            "n_pool": n_pool,
            "n_eval_keep": int(ctx["eval_keep_idx"].size),
            "n_normal_keep": int((~ctx["y_keep"]).sum()),
            "n_trisomy_keep": int(ctx["y_keep"].sum()),
            "pool_samples_file": pool_samples,
            "pool_source": pool_source,
            "exclude_eval_samples_file": exclude_eval_samples,
            "eval_samples_file": eval_samples,
            "n_eval_listed": len(eval_list) if eval_list else 0,
            "extra_input_dir": extra_input_dir,
            "n_exclude_eval": len(excl_list) if excl_list else 0,
            "input_dir": str(input_dir),
        }
        (out / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    rng = np.random.default_rng(seed)
    ref_draws, ez_draws = _generate_half_partitions(
        pool_size=n_pool, half=ref_n, n_repeats=total_repeats, rng=rng
    )

    n_chr = ctx["z_array"].shape[0]
    repeat_id = np.arange(repeat_start, repeat_end, dtype=np.int32)
    fp = np.zeros(n_shard, dtype=np.int16)
    fn = np.zeros(n_shard, dtype=np.int16)
    mem_epz = np.zeros((n_shard, n_pool), dtype=np.uint8)
    mem_ez = np.zeros((n_shard, n_pool), dtype=np.uint8)
    ff_stats = {k: np.zeros(n_shard, dtype=np.float32) for k in (
        "ff_epz_mean", "ff_epz_std", "ff_epz_min", "ff_epz_max",
        "ff_ez_mean", "ff_ez_std", "ff_ez_min", "ff_ez_max",
        "ff_80_mean", "ff_80_std", "ff_80_min", "ff_80_max",
    )}
    ez_mu = np.zeros((n_shard, n_chr), dtype=np.float32)
    ez_sd = np.zeros((n_shard, n_chr), dtype=np.float32)
    pct_mu = np.zeros((n_shard, n_chr), dtype=np.float32)
    pct_sd = np.zeros((n_shard, n_chr), dtype=np.float32)

    for local_i, rid in enumerate(range(repeat_start, repeat_end)):
        rec = score_one_repeat(
            ctx, ref_draws[rid], ez_draws[rid], ez_cutoff=ez_cutoff
        )
        fp[local_i] = rec["fp"]
        fn[local_i] = rec["fn"]
        mem_epz[local_i, ref_draws[rid]] = 1
        mem_ez[local_i, ez_draws[rid]] = 1
        for prefix, stats in (
            ("ff_epz", rec["ff_epz"]),
            ("ff_ez", rec["ff_ez"]),
            ("ff_80", rec["ff_80"]),
        ):
            ff_stats[f"{prefix}_mean"][local_i] = stats[0]
            ff_stats[f"{prefix}_std"][local_i] = stats[1]
            ff_stats[f"{prefix}_min"][local_i] = stats[2]
            ff_stats[f"{prefix}_max"][local_i] = stats[3]
        ez_mu[local_i] = rec["ez_mu"]
        ez_sd[local_i] = rec["ez_sd"]
        pct_mu[local_i] = rec["pct_mu"]
        pct_sd[local_i] = rec["pct_sd"]
        if (local_i + 1) % 200 == 0 or local_i + 1 == n_shard:
            console.print(f"  completed {rid + 1}/{repeat_end}")

    tot = fp.astype(np.int16) + fn.astype(np.int16)
    dest = out / f"repeats_{repeat_start}_{repeat_end}.npz"
    np.savez_compressed(
        dest,
        repeat_id=repeat_id,
        fp=fp,
        fn=fn,
        fp_plus_fn=tot,
        mem_epz=mem_epz,
        mem_ez=mem_ez,
        ez_mu=ez_mu,
        ez_sd=ez_sd,
        pct_mu=pct_mu,
        pct_sd=pct_sd,
        **ff_stats,
    )
    console.print(
        f"[green]Done[/green] n={n_shard} perfect={(tot == 0).mean():.4f} "
        f"mean FP+FN={tot.mean():.3f} -> {dest}"
    )


if __name__ == "__main__":
    main()
