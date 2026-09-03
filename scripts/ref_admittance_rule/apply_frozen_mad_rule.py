#!/usr/bin/env python3
"""Apply MAD/FF screens frozen on the original 96-dev pool to test_ref_candidates.

This is the non-circular transfer: fences (per-chr median/MAD of percentage and
z_intra, plus FF 5th–95th percentiles) are estimated on the **dev** 96 only.
Test samples are scored against those fences. 40+40 FP/FN on the test pool is
never used to decide who is dropped.

Writes admitted / dropped / size-matched random lists for every screen that
still leaves ≥80 samples (needed for 40+40).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from common import (
    CHR_LIST,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUT_BASE,
    DEFAULT_REF_N,
    load_universe,
    mad_z_vs_ref,
    parse_sample_list,
)

console = Console()

RULES = (
    "pct_mad_3_5",
    "intra_mad_3_5",
    "ff_tail_5_95",
    "mad_or_ff",
    "mad_keep80",
)


def _outlier_chrs(abs_z: np.ndarray, cutoff: float) -> str:
    hits = [CHR_LIST[i] for i, v in enumerate(abs_z) if v > cutoff]
    return ",".join(hits)


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""))


def _random_keep(samples: list[str], n_drop: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    n = len(samples)
    mask = np.ones(n, dtype=bool)
    if n_drop > 0:
        drop = rng.choice(n, size=n_drop, replace=False)
        mask[drop] = False
    return [s for s, keep in zip(samples, mask) if keep]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input-dir", default=str(DEFAULT_INPUT_DIR), type=click.Path(exists=True, file_okay=False))
@click.option(
    "--check-dir",
    default=str(DEFAULT_OUT_BASE / "ref_admittance_check"),
    type=click.Path(file_okay=False),
)
@click.option(
    "--dev-pool",
    default=str(DEFAULT_OUT_BASE / "baseline96" / "pool_samples.tsv"),
    type=click.Path(exists=True, dir_okay=False),
    help="Original 96-dev pool (fences are estimated here).",
)
@click.option(
    "--test-pool",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="test_ref_candidates.txt (default: <check-dir>/test_ref_candidates.txt).",
)
@click.option("--ref-n", default=DEFAULT_REF_N, show_default=True, type=int)
@click.option("--random-seed", default=7, show_default=True, type=int)
@click.option("--mad-cutoff", default=3.5, show_default=True, type=float)
def main(
    input_dir: str,
    check_dir: str,
    dev_pool: str,
    test_pool: str | None,
    ref_n: int,
    random_seed: int,
    mad_cutoff: float,
) -> None:
    check = Path(check_dir)
    check.mkdir(parents=True, exist_ok=True)
    test_path = Path(test_pool) if test_pool else check / "test_ref_candidates.txt"
    test_ids = parse_sample_list(test_path)
    dev_ids = parse_sample_list(Path(dev_pool))
    n_keep_min = 2 * ref_n
    out = check / "mad_frozen"
    out.mkdir(parents=True, exist_ok=True)

    ctx = load_universe(Path(input_dir), pool_source="listed", pool_samples=dev_ids)
    sample_index = ctx["sample_index"]
    missing_test = [s for s in test_ids if s not in sample_index]
    if missing_test:
        raise click.ClickException(f"test samples missing from universe: {missing_test[:5]}")
    missing_dev = [s for s in dev_ids if s not in sample_index]
    if missing_dev:
        raise click.ClickException(f"dev samples missing from universe: {missing_dev[:5]}")

    dev_idx = np.array([sample_index[s] for s in dev_ids], dtype=np.int64)
    test_idx = np.array([sample_index[s] for s in test_ids], dtype=np.int64)
    hypo = ctx["ep_arrays"][0]
    hyper = ctx["ep_arrays"][1]
    pct = ctx["z_array"]
    ff = ctx["ff_arr"]

    hypo_z = mad_z_vs_ref(hypo[:, test_idx], hypo[:, dev_idx], axis=1)
    hyper_z = mad_z_vs_ref(hyper[:, test_idx], hyper[:, dev_idx], axis=1)
    pct_z = mad_z_vs_ref(pct[:, test_idx], pct[:, dev_idx], axis=1)
    intra = np.maximum(np.abs(hypo_z), np.abs(hyper_z))
    abs_pct = np.abs(pct_z)

    ff_dev = ff[dev_idx].astype(float)
    q05, q95 = np.nanquantile(ff_dev, [0.05, 0.95])
    ff_test = ff[test_idx].astype(float)

    rows = []
    for j, sample in enumerate(test_ids):
        pz = abs_pct[:, j]
        iz = intra[:, j]
        rows.append(
            {
                "sample": sample,
                "ff_before_mq": float(ff_test[j]) if np.isfinite(ff_test[j]) else float("nan"),
                "max_abs_pct_madz_vs_dev": float(np.nanmax(pz)),
                "max_abs_intra_madz_vs_dev": float(np.nanmax(iz)),
                "n_chr_pct_madz_gt_cutoff": int((pz > mad_cutoff).sum()),
                "n_chr_intra_madz_gt_cutoff": int((iz > mad_cutoff).sum()),
                "outlier_chrs_pct": _outlier_chrs(pz, mad_cutoff),
                "outlier_chrs_intra": _outlier_chrs(iz, mad_cutoff),
                "fail_pct_mad_3_5": bool((pz > mad_cutoff).any()),
                "fail_intra_mad_3_5": bool((iz > mad_cutoff).any()),
                "fail_ff_tail_5_95": bool(
                    np.isfinite(ff_test[j]) and ((ff_test[j] < q05) or (ff_test[j] > q95))
                ),
            }
        )
        for i, chr_name in enumerate(CHR_LIST):
            rows[-1][f"{chr_name}_pct_madz_vs_dev"] = float(pct_z[i, j])
            rows[-1][f"{chr_name}_hypo_madz_vs_dev"] = float(hypo_z[i, j])
            rows[-1][f"{chr_name}_hyper_madz_vs_dev"] = float(hyper_z[i, j])
    feat = pd.DataFrame(rows)
    feat["fail_mad_or_ff"] = (
        feat["fail_pct_mad_3_5"] | feat["fail_intra_mad_3_5"] | feat["fail_ff_tail_5_95"]
    )
    feat["mad_rank_score"] = feat[
        ["max_abs_pct_madz_vs_dev", "max_abs_intra_madz_vs_dev"]
    ].max(axis=1)

    toxic_path = check / "all_96_test" / "analysis" / "proof" / "dropped_samples.txt"
    toxic = set(parse_sample_list(toxic_path)) if toxic_path.is_file() else set()
    feat["dropped_by_toxic_keep80"] = feat["sample"].isin(toxic)

    feat.to_csv(out / "sample_features_vs_dev.tsv", sep="\t", index=False, float_format="%.6f")

    n_drop_keep80 = max(0, len(test_ids) - n_keep_min)
    keep80_drop = set(
        feat.nlargest(n_drop_keep80, "mad_rank_score")["sample"].astype(str)
    )
    feat["fail_mad_keep80"] = feat["sample"].isin(keep80_drop)

    pass_masks = {
        "pct_mad_3_5": ~feat["fail_pct_mad_3_5"].to_numpy(dtype=bool),
        "intra_mad_3_5": ~feat["fail_intra_mad_3_5"].to_numpy(dtype=bool),
        "ff_tail_5_95": ~feat["fail_ff_tail_5_95"].to_numpy(dtype=bool),
        "mad_or_ff": ~feat["fail_mad_or_ff"].to_numpy(dtype=bool),
        "mad_keep80": ~feat["fail_mad_keep80"].to_numpy(dtype=bool),
    }

    rule_rows = []
    viable = []
    for name in RULES:
        pmask = pass_masks[name]
        dropped = feat.loc[~pmask, "sample"].astype(str).tolist()
        admitted = feat.loc[pmask, "sample"].astype(str).tolist()
        n_drop = len(dropped)
        can = len(admitted) >= n_keep_min
        overlap = sorted(set(dropped) & toxic)
        rec = {
            "rule": name,
            "n_admitted": len(admitted),
            "n_dropped": n_drop,
            "can_redraw_40_40": can,
            "n_overlap_toxic_keep80": len(overlap),
            "overlap_toxic_keep80": ",".join(overlap),
            "dropped_samples": ",".join(dropped),
        }
        rule_rows.append(rec)
        dest = out / name
        _write_ids(dest / "dropped_samples.txt", dropped)
        _write_ids(dest / "admitted_samples.txt", admitted)
        if can:
            random_keep = _random_keep(test_ids, n_drop, random_seed)
            _write_ids(dest / "random_control_samples.txt", random_keep)
            viable.append(name)
        console.print(
            f"  {name}: drop {n_drop}/{len(test_ids)} keep={len(admitted)} "
            f"40+40={'yes' if can else 'NO'} overlap_toxic16={len(overlap)}"
        )

    summary = {
        "fence_source": "original_96_dev",
        "dev_pool": str(Path(dev_pool).resolve()),
        "n_dev": len(dev_ids),
        "test_pool": str(test_path.resolve()),
        "n_test": len(test_ids),
        "mad_cutoff": mad_cutoff,
        "ff_q05_dev": float(q05),
        "ff_q95_dev": float(q95),
        "n_keep_min": n_keep_min,
        "random_seed": random_seed,
        "note": (
            "Fences frozen on the original 96-dev pool. Test 40+40 FP/FN is not "
            "used. This is the independent transfer of a feature-only rule."
        ),
        "viable_rules": viable,
        "rules": rule_rows,
        "n_toxic_keep80": len(toxic),
    }
    (out / "rule_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rule_rows).to_csv(out / "rule_summary.tsv", sep="\t", index=False)
    console.print(f"[green]OK[/green] viable={viable} -> {out}")


if __name__ == "__main__":
    main()
