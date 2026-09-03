#!/usr/bin/env python3
"""Derive QC admittance rules and prove they move ref_free FP+FN.

Rules (applied only to the 96-dev-Normal pool):

  ff_tail_5_95     drop ff outside pool 5th–95th percentile
  pct_mad_3_5      drop any-chr |percentage MAD-z| > 3.5
  intra_mad_3_5    drop any-chr |hypo/hyper z_intra MAD-z| > 3.5
  mad_or_ff        union of the three above (primary QC)
  toxic_heldout    samples with toxic signature on *even* repeats (no leakage)

Proofs
------
1. Retrospective: among scored repeats, those whose 80 members all pass vs not.
2. Matched-N random drop: drop the same count of pool samples at random, K times;
   compare frac_perfect of "all 80 in kept set" to the QC rule.
3. Writes ``admitted_samples.txt`` for a prospective 40+40 redraw
   (``run_admitted_redraw.py`` / ``submit_score_repeats.sh`` with POOL_SAMPLES).

Toxic / MAD rules that leave fewer than 80 samples skip the redraw list and
are still reported retrospectively.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from analyze_perfect_vs_bad import build_enrichment, rank_toxic_protective
from common import (
    DEFAULT_OUT_BASE,
    DEFAULT_REF_N,
    class_mask,
    density_table,
    fp_fn_summary,
    load_repeat_shards,
)

console = Console()


def _pass_masks(feat: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean pass mask over pool rows (True = admitted)."""
    return {
        "ff_tail_5_95": ~feat["fail_ff_tail_5_95"].to_numpy(dtype=bool),
        "pct_mad_3_5": ~feat["fail_pct_mad_3_5"].to_numpy(dtype=bool),
        "intra_mad_3_5": ~feat["fail_intra_mad_3_5"].to_numpy(dtype=bool),
        "mad_or_ff": ~(
            feat["fail_ff_tail_5_95"].to_numpy(dtype=bool)
            | feat["fail_pct_mad_3_5"].to_numpy(dtype=bool)
            | feat["fail_intra_mad_3_5"].to_numpy(dtype=bool)
        ),
    }


def _toxic_mask_heldout(
    data: dict,
    feat: pd.DataFrame,
    even: np.ndarray,
) -> np.ndarray:
    enr = build_enrichment(
        data["mem_epz"][even],
        data["mem_ez"][even],
        data["fp_plus_fn"][even],
        data["pool"],
    )
    ranked = rank_toxic_protective(enr)
    toxic = set(ranked.loc[ranked["flag"] == "toxic", "sample"].astype(str))
    samples = feat["sample"].astype(str).to_numpy()
    return np.array([s not in toxic for s in samples], dtype=bool)


def _toxic_keep80_mask(
    data: dict,
    feat: pd.DataFrame,
    even: np.ndarray,
    ref_n: int,
) -> np.ndarray:
    """Drop the highest toxic_score samples on even repeats until 2*ref_n remain."""
    enr = build_enrichment(
        data["mem_epz"][even],
        data["mem_ez"][even],
        data["fp_plus_fn"][even],
        data["pool"],
    )
    ranked = rank_toxic_protective(enr)
    n_keep = 2 * ref_n
    n_drop = max(0, len(feat) - n_keep)
    drop = set(ranked.nlargest(n_drop, "toxic_score")["sample"].astype(str))
    samples = feat["sample"].astype(str).to_numpy()
    return np.array([s not in drop for s in samples], dtype=bool)


def _n_fail_members(mem_either: np.ndarray, pass_mask: np.ndarray) -> np.ndarray:
    fail = ~pass_mask
    return (mem_either.astype(np.int16) @ fail.astype(np.int16)).astype(np.int16)


def _all80_pass(mem_either: np.ndarray, pass_mask: np.ndarray) -> np.ndarray:
    """Repeats where every member of the 80 is in the admitted set."""
    return _n_fail_members(mem_either, pass_mask) == 0


def _dose_response(n_fail: np.ndarray, tot: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(int(n_fail.max()) + 1 if n_fail.size else 0):
        keep = n_fail == k
        n = int(keep.sum())
        if n == 0:
            continue
        sub = tot[keep]
        rows.append(
            {
                "n_fail_members": k,
                "n_repeats": n,
                "frac_repeats": n / tot.size,
                "frac_perfect": float((sub == 0).mean()),
                "mean_fp_plus_fn": float(sub.mean()),
                "mean_fn": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        r = spearmanr(x[m], y[m]).correlation
        return float(r) if r is not None else float("nan")
    except Exception:
        rx = pd.Series(x[m]).rank().to_numpy()
        ry = pd.Series(y[m]).rank().to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1])


def _summarize_subset(fp, fn, tot, keep: np.ndarray, label: str) -> dict:
    if keep.sum() == 0:
        return {
            "label": label,
            "n_repeats": 0,
            "frac_of_all": 0.0,
            "frac_perfect": float("nan"),
            "mean_fp_plus_fn": float("nan"),
            "mean_fp": float("nan"),
            "mean_fn": float("nan"),
            "frac_fp_plus_fn_ge_5": float("nan"),
        }
    s = fp_fn_summary(fp[keep], fn[keep], tot[keep])
    s["label"] = label
    s["frac_of_all"] = float(keep.mean())
    return s


def _random_drop_pass(n_pool: int, n_drop: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones(n_pool, dtype=bool)
    if n_drop <= 0:
        return mask
    drop = rng.choice(n_pool, size=n_drop, replace=False)
    mask[drop] = False
    return mask


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--score-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--analysis-dir", default=None, type=click.Path(file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--n-random", default=20, show_default=True, type=int)
@click.option("--random-seed", default=7, show_default=True, type=int)
@click.option("--ref-n", default=DEFAULT_REF_N, show_default=True, type=int)
@click.option(
    "--force-rule",
    default=None,
    type=str,
    help="If set, write admitted/dropped from this rule (e.g. toxic_keep80).",
)
def main(
    score_dir: str,
    analysis_dir: str | None,
    output_dir: str | None,
    n_random: int,
    random_seed: int,
    ref_n: int,
    force_rule: str | None,
) -> None:
    score_path = Path(score_dir)
    analysis = Path(analysis_dir) if analysis_dir else score_path / "analysis"
    out = Path(output_dir) if output_dir else analysis / "proof"
    out.mkdir(parents=True, exist_ok=True)
    console.rule("[bold blue]derive + prove admittance rules")

    feat = pd.read_csv(analysis / "sample_features.tsv", sep="\t")
    data = load_repeat_shards(score_path)
    tot = data["fp_plus_fn"]
    fp, fn = data["fp"], data["fn"]
    mem_either = (data["mem_epz"] | data["mem_ez"]).astype(bool)
    n_rep = tot.size
    even = (data["repeat_id"] % 2) == 0
    odd = ~even

    masks = _pass_masks(feat)
    masks["toxic_heldout"] = _toxic_mask_heldout(data, feat, even)
    masks["toxic_keep80"] = _toxic_keep80_mask(data, feat, even, ref_n)
    masks["mad_or_ff_and_toxic"] = masks["mad_or_ff"] & masks["toxic_heldout"]

    baseline = _summarize_subset(fp, fn, tot, np.ones(n_rep, dtype=bool), "all_repeats")
    rows = [baseline]
    # also odd-only baseline for held-out toxic
    rows.append(_summarize_subset(fp, fn, tot, odd, "odd_repeats_baseline"))

    rule_meta = []
    dose_rows = []
    for name, pmask in masks.items():
        n_drop = int((~pmask).sum())
        n_keep = int(pmask.sum())
        dropped = feat.loc[~pmask, "sample"].astype(str).tolist()
        eval_slice = (
            odd
            if name in {"toxic_heldout", "toxic_keep80", "mad_or_ff_and_toxic"}
            else np.ones(n_rep, dtype=bool)
        )
        n_fail = _n_fail_members(mem_either[eval_slice], pmask)
        tot_s = tot[eval_slice]
        fp_s = fp[eval_slice]
        fn_s = fn[eval_slice]
        keep_rep = n_fail == 0
        stats = _summarize_subset(fp_s, fn_s, tot_s, keep_rep, f"rule:{name}")
        stats["n_pool_kept"] = n_keep
        stats["n_pool_dropped"] = n_drop
        stats["dropped_samples"] = ",".join(dropped)
        stats["can_redraw_40_40"] = n_keep >= 2 * ref_n
        stats["spearman_nfail_vs_fpfn"] = _spearman(n_fail.astype(float), tot_s.astype(float))
        stats["mean_n_fail_perfect"] = (
            float(n_fail[tot_s == 0].mean()) if (tot_s == 0).any() else float("nan")
        )
        stats["mean_n_fail_bad"] = (
            float(n_fail[tot_s >= 5].mean()) if (tot_s >= 5).any() else float("nan")
        )
        rows.append(stats)

        dose = _dose_response(n_fail, tot_s)
        dose["rule"] = name
        dose["kind"] = "qc"
        dose_rows.append(dose)

        rng = np.random.default_rng(random_seed)
        rand_rho = []
        rand_fracs = []
        rand_means = []
        for k in range(n_random):
            rmask = _random_drop_pass(len(pmask), n_drop, rng)
            rn_fail = _n_fail_members(mem_either[eval_slice], rmask)
            rkeep = rn_fail == 0
            rand_rho.append(_spearman(rn_fail.astype(float), tot_s.astype(float)))
            rd = _dose_response(rn_fail, tot_s)
            rd["rule"] = name
            rd["kind"] = f"random_{k}"
            dose_rows.append(rd)
            if rkeep.sum() == 0:
                continue
            rand_fracs.append(float((tot_s[rkeep] == 0).mean()))
            rand_means.append(float(tot_s[rkeep].mean()))
        stats["random_n_valid"] = len(rand_fracs)
        stats["random_frac_perfect_mean"] = float(np.mean(rand_fracs)) if rand_fracs else float("nan")
        stats["random_frac_perfect_sd"] = float(np.std(rand_fracs)) if rand_fracs else float("nan")
        stats["random_mean_fpfn_mean"] = float(np.mean(rand_means)) if rand_means else float("nan")
        stats["random_spearman_mean"] = float(np.nanmean(rand_rho)) if rand_rho else float("nan")
        stats["random_spearman_sd"] = float(np.nanstd(rand_rho)) if rand_rho else float("nan")
        stats["spearman_minus_random"] = (
            stats["spearman_nfail_vs_fpfn"] - stats["random_spearman_mean"]
            if np.isfinite(stats["spearman_nfail_vs_fpfn"])
            and np.isfinite(stats["random_spearman_mean"])
            else float("nan")
        )
        delta = (
            stats["frac_perfect"] - stats["random_frac_perfect_mean"]
            if rand_fracs and np.isfinite(stats["frac_perfect"])
            else float("nan")
        )
        stats["frac_perfect_minus_random"] = delta
        rule_meta.append(stats)
        if keep_rep.any():
            dens = density_table(fp_s[keep_rep], fn_s[keep_rep], tot_s[keep_rep])
            dens.to_csv(out / f"density_{name}.tsv", sep="\t", index=False, float_format="%.6f")
        console.print(
            f"  {name}: drop {n_drop}/{len(pmask)} | "
            f"ρ(n_fail, FP+FN)={stats['spearman_nfail_vs_fpfn']:.3f} "
            f"vs random {stats['random_spearman_mean']:.3f} | "
            f"all80 n={stats['n_repeats']} perfect={stats['frac_perfect']}"
        )

    proof = pd.DataFrame(rows)
    proof.to_csv(out / "proof_retrospective.tsv", sep="\t", index=False, float_format="%.6f")
    if dose_rows:
        pd.concat(dose_rows, ignore_index=True).to_csv(
            out / "proof_dose_response.tsv", sep="\t", index=False, float_format="%.6f"
        )

    # Prefer the QC rule with the largest Spearman lift vs random among pools that still support 40+40.
    if force_rule:
        if force_rule not in masks:
            raise click.ClickException(f"unknown --force-rule {force_rule}")
        primary = force_rule
        pmask = masks[primary]
        if int(pmask.sum()) < 2 * ref_n:
            raise click.ClickException(
                f"--force-rule {force_rule} leaves {int(pmask.sum())} < {2 * ref_n}"
            )
        console.print(f"  primary forced: {primary} keep={int(pmask.sum())}")
    else:
        viable = []
        for name, pmask in masks.items():
            if int(pmask.sum()) < 2 * ref_n:
                continue
            meta = next((r for r in rule_meta if r.get("label") == f"rule:{name}"), None)
            lift = float(meta["spearman_minus_random"]) if meta else float("-inf")
            if not np.isfinite(lift):
                lift = float("-inf")
            viable.append((lift, name, pmask))
        if viable:
            viable.sort(key=lambda t: t[0], reverse=True)
            _, primary, pmask = viable[0]
        else:
            primary = "pct_mad_3_5"
            pmask = masks[primary]
        console.print(f"  primary selected: {primary} keep={int(pmask.sum())}")
    admitted = feat.loc[pmask, "sample"].astype(str).tolist()
    dropped = feat.loc[~pmask, "sample"].astype(str).tolist()
    (out / "admitted_samples.txt").write_text("\n".join(admitted) + "\n")
    (out / "dropped_samples.txt").write_text("\n".join(dropped) + "\n")

    rng = np.random.default_rng(random_seed)
    rand_mask = _random_drop_pass(len(pmask), int((~pmask).sum()), rng)
    random_samples = feat.loc[rand_mask, "sample"].astype(str).tolist()
    (out / "random_control_samples.txt").write_text("\n".join(random_samples) + "\n")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    summary = {
        "primary_rule": primary,
        "n_pool": int(len(feat)),
        "n_admitted": len(admitted),
        "n_dropped": len(dropped),
        "dropped_samples": dropped,
        "can_redraw_40_40": len(admitted) >= 2 * ref_n,
        "n_random_control": len(random_samples),
        "baseline": baseline,
        "rules": rule_meta,
    }
    (out / "proof_summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    console.print(
        f"[green]OK[/green] primary={primary} admitted={len(admitted)} "
        f"dropped={dropped} -> {out}"
    )


if __name__ == "__main__":
    main()
