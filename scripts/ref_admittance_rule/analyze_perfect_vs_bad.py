#!/usr/bin/env python3
"""Q1 sample enrichment + Q2 set-level distributions for perfect vs bad 40+40.

Reads compact ``repeats_*.npz`` from ``score_repeats.py`` and writes:

  density_check.tsv / density_vs_20260810.tsv
  sample_features.tsv
  sample_enrichment.tsv
  toxic_protective.tsv
  set_feature_compare.tsv
  set_features_by_class.tsv (long, sampled if huge)
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from common import (
    BAD_K,
    CHR_LIST,
    DEFAULT_DENSITY_TSV,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUT_BASE,
    class_mask,
    cliffs_delta,
    density_table,
    fp_fn_summary,
    load_repeat_shards,
    load_universe,
    mad_z,
    parse_sample_list,
    try_fisher,
    try_mannwhitney,
)

console = Console()

ROLES = ("epz", "ez", "either", "unused")
CLASSES = ("perfect", "ok", "bad", "worst")


def _role_masks(mem_epz: np.ndarray, mem_ez: np.ndarray) -> dict[str, np.ndarray]:
    either = (mem_epz | mem_ez).astype(bool)
    return {
        "epz": mem_epz.astype(bool),
        "ez": mem_ez.astype(bool),
        "either": either,
        "unused": ~either,
    }


def _outlier_chr_string(abs_z: np.ndarray, cutoff: float) -> str:
    hits = [CHR_LIST[i] for i, v in enumerate(abs_z) if v > cutoff]
    return ",".join(hits)


def build_sample_features(ctx: dict, pool: pd.DataFrame) -> pd.DataFrame:
    pool_idx = ctx["ref_pool_idx"]
    hypo = ctx["ep_arrays"][0][:, pool_idx]  # chr x pool
    hyper = ctx["ep_arrays"][1][:, pool_idx]
    pct = ctx["z_array"][:, pool_idx]
    hypo_z = mad_z(hypo, axis=1)
    hyper_z = mad_z(hyper, axis=1)
    pct_z = mad_z(pct, axis=1)
    rows = []
    for j in range(len(pool)):
        row = pool.iloc[j]
        hz = np.abs(hypo_z[:, j])
        xz = np.abs(hyper_z[:, j])
        pz = np.abs(pct_z[:, j])
        intra = np.maximum(hz, xz)
        rows.append(
            {
                "pool_index": int(row["pool_index"]),
                "sample": str(row["sample"]),
                "ff_before_mq": float(row["ff_before_mq"])
                if pd.notna(row["ff_before_mq"])
                else float("nan"),
                "coverage": float(ctx["coverage_arr"][pool_idx[j]])
                if np.isfinite(ctx["coverage_arr"][pool_idx[j]])
                else float("nan"),
                "max_abs_final_z": float(ctx["max_abs_final"][pool_idx[j]]),
                "max_abs_pct_madz": float(np.nanmax(pz)),
                "max_abs_intra_madz": float(np.nanmax(intra)),
                "n_chr_pct_madz_gt3": int((pz > 3.0).sum()),
                "n_chr_pct_madz_gt3_5": int((pz > 3.5).sum()),
                "n_chr_intra_madz_gt3": int((intra > 3.0).sum()),
                "n_chr_intra_madz_gt3_5": int((intra > 3.5).sum()),
                "outlier_chrs_pct_3": _outlier_chr_string(pz, 3.0),
                "outlier_chrs_pct_3_5": _outlier_chr_string(pz, 3.5),
                "outlier_chrs_intra_3": _outlier_chr_string(intra, 3.0),
                "outlier_chrs_intra_3_5": _outlier_chr_string(intra, 3.5),
                "fail_pct_mad_3_5": bool((pz > 3.5).any()),
                "fail_intra_mad_3_5": bool((intra > 3.5).any()),
            }
        )
        for i, chr_name in enumerate(CHR_LIST):
            rows[-1][f"{chr_name}_pct_madz"] = float(pct_z[i, j])
            rows[-1][f"{chr_name}_hypo_madz"] = float(hypo_z[i, j])
            rows[-1][f"{chr_name}_hyper_madz"] = float(hyper_z[i, j])
    feat = pd.DataFrame(rows)
    ff = feat["ff_before_mq"].to_numpy(dtype=float)
    q05, q95 = np.nanquantile(ff, [0.05, 0.95])
    q01, q99 = np.nanquantile(ff, [0.01, 0.99])
    feat["ff_q05"] = q05
    feat["ff_q95"] = q95
    feat["fail_ff_tail_5_95"] = (ff < q05) | (ff > q95)
    feat["fail_ff_tail_1_99"] = (ff < q01) | (ff > q99)
    return feat


def build_enrichment(
    mem_epz: np.ndarray,
    mem_ez: np.ndarray,
    tot: np.ndarray,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    roles = _role_masks(mem_epz, mem_ez)
    n_rep = int(tot.size)
    n_pool = mem_epz.shape[1]
    class_masks = {c: class_mask(tot, c) for c in CLASSES}
    rows = []
    for j in range(n_pool):
        base = {
            "pool_index": j,
            "sample": str(pool.iloc[j]["sample"]),
        }
        for role, mask in roles.items():
            in_role = mask[:, j]
            p_in = float(in_role.mean())
            rec = {
                **base,
                "role": role,
                "p_in_role": p_in,
                "n_in_role": int(in_role.sum()),
            }
            for cname, cm in class_masks.items():
                n_class = int(cm.sum())
                n_in_and = int((in_role & cm).sum())
                p_in_class = (n_in_and / n_class) if n_class else float("nan")
                lift = (p_in_class / p_in) if p_in > 0 else float("nan")
                rec[f"n_{cname}"] = n_class
                rec[f"n_in_{cname}"] = n_in_and
                rec[f"p_in_given_{cname}"] = p_in_class
                rec[f"lift_{cname}"] = lift
                # P(class | in_role) vs P(class)
                n_in = int(in_role.sum())
                p_class_in = (n_in_and / n_in) if n_in else float("nan")
                p_class = n_class / n_rep if n_rep else float("nan")
                rec[f"p_{cname}_given_in"] = p_class_in
                rec[f"p_{cname}"] = p_class
                # 2x2: in_role vs class
                a = n_in_and
                b = n_in - n_in_and
                c = n_class - n_in_and
                d = n_rep - n_in - n_class + n_in_and
                oddsr, pval = try_fisher(np.array([[a, b], [c, d]]))
                rec[f"fisher_or_{cname}"] = oddsr
                rec[f"fisher_p_{cname}"] = pval
            rows.append(rec)
    return pd.DataFrame(rows)


def rank_toxic_protective(enr: pd.DataFrame) -> pd.DataFrame:
    either = enr.loc[enr["role"] == "either"].copy()
    unused = enr.loc[enr["role"] == "unused"].copy()
    u = unused.set_index("sample")
    either["p_perfect_given_unused"] = either["sample"].map(u["p_perfect_given_in"])
    either["lift_perfect_unused"] = either["sample"].map(u["lift_perfect"])
    either["toxic_score"] = (
        either["lift_bad"].fillna(1.0)
        + (either["p_perfect_given_unused"] - either["p_perfect_given_in"]).fillna(0.0)
    )
    either["protective_score"] = either["lift_perfect"].fillna(1.0)
    either["flag"] = "neutral"
    # toxic: enriched in bad AND more perfect when left out
    toxic = (
        (either["lift_bad"] > 1.0)
        & (either["p_perfect_given_unused"] > either["p_perfect_given_in"])
        & (either["fisher_p_bad"] < 0.05)
    )
    prot = (
        (either["lift_perfect"] > 1.0)
        & (either["p_perfect_given_in"] > either["p_perfect"].fillna(0))
        & (either["fisher_p_perfect"] < 0.05)
    )
    either.loc[toxic, "flag"] = "toxic"
    either.loc[prot & ~toxic, "flag"] = "protective"
    keep = [
        "sample",
        "flag",
        "p_in_role",
        "lift_perfect",
        "lift_bad",
        "p_perfect_given_in",
        "p_perfect_given_unused",
        "p_bad_given_in",
        "fisher_p_perfect",
        "fisher_p_bad",
        "toxic_score",
        "protective_score",
    ]
    flag_order = {"toxic": 0, "protective": 1, "neutral": 2}
    either["_ord"] = either["flag"].map(flag_order)
    return (
        either[keep + ["_ord"]]
        .sort_values(["_ord", "toxic_score"], ascending=[True, False])
        .drop(columns="_ord")
    )


def build_set_compare(data: dict, feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tot = data["fp_plus_fn"]
    mem_either = (data["mem_epz"] | data["mem_ez"]).astype(bool)
    fail_any = (
        feat["fail_pct_mad_3_5"].to_numpy(dtype=bool)
        | feat["fail_intra_mad_3_5"].to_numpy(dtype=bool)
        | feat["fail_ff_tail_5_95"].to_numpy(dtype=bool)
    )
    n_fail_members = mem_either.astype(np.int16) @ fail_any.astype(np.int16)
    ez_sd_mean = data["ez_sd"].mean(axis=1)
    ez_sd_max = data["ez_sd"].max(axis=1)
    pct_sd_mean = data["pct_sd"].mean(axis=1)
    pct_sd_max = data["pct_sd"].max(axis=1)
    ff_range = data["ff_80_max"] - data["ff_80_min"]

    metrics = {
        "ff_80_mean": data["ff_80_mean"],
        "ff_80_std": data["ff_80_std"],
        "ff_80_min": data["ff_80_min"],
        "ff_80_max": data["ff_80_max"],
        "ff_80_range": ff_range,
        "ff_epz_mean": data["ff_epz_mean"],
        "ff_epz_std": data["ff_epz_std"],
        "ff_ez_mean": data["ff_ez_mean"],
        "ff_ez_std": data["ff_ez_std"],
        "ez_sd_mean": ez_sd_mean,
        "ez_sd_max": ez_sd_max,
        "pct_sd_mean": pct_sd_mean,
        "pct_sd_max": pct_sd_max,
        "n_fail_members": n_fail_members.astype(float),
        "fp_plus_fn": tot.astype(float),
        "fp": data["fp"].astype(float),
        "fn": data["fn"].astype(float),
    }
    # per-chr ez sd
    for i, chr_name in enumerate(CHR_LIST):
        metrics[f"ez_sd_{chr_name}"] = data["ez_sd"][:, i]
        metrics[f"pct_sd_{chr_name}"] = data["pct_sd"][:, i]

    perfect = class_mask(tot, "perfect")
    bad = class_mask(tot, "bad")
    rows = []
    for name, arr in metrics.items():
        x = arr[perfect]
        y = arr[bad]
        rows.append(
            {
                "metric": name,
                "n_perfect": int(perfect.sum()),
                "n_bad": int(bad.sum()),
                "mean_perfect": float(np.nanmean(x)),
                "mean_bad": float(np.nanmean(y)),
                "median_perfect": float(np.nanmedian(x)),
                "median_bad": float(np.nanmedian(y)),
                "delta_mean_bad_minus_perfect": float(np.nanmean(y) - np.nanmean(x)),
                "mannwhitney_p": try_mannwhitney(x, y),
                "cliffs_delta_bad_minus_perfect": cliffs_delta(y, x),
            }
        )
    compare = pd.DataFrame(rows).sort_values(
        "cliffs_delta_bad_minus_perfect", key=lambda s: s.abs(), ascending=False
    )
    by_class = pd.DataFrame(
        {
            "repeat_id": data["repeat_id"],
            "class": np.where(perfect, "perfect", np.where(bad, "bad", "ok")),
            "fp": data["fp"],
            "fn": data["fn"],
            "fp_plus_fn": tot,
            "ff_80_mean": data["ff_80_mean"],
            "ff_80_std": data["ff_80_std"],
            "ff_80_range": ff_range,
            "ez_sd_mean": ez_sd_mean,
            "ez_sd_max": ez_sd_max,
            "pct_sd_mean": pct_sd_mean,
            "n_fail_members": n_fail_members,
        }
    )
    return compare, by_class


def compare_density_to_ref(dens: pd.DataFrame, ref_path: Path) -> pd.DataFrame:
    if not ref_path.is_file():
        return pd.DataFrame()
    ref = pd.read_csv(ref_path, sep="\t")
    merged = dens.merge(
        ref[["fp_plus_fn", "density", "n_repeats"]].rename(
            columns={"density": "density_1e6", "n_repeats": "n_repeats_1e6"}
        ),
        on="fp_plus_fn",
        how="outer",
    ).sort_values("fp_plus_fn")
    merged["density_diff"] = merged["density"] - merged["density_1e6"]
    return merged


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--score-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--input-dir", default=str(DEFAULT_INPUT_DIR), type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, type=click.Path(file_okay=False))
@click.option("--ref-density-tsv", default=str(DEFAULT_DENSITY_TSV), type=click.Path())
def main(score_dir: str, input_dir: str, output_dir: str | None, ref_density_tsv: str) -> None:
    score_path = Path(score_dir)
    out = Path(output_dir) if output_dir else score_path / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    console.rule("[bold blue]analyze perfect vs bad")

    data = load_repeat_shards(score_path)
    tot = data["fp_plus_fn"]
    fp, fn = data["fp"], data["fn"]
    summary = fp_fn_summary(fp, fn, tot)
    dens = density_table(fp, fn, tot)
    dens.to_csv(out / "fp_fn_density.tsv", sep="\t", index=False, float_format="%.6f")
    (out / "fp_fn_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print(
        f"  n={summary['n_repeats']} perfect={summary['frac_perfect']:.4f} "
        f"mean FP+FN={summary['mean_fp_plus_fn']:.3f} bad≥5={summary['frac_fp_plus_fn_ge_5']:.4f}"
    )

    vs = compare_density_to_ref(dens, Path(ref_density_tsv))
    if not vs.empty:
        vs.to_csv(out / "density_vs_20260810.tsv", sep="\t", index=False, float_format="%.6f")
        max_diff = float(vs["density_diff"].abs().max())
        console.print(f"  max |density − 20260810 1e6| = {max_diff:.4f}")

    cfg = data.get("cfg") or {}
    excl_file = cfg.get("exclude_eval_samples_file")
    excl = parse_sample_list(Path(excl_file)) if excl_file and Path(excl_file).is_file() else None
    ctx = load_universe(
        Path(input_dir),
        pool_samples=data["pool"]["sample"].astype(str).tolist(),
        pool_source=str(cfg.get("pool_source", "dev_normal")),
        exclude_eval_samples=excl,
    )
    feat = build_sample_features(ctx, data["pool"])
    feat.to_csv(out / "sample_features.tsv", sep="\t", index=False, float_format="%.6f")

    enr = build_enrichment(data["mem_epz"], data["mem_ez"], tot, data["pool"])
    enr = enr.merge(
        feat[
            [
                "sample",
                "ff_before_mq",
                "max_abs_pct_madz",
                "max_abs_intra_madz",
                "n_chr_pct_madz_gt3_5",
                "n_chr_intra_madz_gt3_5",
                "outlier_chrs_pct_3_5",
                "outlier_chrs_intra_3_5",
                "fail_pct_mad_3_5",
                "fail_intra_mad_3_5",
                "fail_ff_tail_5_95",
                "coverage",
                "max_abs_final_z",
            ]
        ],
        on="sample",
        how="left",
    )
    enr.to_csv(out / "sample_enrichment.tsv", sep="\t", index=False, float_format="%.6f")

    ranked = rank_toxic_protective(enr)
    ranked = ranked.merge(
        feat[
            [
                "sample",
                "ff_before_mq",
                "max_abs_pct_madz",
                "max_abs_intra_madz",
                "outlier_chrs_pct_3_5",
                "outlier_chrs_intra_3_5",
                "fail_pct_mad_3_5",
                "fail_intra_mad_3_5",
                "fail_ff_tail_5_95",
            ]
        ],
        on="sample",
        how="left",
    )
    ranked.to_csv(out / "toxic_protective.tsv", sep="\t", index=False, float_format="%.6f")
    console.print(
        "  flags:",
        ranked["flag"].value_counts().to_dict(),
        "n_pct_mad_fail",
        int(feat["fail_pct_mad_3_5"].sum()),
        "n_intra_mad_fail",
        int(feat["fail_intra_mad_3_5"].sum()),
        "n_ff_tail",
        int(feat["fail_ff_tail_5_95"].sum()),
    )

    compare, by_class = build_set_compare(data, feat)
    compare.to_csv(out / "set_feature_compare.tsv", sep="\t", index=False, float_format="%.6f")
    by_class.to_csv(out / "set_features_by_class.tsv", sep="\t", index=False, float_format="%.6f")
    top = compare.head(12)[
        ["metric", "mean_perfect", "mean_bad", "cliffs_delta_bad_minus_perfect", "mannwhitney_p"]
    ]
    console.print(top.to_string(index=False))
    console.print(f"[green]OK[/green] wrote {out}")


if __name__ == "__main__":
    main()
