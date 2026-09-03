#!/usr/bin/env python3
"""
Re-select ref_40 to match meta (ref_17) pred_labels on strong calls.

Root cause of PTAY1472P9S1 Gray_T16→T16: the previous search optimized against a
*recomputed* early_ref baseline (already T16 at ez16≈4.554), not the stored meta
final_zscores (Gray_T16 at ez16≈4.441). This script uses meta final_zscores as the
pred baseline, excludes emergency samples, and hard-protects known borderline
Gray samples (incl. PTAY1472P9S1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "reference_explore"))
sys.path.insert(0, str(SCRIPT_DIR))

from calc_zscore_episcore_ezscore import (  # noqa: E402
    DEFAULT_EZSCORE_REF_SAMPLES,
    _ref_mean_std,
    build_ezscore_ref_mask,
)
from pred_label_utils import (  # noqa: E402
    assign_pred_labels_matrix,
    parse_comma_scores,
)
from select_ref40 import (  # noqa: E402
    _compute_all_scores,
    _load_merged,
    _mean_std_distance,
    _meanstd_compare_table,
    _pred_masks,
    _prepare_arrays,
    _score_table,
)
console = Console()

# Borderline Gray samples that flipped Gray→T under the 20260730 ref40
DEFAULT_PROTECT = {
    "PTAY1472P9S1": 15,  # chr16
    "PTAY1253P6H1": 14,  # chr15
    "PTAY0704P7H1": 14,  # chr15
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-dir",
    default="/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260730-stable_ref40",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--output-dir",
    default="/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260812-stable_ref40",
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--meta-csv",
    default="/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet.csv",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--ref-n", default=40, show_default=True, type=int)
@click.option("--n-early-seed", default=8000, show_default=True, type=int)
@click.option("--n-random", default=40000, show_default=True, type=int)
@click.option("--n-swap-rounds", default=30000, show_default=True, type=int)
@click.option("--seed", default=20260812, show_default=True, type=int)
@click.option("--meta-agree-max-delta", default=1.0, show_default=True, type=float)
@click.option(
    "--write-meta/--no-write-meta",
    default=True,
    show_default=True,
    help="Also write temporary_updated_samplesheet_ref40.csv",
)
@click.option(
    "--copy-plot-dir",
    default="/lustre1/cqyi/syfan/nipt_article_plot",
    show_default=True,
    type=click.Path(file_okay=False),
)
def main(
    input_dir: str,
    output_dir: str,
    meta_csv: str,
    ref_n: int,
    n_early_seed: int,
    n_random: int,
    n_swap_rounds: int,
    seed: int,
    meta_agree_max_delta: float,
    write_meta: bool,
    copy_plot_dir: str,
) -> None:
    """Search ref_40 matching meta strong calls; protect borderline Grays."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse beta/percentage/meta tables from prior build
    for name in ("beta.csv", "percentage.csv", "meta.csv"):
        src, dst = in_dir / name, out_dir / name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    merged, pct_path = _load_merged(out_dir, out_dir / "meta.csv")
    arrays = _prepare_arrays(merged, pct_path)
    ez_idx = np.flatnonzero(
        build_ezscore_ref_mask(merged, samples_file=DEFAULT_EZSCORE_REF_SAMPLES)
    )
    early_idx = np.flatnonzero((merged["ref_type"].astype(str) == "early_ref").to_numpy())
    pct_samples = set(
        pd.read_csv(pct_path, sep="\t", usecols=["sample"])["sample"].astype(str)
    )
    pool_mask = (
        (merged["set"].astype(str) == "dev")
        & (merged["label"].astype(str) == "Normal")
        & merged["sample"].astype(str).isin(pct_samples)
    )
    pool_idx = np.flatnonzero(pool_mask.to_numpy())
    target = {
        k: _ref_mean_std(arrays[k][early_idx]) for k in ("hypo_z", "hyper_z", "pct")
    }

    meta = pd.read_csv(meta_csv).drop_duplicates("sample")
    meta["sample"] = meta["sample"].astype(str)
    mm = merged[["sample"]].merge(
        meta[["sample", "final_zscores", "pred_label", "set"]], on="sample", how="left"
    )
    ez_meta = np.vstack([parse_comma_scores(v) for v in mm["final_zscores"]])
    has_meta = np.isfinite(ez_meta).all(axis=1)
    base_strong, base_gray = _pred_masks(ez_meta)
    meta_pred = np.array(assign_pred_labels_matrix(ez_meta))

    early_mask = (merged["ref_type"].astype(str) == "early_ref").to_numpy()
    has_pct = merged["sample"].astype(str).isin(pct_samples).to_numpy()
    not_emerg = (merged["set"].astype(str) != "emergency").to_numpy()
    z0, e0, ez0 = _compute_all_scores(arrays, early_idx, ez_idx)
    delta = np.nanmax(
        np.where(np.isfinite(ez_meta), np.abs(ez0 - ez_meta), np.nan), axis=1
    )
    agree = np.isfinite(delta) & (delta < meta_agree_max_delta)
    base_compare = has_pct & has_meta & (~early_mask) & not_emerg & agree

    protect = dict(DEFAULT_PROTECT)
    protect_idx = {
        name: int(np.flatnonzero(merged["sample"].astype(str).eq(name))[0])
        for name in protect
    }
    si = protect_idx["PTAY1472P9S1"]

    console.print(
        f"pool={pool_idx.size} compare={base_compare.sum()} "
        f"early_ref={early_idx.size} protect={list(protect)}"
    )

    def eval_ref(ref: np.ndarray) -> dict:
        dist = _mean_std_distance(arrays, ref, target)
        z, e, ez = _compute_all_scores(arrays, ref, ez_idx)
        rmask = np.zeros(len(merged), dtype=bool)
        rmask[ref] = True
        c = base_compare & (~rmask)
        new_strong, new_gray = _pred_masks(ez)
        ns = int((((base_strong != new_strong).any(axis=1)) & c).sum())
        ng = int((((base_gray != new_gray).any(axis=1)) & c).sum())
        n_protect_fail = 0
        for name, chr_i in protect.items():
            i = protect_idx[name]
            if not (3.0 <= ez[i, chr_i] <= 4.5):
                n_protect_fail += 1
        return {
            "dist": float(dist),
            "z": z,
            "e": e,
            "ez": ez,
            "ns": ns,
            "ng": ng,
            "n_protect_fail": n_protect_fail,
            "compare": c,
            "ref": ref.copy(),
            "key": (n_protect_fail, ns, ng, float(dist)),
        }

    rng = np.random.default_rng(seed)
    best: Optional[dict] = None
    candidates: List[dict] = []

    def consider(ref: np.ndarray, tag: str) -> None:
        nonlocal best
        res = eval_ref(ref)
        res["tag"] = tag
        candidates.append(
            {
                "tag": tag,
                "dist": res["dist"],
                "ns": res["ns"],
                "ng": res["ng"],
                "n_protect_fail": res["n_protect_fail"],
            }
        )
        if best is None or res["key"] < best["key"]:
            best = res
            console.print(
                f"  new best {tag}: protect_fail={res['n_protect_fail']} "
                f"ns={res['ns']} ng={res['ng']} dist={res['dist']:.4f} "
                f"ez16={res['ez'][si, 15]:.4f}"
            )

    early_in_pool = np.intersect1d(early_idx, pool_idx)
    fillers = np.setdiff1d(pool_idx, early_in_pool)
    need = ref_n - int(early_in_pool.size)
    console.print(f"early_seed draws ({n_early_seed}), need fillers={need} ...")
    for i in range(n_early_seed):
        fill = rng.choice(fillers, size=need, replace=False)
        consider(np.sort(np.concatenate([early_in_pool, fill])), f"early_seed_{i}")

    console.print(f"random draws ({n_random}) ...")
    seen = set()
    for i in range(n_random):
        draw = np.sort(rng.choice(pool_idx, size=ref_n, replace=False))
        key = draw.tobytes()
        if key in seen:
            continue
        seen.add(key)
        consider(draw, f"random_{i}")

    assert best is not None
    console.print(f"local swaps ({n_swap_rounds}) ...")
    cur = best["ref"].copy()
    cur_key = best["key"]
    for i in range(n_swap_rounds):
        in_ref = set(map(int, cur))
        out_cand = [int(x) for x in pool_idx if int(x) not in in_ref]
        drop = int(rng.choice(cur))
        add = int(rng.choice(out_cand))
        trial = np.sort(
            np.array([x for x in cur if x != drop] + [add], dtype=np.int64)
        )
        res = eval_ref(trial)
        res["tag"] = f"swap_{i}"
        candidates.append(
            {
                "tag": res["tag"],
                "dist": res["dist"],
                "ns": res["ns"],
                "ng": res["ng"],
                "n_protect_fail": res["n_protect_fail"],
            }
        )
        if res["key"] < cur_key:
            cur = trial
            cur_key = res["key"]
        if res["key"] < best["key"]:
            best = res
            console.print(
                f"  new best swap_{i}: protect_fail={res['n_protect_fail']} "
                f"ns={res['ns']} ng={res['ng']} dist={res['dist']:.4f} "
                f"ez16={res['ez'][si, 15]:.4f}"
            )

    pred = np.array(assign_pred_labels_matrix(best["ez"]))
    ref_samples = merged.iloc[best["ref"]]["sample"].astype(str).tolist()
    console.rule("[bold green]FINAL")
    console.print(
        f"key={best['key']} tag={best['tag']} PTAY1472 ez16={best['ez'][si, 15]:.4f} "
        f"pred={pred[si]}"
    )
    for name, chr_i in protect.items():
        i = protect_idx[name]
        console.print(
            f"  {name}: ez_chr{chr_i + 1}={best['ez'][i, chr_i]:.4f} "
            f"pred={pred[i]} meta={mm.iloc[i]['pred_label']}"
        )

    new_strong, _ = _pred_masks(best["ez"])
    remain = np.flatnonzero(
        ((base_strong != new_strong).any(axis=1)) & best["compare"]
    )
    for i in remain:
        console.print(
            f"  remain STRONG {merged.iloc[i]['sample']} "
            f"{mm.iloc[i]['pred_label']} -> {pred[i]}"
        )

    # Write artifacts
    (out_dir / "ref40_samples.txt").write_text("\n".join(ref_samples) + "\n")
    pd.DataFrame({"sample": ref_samples}).to_csv(
        out_dir / "ref40_samples.tsv", sep="\t", index=False
    )
    _score_table(merged["sample"].tolist(), z0, e0, ez0).to_csv(
        out_dir / "baseline_score.tsv", sep="\t", index=False
    )
    score_df = _score_table(
        merged["sample"].tolist(), best["z"], best["e"], best["ez"]
    )
    score_df.to_csv(out_dir / "ref40_score.tsv", sep="\t", index=False)
    meanstd = _meanstd_compare_table(arrays, early_idx, best["ref"])
    meanstd.to_csv(out_dir / "reference_meanstd_compare.tsv", sep="\t", index=False)

    ref_mask = np.zeros(len(merged), dtype=bool)
    ref_mask[best["ref"]] = True
    cmp_df = pd.DataFrame(
        {
            "sample": merged["sample"].astype(str),
            "label": merged["label"].astype(str),
            "set": merged["set"].astype(str),
            "old_ref_type": merged["ref_type"].astype(str),
            "in_early_ref": early_mask,
            "in_ref40": ref_mask,
            "compared": best["compare"],
            "pred_label_meta": meta_pred,
            "pred_label_baseline_recompute": np.array(
                assign_pred_labels_matrix(ez0)
            ),
            "pred_label_ref40": pred,
            "changed_vs_meta": meta_pred != pred,
            "strong_changed_vs_meta": (base_strong != (best["ez"] > 4.5)).any(axis=1),
            "meta_pred_label": mm["pred_label"].astype(str),
        }
    )
    cmp_df.to_csv(out_dir / "pred_label_compare.tsv", sep="\t", index=False)

    cand_df = (
        pd.DataFrame(candidates)
        .sort_values(["n_protect_fail", "ns", "ng", "dist"])
        .head(50)
        .reset_index(drop=True)
    )
    cand_df.insert(0, "rank", range(1, len(cand_df) + 1))
    cand_df.to_csv(out_dir / "search_top_candidates.tsv", sep="\t", index=False)

    summary = {
        "ref_n": ref_n,
        "seed": seed,
        "n_early_seed": n_early_seed,
        "n_random": n_random,
        "n_swap_rounds": n_swap_rounds,
        "pred_baseline": "meta_final_zscores",
        "exclude_sets": ["emergency"],
        "meta_agree_max_delta": meta_agree_max_delta,
        "protect_gray_samples": protect,
        "pool_size": int(pool_idx.size),
        "early_ref_n": int(early_idx.size),
        "overlap_with_early_ref": len(
            set(ref_samples)
            & set(merged.iloc[early_idx]["sample"].astype(str))
        ),
        "best_tag": best["tag"],
        "best_n_protect_fail": int(best["n_protect_fail"]),
        "best_n_strong_changed": int(best["ns"]),
        "best_n_gray_changed": int(best["ng"]),
        "best_n_compare": int(best["compare"].sum()),
        "best_meanstd_dist": float(best["dist"]),
        "ptay1472_ezscore_chr16": float(best["ez"][si, 15]),
        "ptay1472_pred_label": pred[si],
        "ref40_samples": ref_samples,
        "mean_abs_delta_mean": {
            feat: float(
                meanstd.loc[meanstd["feature"] == feat, "delta_mean"].abs().mean()
            )
            for feat in meanstd["feature"].unique()
        },
        "mean_abs_delta_std": {
            feat: float(
                meanstd.loc[meanstd["feature"] == feat, "delta_std"].abs().mean()
            )
            for feat in meanstd["feature"].unique()
        },
        "note": (
            "Selected vs meta final_zscores (ref_17 samplesheet), not recomputed "
            "early_ref baseline. Protects Gray→T borderlines incl. PTAY1472P9S1."
        ),
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"[green]OK[/green] Wrote selection artifacts under {out_dir}")

    if write_meta:
        out_csv = out_dir / "temporary_updated_samplesheet_ref40.csv"
        meta_out = pd.read_csv(meta_csv)
        meta_out["sample"] = meta_out["sample"].astype(str)
        score_cols = ["sample", "beta_zscores", "rc_zscores", "final_zscores", "pred_label"]
        out = meta_out.merge(
            score_df[score_cols], on="sample", how="left", suffixes=("", "_new")
        )
        for col in ("beta_zscores", "rc_zscores", "final_zscores", "pred_label"):
            new_col = f"{col}_new"
            out[col] = out[new_col].where(out[new_col].notna(), out[col])
            out = out.drop(columns=[new_col])
        ref_set = set(ref_samples)
        old_early = out["ref_type"].astype(str) == "early_ref"
        in_ref40 = out["sample"].isin(ref_set)
        out.loc[old_early & ~in_ref40, "ref_type"] = "analyze"
        out.loc[in_ref40, "ref_type"] = "early_ref"
        out.loc[out["ref_type"].astype(str) == "early_ref", "pred_label"] = pd.NA
        out.to_csv(out_csv, index=False)
        console.print(f"[green]OK[/green] Wrote {out_csv}")

        plot_dir = Path(copy_plot_dir)
        if plot_dir.is_dir():
            dest = plot_dir / "temporary_updated_samplesheet_ref40.csv"
            dest.write_bytes(out_csv.read_bytes())
            console.print(f"[green]OK[/green] Copied to {dest}")

        row = out.loc[out["sample"].astype(str) == "PTAY1472P9S1"].iloc[0]
        zs = [float(x) for x in str(row["final_zscores"]).split(",")]
        console.print(
            f"Verify PTAY1472P9S1: pred_label={row['pred_label']} "
            f"ez16={zs[15]:.4f} (meta ref17 was Gray_T16 / 4.4409)"
        )


if __name__ == "__main__":
    try:
        main(standalone_mode=False)
    except click.ClickException as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
