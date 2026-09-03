#!/usr/bin/env python3
"""Search chr21 epi/z combos for T21 detection of JPTAY query batches.

Keeps the production 17 early_ref (episcore/zscore mu/sigma) and 25 ez-ref
samples unchanged. Reference raw scores are read from the previous grid-search
parquets so mu/sigma are not recomputed from BAM/deconv.

For each (epi_combo, z_combo) on ``--target-chr`` (default chr21):
    episcore = s_inter vs 17 early_ref
    zscore   = percentage z vs 17 early_ref
    ezscore  = z-normalize(episcore + zscore) vs 25 ez refs

A combo "detects T21" when every query batch has target-chr ezscore > 4.5.
Other chromosomes stay on the production combo (0.5/0.65 and 0.85/0.95) in
the per-batch report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from t21_combo_common import (
    CHR_LIST,
    DEFAULT_GRID_INPUT,
    EZ_CUTOFF,
    PRODUCTION_EP,
    PRODUCTION_Z,
    fmt_combo,
    load_sample_list,
    maybe_fraction,
    pred_label_from_ez,
    read_combo_csv,
)

_GS = Path(__file__).resolve().parents[1] / "ref_explore_plus_grid_search"
if str(_GS) not in sys.path:
    sys.path.insert(0, str(_GS))

from grid_search_ref40 import _build_dense, compute_episcore, compute_zscore  # noqa: E402

console = Console()

EP_COLS = ["hypo_z_intra", "hyper_z_intra", "hypo_cpgs_count", "hyper_cpgs_count"]
TARGET_DEFAULT = "chr21"


def _read_filtered_parquet(path: Path, samples: list[str], columns: list[str]) -> pd.DataFrame:
    wanted = set(samples)
    try:
        df = pd.read_parquet(path, columns=columns, filters=[("sample", "in", list(wanted))])
    except (TypeError, ValueError, NotImplementedError):
        df = pd.read_parquet(path, columns=columns)
        df = df[df["sample"].astype(str).isin(wanted)]
    df["sample"] = df["sample"].astype(str)
    df["chr"] = df["chr"].astype(str)
    df["threshold"] = df["threshold"].astype(float)
    df["recall"] = df["recall"].astype(float)
    return df


def _load_query_parquets(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.glob(pattern))
    if not paths:
        paths = sorted(p for p in root.rglob("*.parquet") if p.is_file())
    if not paths:
        raise click.ClickException(f"No query parquets under {root} (tried {pattern})")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["sample"] = df["sample"].astype(str)
    df["chr"] = df["chr"].astype(str)
    df["threshold"] = df["threshold"].astype(float)
    df["recall"] = df["recall"].astype(float)
    return df


def _score_at_combo(score_all: np.ndarray, combos: list[tuple[float, float]], combo: tuple[float, float]) -> np.ndarray:
    try:
        idx = combos.index(combo)
    except ValueError as exc:
        raise click.ClickException(f"Combo {combo} not in array") from exc
    return score_all[idx]


def _ez_from_ep_z(ep_vec: np.ndarray, z_vec: np.ndarray, ez_idx: np.ndarray) -> tuple[np.ndarray, float, float]:
    combined = ep_vec + z_vec
    ref = combined[ez_idx]
    with np.errstate(invalid="ignore"):
        mu = float(np.nanmean(ref))
        sd = float(np.nanstd(ref, ddof=0))
    mu = mu if np.isfinite(mu) else 0.0
    sd_safe = sd if sd > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ez = (combined - mu) / sd_safe
    return ez, mu, sd


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--units", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--units-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Dir with early_ref_samples.txt, ez_ref_samples.txt, *combos.csv")
@click.option("--query-ep-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Root containing thr_<t>/{unit}.parquet episcore grids")
@click.option("--query-z-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Dir containing {unit}.parquet zscore grids")
@click.option("--output-dir", required=True, type=click.Path(file_okay=False))
@click.option("--grid-input", default=str(DEFAULT_GRID_INPUT), type=click.Path(exists=True, file_okay=False))
@click.option("--target-chr", default=TARGET_DEFAULT, show_default=True)
@click.option("--ez-cutoff", default=EZ_CUTOFF, show_default=True, type=float)
@click.option("--top-n", default=50, show_default=True, type=int)
def main(
    units: str,
    units_dir: str,
    query_ep_dir: str,
    query_z_dir: str,
    output_dir: str,
    grid_input: str,
    target_chr: str,
    ez_cutoff: float,
    top_n: int,
) -> None:
    units_path = Path(units_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    console.rule("[bold blue]Search T21 epi/z combos")

    unit_df = pd.read_csv(units)
    unit_df["unit_id"] = unit_df["unit_id"].astype(str)
    query_ids = unit_df["unit_id"].tolist()
    early_ref = load_sample_list(units_path / "early_ref_samples.txt")
    ez_ref = load_sample_list(units_path / "ez_ref_samples.txt")
    if target_chr not in CHR_LIST:
        raise click.ClickException(f"target-chr must be one of {CHR_LIST}")
    target_hi = CHR_LIST.index(target_chr)

    console.print(
        f"  query={len(query_ids)} early_ref={len(early_ref)} ez_ref={len(ez_ref)} "
        f"target={target_chr} cutoff={ez_cutoff:g}"
    )

    console.print("[cyan]Loading reference rows from previous grid-search parquets[/cyan]")
    ref_names = list(dict.fromkeys(early_ref + ez_ref))
    grid = Path(grid_input)
    ep_ref = _read_filtered_parquet(
        grid / "episcore_grid_search.parquet",
        ref_names,
        ["sample", "chr", "threshold", "recall", *EP_COLS],
    )
    z_ref = _read_filtered_parquet(
        grid / "zscore_grid_search.parquet",
        ref_names,
        ["sample", "chr", "threshold", "recall", "percentage"],
    )
    z_ref["percentage"] = maybe_fraction(z_ref["percentage"].to_numpy())

    console.print("[cyan]Loading query grids[/cyan]")
    ep_q = _load_query_parquets(Path(query_ep_dir), "thr_*/*.parquet")
    z_q = _load_query_parquets(Path(query_z_dir), "*.parquet")
    miss_ep = sorted(set(query_ids) - set(ep_q["sample"].astype(str)))
    miss_z = sorted(set(query_ids) - set(z_q["sample"].astype(str)))
    if miss_ep or miss_z:
        raise click.ClickException(
            f"Query grids incomplete: missing episcore={miss_ep} zscore={miss_z}"
        )

    ep_all = pd.concat([ep_ref, ep_q], ignore_index=True)
    z_all = pd.concat([z_ref, z_q], ignore_index=True)

    samples = list(dict.fromkeys(ref_names + query_ids))
    sample_index = {s: i for i, s in enumerate(samples)}
    chr_index = {c: i for i, c in enumerate(CHR_LIST)}
    early_idx = np.array([sample_index[s] for s in early_ref], dtype=np.int64)
    ez_idx = np.array([sample_index[s] for s in ez_ref], dtype=np.int64)
    query_idx = np.array([sample_index[s] for s in query_ids], dtype=np.int64)

    console.print("[cyan]Building dense arrays (refs + query)[/cyan]")
    ep_combos, ep_arrays = _build_dense(ep_all, EP_COLS, sample_index, chr_index)
    z_combos, z_arrays = _build_dense(z_all, ["percentage"], sample_index, chr_index)
    console.print(f"  epi combos={len(ep_combos)} z combos={len(z_combos)}")

    episcore_all = compute_episcore(ep_arrays[0], ep_arrays[1], ep_arrays[2], ep_arrays[3], early_idx)
    zscore_all = compute_zscore(z_arrays[0], early_idx)

    # Restrict search to combos that are non-NaN for every query on the target chr.
    def _complete_combos(score_all, combos, hi):
        keep = []
        for i, combo in enumerate(combos):
            vals = score_all[i, hi, query_idx]
            if np.isfinite(vals).all():
                keep.append(combo)
        return keep

    ep_search = _complete_combos(episcore_all, ep_combos, target_hi)
    z_search = _complete_combos(zscore_all, z_combos, target_hi)
    if not ep_search or not z_search:
        raise click.ClickException(
            f"No complete query combos on {target_chr}: "
            f"epi={len(ep_search)} z={len(z_search)}"
        )
    console.print(f"  searchable epi={len(ep_search)} z={len(z_search)}")

    ep_index = {c: i for i, c in enumerate(ep_combos)}
    z_index = {c: i for i, c in enumerate(z_combos)}

    rows = []
    n_ep, n_z = len(ep_search), len(z_search)
    for ei, ep_c in enumerate(ep_search):
        ep_vec = episcore_all[ep_index[ep_c], target_hi, :]
        if ei % 50 == 0:
            console.print(f"  searching epi {ei + 1}/{n_ep}")
        for z_c in z_search:
            z_vec = zscore_all[z_index[z_c], target_hi, :]
            ez, mu, sd = _ez_from_ep_z(ep_vec, z_vec, ez_idx)
            q = ez[query_idx]
            if not np.isfinite(q).all():
                continue
            n_pass = int((q > ez_cutoff).sum())
            rows.append(
                {
                    "ep_threshold": ep_c[0],
                    "ep_recall": ep_c[1],
                    "z_threshold": z_c[0],
                    "z_recall": z_c[1],
                    "ez_mu": mu,
                    "ez_sd": sd,
                    "n_pass": n_pass,
                    "all_pass": n_pass == len(query_ids),
                    "min_ez": float(np.min(q)),
                    "median_ez": float(np.median(q)),
                    "mean_ez": float(np.mean(q)),
                    "recall_sum": ep_c[1] + z_c[1],
                    "thr_dist": abs(ep_c[0] - PRODUCTION_EP[0]) + abs(z_c[0] - PRODUCTION_Z[0]),
                    **{f"ez_{uid}": float(q[j]) for j, uid in enumerate(query_ids)},
                }
            )

    cand = pd.DataFrame(rows)
    cand = cand.sort_values(
        ["all_pass", "min_ez", "recall_sum", "thr_dist"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    cand.to_csv(out / "chr21_combo_candidates.tsv", sep="\t", index=False, float_format="%.6f")
    cand.head(top_n).to_csv(
        out / f"chr21_combo_top{top_n}.tsv", sep="\t", index=False, float_format="%.6f"
    )
    n_pass_all = int(cand["all_pass"].sum()) if not cand.empty else 0
    best = cand.iloc[0]
    chosen_ep = (float(best["ep_threshold"]), float(best["ep_recall"]))
    chosen_z = (float(best["z_threshold"]), float(best["z_recall"]))
    console.print(
        f"  passing combos (all batches ez>{ez_cutoff:g}): {n_pass_all}/{len(cand)}\n"
        f"  chosen epi={chosen_ep} z={chosen_z} min_ez={best['min_ez']:.3f}"
    )

    # Baseline + chosen per-chr scores. Other chr stay on production combo.
    def _chr_scores(ep_by_chr: dict, z_by_chr: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ep = np.empty((len(CHR_LIST), len(samples)), dtype=np.float64)
        z = np.empty_like(ep)
        ez = np.empty_like(ep)
        for hi, chrom in enumerate(CHR_LIST):
            ep[hi] = _score_at_combo(episcore_all, ep_combos, ep_by_chr[chrom])[hi]
            z[hi] = _score_at_combo(zscore_all, z_combos, z_by_chr[chrom])[hi]
            ez[hi], _, _ = _ez_from_ep_z(ep[hi], z[hi], ez_idx)
        return ep, z, ez

    prod_ep_map = {c: PRODUCTION_EP for c in CHR_LIST}
    prod_z_map = {c: PRODUCTION_Z for c in CHR_LIST}
    chosen_ep_map = dict(prod_ep_map)
    chosen_z_map = dict(prod_z_map)
    chosen_ep_map[target_chr] = chosen_ep
    chosen_z_map[target_chr] = chosen_z

    base_ep, base_z, base_ez = _chr_scores(prod_ep_map, prod_z_map)
    ch_ep, ch_z, ch_ez = _chr_scores(chosen_ep_map, chosen_z_map)

    def _batch_table(ep_mat, z_mat, ez_mat, tag: str) -> pd.DataFrame:
        recs = []
        for j, uid in enumerate(query_ids):
            si = query_idx[j]
            meta = unit_df.loc[unit_df["unit_id"] == uid].iloc[0]
            ez_vec = ez_mat[:, si]
            recs.append(
                {
                    "combo_set": tag,
                    "unit_id": uid,
                    "sample": meta["sample"],
                    "batch": meta["batch"],
                    "ff_before_mq": meta.get("ff_before_mq", np.nan),
                    "meta_pred_label": meta.get("pred_label", ""),
                    "pred_label": pred_label_from_ez(ez_vec),
                    f"{target_chr}_episcore": float(ep_mat[target_hi, si]),
                    f"{target_chr}_zscore": float(z_mat[target_hi, si]),
                    f"{target_chr}_ezscore": float(ez_vec[target_hi]),
                    f"{target_chr}_called": bool(ez_vec[target_hi] > ez_cutoff),
                    "max_other_ez": float(
                        np.nanmax(np.delete(ez_vec, target_hi))
                    ),
                    "n_chr_gt_cutoff": int(np.nansum(ez_vec > ez_cutoff)),
                    **{f"ez_{c}": float(ez_vec[i]) for i, c in enumerate(CHR_LIST)},
                    **{f"ep_{c}": float(ep_mat[i, si]) for i, c in enumerate(CHR_LIST)},
                    **{f"z_{c}": float(z_mat[i, si]) for i, c in enumerate(CHR_LIST)},
                }
            )
        return pd.DataFrame.from_records(recs)

    baseline_tbl = _batch_table(base_ep, base_z, base_ez, "production")
    chosen_tbl = _batch_table(ch_ep, ch_z, ch_ez, "chosen")
    baseline_tbl.to_csv(out / "baseline_per_batch.tsv", sep="\t", index=False, float_format="%.6f")
    chosen_tbl.to_csv(out / "chosen_per_batch.tsv", sep="\t", index=False, float_format="%.6f")
    pd.concat([baseline_tbl, chosen_tbl], ignore_index=True).to_csv(
        out / "per_batch_ezscore.tsv", sep="\t", index=False, float_format="%.6f"
    )

    long_rows = []
    for tbl, ep_map, z_map in (
        (baseline_tbl, prod_ep_map, prod_z_map),
        (chosen_tbl, chosen_ep_map, chosen_z_map),
    ):
        tag = tbl["combo_set"].iloc[0]
        for _, r in tbl.iterrows():
            for chrom in CHR_LIST:
                long_rows.append(
                    {
                        "combo_set": tag,
                        "unit_id": r["unit_id"],
                        "sample": r["sample"],
                        "batch": r["batch"],
                        "chr": chrom,
                        "ep_threshold": ep_map[chrom][0],
                        "ep_recall": ep_map[chrom][1],
                        "z_threshold": z_map[chrom][0],
                        "z_recall": z_map[chrom][1],
                        "episcore": r[f"ep_{chrom}"],
                        "zscore": r[f"z_{chrom}"],
                        "ezscore": r[f"ez_{chrom}"],
                    }
                )
    pd.DataFrame(long_rows).to_csv(
        out / "per_batch_ezscore_long.tsv", sep="\t", index=False, float_format="%.6f"
    )

    def _combo_csv(path: Path, ep_map, z_map, kind: str) -> None:
        if kind == "ep":
            pd.DataFrame(
                {
                    "chr": CHR_LIST,
                    "threshold": [ep_map[c][0] for c in CHR_LIST],
                    "recall": [ep_map[c][1] for c in CHR_LIST],
                }
            ).to_csv(path, index=False)
        else:
            pd.DataFrame(
                {
                    "chr": CHR_LIST,
                    "threshold": [z_map[c][0] for c in CHR_LIST],
                    "recall": [z_map[c][1] for c in CHR_LIST],
                }
            ).to_csv(path, index=False)

    _combo_csv(out / "best_combo_episcore.csv", chosen_ep_map, chosen_z_map, "ep")
    _combo_csv(out / "best_combo_zscore.csv", chosen_ep_map, chosen_z_map, "z")
    _combo_csv(out / "production_combo_episcore.csv", prod_ep_map, prod_z_map, "ep")
    _combo_csv(out / "production_combo_zscore.csv", prod_ep_map, prod_z_map, "z")

    lines = [
        "JPTAY T21 combo search",
        f"target_chr={target_chr} ez_cutoff={ez_cutoff:g}",
        f"early_ref_n={len(early_ref)} ez_ref_n={len(ez_ref)} query_batches={len(query_ids)}",
        f"searchable_epi_combos={len(ep_search)} searchable_z_combos={len(z_search)}",
        f"candidates={len(cand)} all_batches_pass={n_pass_all}",
        f"chosen_ep=threshold={fmt_combo(chosen_ep[0])} recall={fmt_combo(chosen_ep[1])}",
        f"chosen_z=threshold={fmt_combo(chosen_z[0])} recall={fmt_combo(chosen_z[1])}",
        f"chosen_min_{target_chr}_ez={best['min_ez']:.4f} all_pass={bool(best['all_pass'])}",
        "",
        "Per-batch calls:",
    ]
    for _, r in chosen_tbl.iterrows():
        b = baseline_tbl.loc[baseline_tbl["unit_id"] == r["unit_id"]].iloc[0]
        lines.append(
            f"  {r['unit_id']}  baseline {target_chr}_ez={b[f'{target_chr}_ezscore']:.3f} "
            f"pred={b['pred_label']}  ->  chosen {target_chr}_ez={r[f'{target_chr}_ezscore']:.3f} "
            f"pred={r['pred_label']}  called={r[f'{target_chr}_called']}"
        )
    if not bool(best["all_pass"]):
        lines.append(
            "\nWARNING: no combo put every batch above the cutoff. "
            "The chosen row is the max min-ez pair."
        )
    (out / "search_summary.txt").write_text("\n".join(lines) + "\n")
    console.print("\n".join(lines))
    console.print(f"[green]OK[/green] Wrote {out}")


if __name__ == "__main__":
    main()
