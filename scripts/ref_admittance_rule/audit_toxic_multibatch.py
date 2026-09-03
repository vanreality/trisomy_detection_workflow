#!/usr/bin/env python3
"""Per-batch MAD for multi-batch toxics + pool-level commonality tests.

1. Score each ``intermediate_each_batch_modeA`` unit against MAD fences frozen
   on the merged expanded Normal pool (same 3.5 cutoff as
   ``score_expanded_pool_mad.py``).
2. Compare toxic vs OK candidates on FF, coverage, pred_label mismatch, etc.

Writes under ``--outdir``:
  - ``toxic_multibatch_units.tsv``
  - ``pool_multibatch_units.tsv``
  - ``toxic_commonality_report.md``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import fisher_exact, mannwhitneyu

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import mad_z_vs_ref  # noqa: E402
from score_expanded_pool_mad import CHR_LIST, CUTOFF, _chr_block, _outlier_chrs  # noqa: E402

console = Console()

DEFAULT_OUTDIR = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
    "/expanded_pool_mad"
)
DEFAULT_EACH = Path(
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_each_batch_modeA.parquet"
)
DEFAULT_MERGED = Path(
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet"
)
DEFAULT_META = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv")


def _md_escape(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val).replace("\n", " ").replace("|", "\\|")


def _md_table(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    use = list(cols) if cols is not None else list(df.columns)
    header = "| " + " | ".join(use) + " |"
    sep = "| " + " | ".join("---" for _ in use) + " |"
    rows = [
        "| " + " | ".join(_md_escape(r[c]) for c in use) + " |"
        for _, r in df.iterrows()
    ]
    return [header, sep, *rows]


def name_kind(sample: str) -> str:
    s = str(sample)
    if re.fullmatch(r"HCPT\d+", s):
        return "HCPT"
    if re.fullmatch(r"(?:J)?PTAY\d+P$", s):
        return "legacy_P"
    if re.search(r"S\d+$", s):
        return "S_miscarriage"
    if re.search(r"H\d+$", s):
        return "H_pregnancy"
    return "other"


def score_units_vs_ref(units: pd.DataFrame, ref: pd.DataFrame, pool: list[str]) -> pd.DataFrame:
    ref_idx = ref.set_index("sample").reindex(pool)
    ref_pct = _chr_block(ref_idx, "percentage_after_mq")
    ref_hypo = _chr_block(ref_idx, "hypo_z_intra_after_mq")
    ref_hyper = _chr_block(ref_idx, "hyper_z_intra_after_mq")
    pct = mad_z_vs_ref(_chr_block(units, "percentage_after_mq"), ref_pct, axis=1)
    hypo = mad_z_vs_ref(_chr_block(units, "hypo_z_intra_after_mq"), ref_hypo, axis=1)
    hyper = mad_z_vs_ref(_chr_block(units, "hyper_z_intra_after_mq"), ref_hyper, axis=1)
    intra = np.maximum(np.abs(hypo), np.abs(hyper))
    abs_pct = np.abs(pct)
    rows = []
    for j, (_, r) in enumerate(units.iterrows()):
        pz = abs_pct[:, j]
        iz = intra[:, j]
        per = np.maximum(pz, iz)
        peak_i = int(np.nanargmax(per))
        pct_peak = float(pz[peak_i])
        intra_peak = float(iz[peak_i])
        hits = [
            CHR_LIST[i]
            for i, v in enumerate(per)
            if v >= CUTOFF
        ]
        rows.append(
            {
                "sample": r["sample"],
                "batch": str(r["batch"]),
                "ff_before_mq": r.get("ff_before_mq", np.nan),
                "mad_score": float(np.nanmax(per)),
                "max_mad_chr": CHR_LIST[peak_i],
                "max_mad_track": "percentage" if pct_peak >= intra_peak else "z_intra",
                "n_chr_ge_cutoff": len(hits),
                "outlier_chrs": ",".join(hits),
                "toxic_unit": bool(np.nanmax(per) >= CUTOFF),
            }
        )
    return pd.DataFrame(rows)


def fisher_bool(toxic: np.ndarray, flag: np.ndarray) -> tuple[str, float, float, int, int]:
    a = int((toxic & flag).sum())
    b = int((toxic & ~flag).sum())
    c = int((~toxic & flag).sum())
    d = int((~toxic & ~flag).sum())
    or_, p = fisher_exact([[a, b], [c, d]])
    n_t, n_o = a + b, c + d
    line = (
        f"toxic {a}/{n_t} ({a / n_t:.1%}) vs ok {c}/{n_o} ({c / n_o:.1%}); "
        f"OR={or_:.2f}, p={p:.4g}"
    )
    return line, or_, p, a, c


def write_report(
    *,
    tox_units: pd.DataFrame,
    pool_units: pd.DataFrame,
    cand: pd.DataFrame,
    out_md: Path,
) -> None:
    n_tox = int(cand["toxic"].sum())
    n_ok = int((~cand["toxic"]).sum())
    n_pool = len(cand)
    tox_samples = sorted(tox_units["sample"].unique())
    ok_sum = (
        pool_units.loc[~pool_units["toxic_sample"]]
        .groupby("sample")
        .agg(
            n_units=("batch", "size"),
            n_toxic_units=("toxic_unit", "sum"),
            min_score=("mad_score", "min"),
            max_score=("mad_score", "max"),
        )
        .reset_index()
    )
    mixed = ok_sum[(ok_sum["n_toxic_units"] > 0) & (ok_sum["n_toxic_units"] < ok_sum["n_units"])]

    lines = [
        "# Multi-batch MAD + toxic commonality",
        "",
        "Per-batch units from `intermediate_each_batch_modeA.parquet`, MAD-z vs "
        "fences frozen on the merged 273-Normal pool, cutoff 3.5.",
        "",
        "## 1. Multi-batch toxic samples",
        "",
        f"Only **{len(tox_samples)} / {n_tox}** toxic samples are multi-batch: "
        + ", ".join(f"`{s}`" for s in tox_samples)
        + ".",
        "",
        "Neither sample has a clean unit — every batch still scores toxic.",
        "",
    ]
    show = tox_units.copy()
    show["mad_score"] = show["mad_score"].map(lambda x: round(float(x), 3))
    lines.extend(
        _md_table(
            show.sort_values(["sample", "batch"]),
            [
                "sample",
                "batch",
                "ff_before_mq",
                "mad_score",
                "max_mad_chr",
                "max_mad_track",
                "n_chr_ge_cutoff",
                "merged_mad",
                "merged_chr",
            ],
        )
    )
    lines += [
        "",
        "- **`PTAY1445P7S1`**: sample-intrinsic. Both batches peak **chr1 percentage** "
        "(4.29 then 3.64); merged (3.63) tracks the later unit. No good/bad split.",
        "- **`PTAY0103P`**: all 3 batches toxic, but **which chromosome and how bad** "
        "moves. `20250627` peak chr10 (4.74, 8 chrs), `20250630` peak chr19 (4.87, 4 chrs), "
        "`20250703` peak chr4 z_intra (**9.33**, 14 chrs). Merged z_intra matches the "
        "latest batch (placeholder), so the published 9.33 is the worst unit. Earlier "
        "repeats are still ≥4.7 — not a clean batch.",
        "",
        f"Control: {len(ok_sum)} multi-batch **OK** pool samples; "
        f"**{len(mixed)}** are mixed (one unit ≥3.5, another below). Merging can hide a "
        "single bad batch for OK samples; it did not rescue the 2 toxics.",
        "",
    ]
    if not mixed.empty:
        lines.append("OK mixed units:")
        lines.append("")
        lines.extend(
            _md_table(
                mixed.sort_values("max_score", ascending=False),
                ["sample", "n_units", "n_toxic_units", "min_score", "max_score"],
            )
        )
        lines.append("")

    def row_line(title: str, text: str) -> None:
        lines.append(f"- **{title}:** {text}")

    tox = cand["toxic"].to_numpy(dtype=bool)
    lines += [
        "## 2. Anything in common?",
        "",
        f"Toxic n={n_tox} vs OK n={n_ok} (pool {n_pool}). Not a shared batch, chr, or CNV.",
        "",
        "### What does cluster",
        "",
    ]
    row_line("Low FF (<1%)", fisher_bool(tox, cand["low_ff"].to_numpy(dtype=bool))[0])
    med_t = float(cand.loc[cand["toxic"], "ff_before_mq"].median())
    med_o = float(cand.loc[~cand["toxic"], "ff_before_mq"].median())
    p_ff = mannwhitneyu(
        cand.loc[cand["toxic"], "ff_before_mq"].dropna(),
        cand.loc[~cand["toxic"], "ff_before_mq"].dropna(),
    ).pvalue
    lines.append(
        f"  Median FF {med_t:.4f} vs {med_o:.4f} (MWU p={p_ff:.4g}). Spearman(mad, FF)="
        f"{cand[['mad_score','ff_before_mq']].corr(method='spearman').iloc[0,1]:.2f}."
    )
    row_line(
        "pred_label ≠ Normal",
        fisher_bool(tox, (~cand["pred_normal"]).to_numpy(dtype=bool))[0],
    )
    row_line(
        "match_status = mismatch",
        fisher_bool(tox, cand["mismatch"].to_numpy(dtype=bool))[0],
    )
    for col, title in (
        ("snp_mean_coverage", "SNP coverage"),
        ("cpg_mean_coverage", "CpG coverage"),
    ):
        a = pd.to_numeric(cand.loc[cand["toxic"], col], errors="coerce").dropna()
        b = pd.to_numeric(cand.loc[~cand["toxic"], col], errors="coerce").dropna()
        p = mannwhitneyu(a, b).pvalue
        lines.append(
            f"- **{title}:** median {float(a.median()):.1f} vs {float(b.median()):.1f} "
            f"(MWU p={p:.4g}) — tracks with lower FF, not a separate chemistry story."
        )
    lines += [
        "",
        "The fixed-ref pipeline already treats many of these as Gray_T / mismatch "
        "despite the DB `Normal` label. Peak MAD chr agrees with a Gray/T chromosome "
        f"in only {int(cand.loc[cand['toxic'], 'peak_in_pred'].sum())}/"
        f"{int(cand.loc[cand['toxic'], 'has_t_pred'].sum())} such samples — not a "
        "simple 'same trisomy' cluster.",
        "",
        "### What does **not** cluster",
        "",
    ]
    row_line("Miscarriage (S) vs pregnancy (H)", fisher_bool(tox, cand["is_S"].to_numpy(dtype=bool))[0])
    row_line("TET (among known conception)", fisher_bool(
        cand.loc[cand["conc_known"], "toxic"].to_numpy(dtype=bool),
        cand.loc[cand["conc_known"], "is_TET"].to_numpy(dtype=bool),
    )[0])
    lines += [
        "- **Set:** toxic 16 dev / 30 test, same mix as the pool (Fisher p=1).",
        "- **Peak chromosome:** 20/22 chrs are a peak at least once (see expanded-pool REPORT).",
        "- **CNV+:** previously 1 overlapping donor; not the driver.",
        "- **Batch:** 17 latest mqres dates; `20260321` is only a soft pile-up (FDR ns).",
        "- **Label source:** mix of birth_outcome (22), nowhere (11), karyotype (9), keyword (4).",
        "",
        "## Takeaway",
        "",
        "Multi-batch does not split toxic vs clean for the 2 toxic donors — they are "
        "toxic in every sequenced batch. Pool-wide, toxics are **low-FF / lower-coverage "
        "Normals that the current caller already more often marks Gray_T or mismatch**. "
        "That is the shared signature; not a single lab batch or a single chromosome.",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option("--each-parquet", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=str(DEFAULT_EACH))
@click.option("--merged-parquet", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=str(DEFAULT_MERGED))
@click.option("--candidates", "cand_path", type=click.Path(path_type=Path, exists=True, dir_okay=False),
              default=str(DEFAULT_OUTDIR / "candidate_mad_scores.tsv"))
@click.option("--toxic", "toxic_path", type=click.Path(path_type=Path, exists=True, dir_okay=False),
              default=str(DEFAULT_OUTDIR / "toxic_samplesheet_with_db_labels.tsv"))
@click.option("--meta", "meta_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=str(DEFAULT_META))
@click.option("--outdir", type=click.Path(path_type=Path, file_okay=False), default=str(DEFAULT_OUTDIR))
def main(
    each_parquet: Path,
    merged_parquet: Path,
    cand_path: Path,
    toxic_path: Path,
    meta_path: Path,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    each = pd.read_parquet(each_parquet)
    merged = pd.read_parquet(merged_parquet)
    cand = pd.read_csv(cand_path, sep="\t")
    toxic = pd.read_csv(toxic_path, sep="\t")
    meta = pd.read_csv(meta_path)
    for df in (each, merged, cand, toxic, meta):
        df["sample"] = df["sample"].astype(str)

    pool = cand["sample"].tolist()
    tox = set(toxic["sample"].astype(str))
    n_per = each.groupby("sample").size()
    multi_pool = set(n_per[n_per >= 2].index) & set(pool)
    units = each.loc[each["sample"].isin(multi_pool)].copy().reset_index(drop=True)
    scored = score_units_vs_ref(units, merged, pool)
    scored["toxic_sample"] = scored["sample"].isin(tox)
    scored["merged_mad"] = scored["sample"].map(cand.set_index("sample")["mad_score"])
    scored["merged_chr"] = scored["sample"].map(cand.set_index("sample")["max_mad_chr"])
    scored["merged_track"] = scored["sample"].map(cand.set_index("sample")["max_mad_track"])

    tox_units = scored.loc[scored["toxic_sample"]].sort_values(["sample", "batch"])
    tox_units.to_csv(outdir / "toxic_multibatch_units.tsv", sep="\t", index=False, float_format="%.6f")
    scored.to_csv(outdir / "pool_multibatch_units.tsv", sep="\t", index=False, float_format="%.6f")

    meta_s = meta.set_index("sample")
    for col in (
        "state",
        "conception_mode",
        "ff_before_mq",
        "mean_target_coverage",
        "cpg_mean_coverage",
        "snp_mean_coverage",
        "pred_label",
        "match_status",
    ):
        if col in meta.columns and col not in cand.columns:
            cand[col] = cand["sample"].map(lambda s, c=col: meta_s.at[s, c] if s in meta_s.index else np.nan)
        elif col in meta.columns:
            cand[col] = cand[col].where(
                cand[col].notna(),
                cand["sample"].map(lambda s, c=col: meta_s.at[s, c] if s in meta_s.index else np.nan),
            )
    if "ff_before_mq" in cand.columns and "ff_before_mq_x" in cand.columns:
        pass
    cand["toxic"] = cand["sample"].isin(tox)
    cand["name_kind"] = cand["sample"].map(name_kind)
    cand["is_S"] = cand["name_kind"].eq("S_miscarriage")
    cand["is_TET"] = cand["conception_mode"].astype(str).str.upper().eq("TET")
    cand["conc_known"] = cand["conception_mode"].notna() & ~cand["conception_mode"].astype(str).isin(
        ["nan", "None", ""]
    )
    cand["low_ff"] = pd.to_numeric(cand["ff_before_mq"], errors="coerce") < 0.01
    cand["pred_normal"] = cand["pred_label"].astype(str).eq("Normal")
    cand["mismatch"] = cand["match_status"].astype(str).eq("mismatch")

    def gray_chrs(pred) -> set[str]:
        return {f"chr{int(p)}" for p in re.findall(r"(?:Gray_)?T(\d+)", str(pred))}

    cand["has_t_pred"] = cand["pred_label"].map(lambda p: bool(gray_chrs(p)))
    cand["peak_in_pred"] = [
        r.max_mad_chr in gray_chrs(r.pred_label) if r.has_t_pred else False
        for r in cand.itertuples(index=False)
    ]

    write_report(
        tox_units=tox_units,
        pool_units=scored,
        cand=cand,
        out_md=outdir / "toxic_commonality_report.md",
    )
    console.print(
        f"multi toxic samples={tox_units['sample'].nunique()} "
        f"units={len(tox_units)} all_toxic={bool(tox_units['toxic_unit'].all())}"
    )
    console.print(f"[green]OK[/green] wrote {outdir / 'toxic_commonality_report.md'}")


if __name__ == "__main__":
    main()
