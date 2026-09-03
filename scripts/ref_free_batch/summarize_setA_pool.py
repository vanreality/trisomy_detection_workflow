#!/usr/bin/env python3
"""Write a markdown summary of Set A pool-size exploration to the result root."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_POOL,
    DROP_SAMPLES,
    EZ_CUTOFFS,
    FF_MIN,
    MODES,
    PURITY_MIN,
    ez_ratio_col,
)

console = Console()


def _is_t(s: str) -> bool:
    return bool(re.match(r"^T\d", str(s)))


def _prep(df: pd.DataFrame, high_ff: bool) -> pd.DataFrame:
    out = df.copy()
    out["ff_before_mq"] = pd.to_numeric(out["ff_before_mq"], errors="coerce")
    out["purity"] = pd.to_numeric(out["purity"], errors="coerce")
    out = out.loc[out["ff_before_mq"].notna()].copy()
    if high_ff:
        out = out.loc[out["ff_before_mq"] >= FF_MIN].copy()
    else:
        out = out.loc[out["ff_before_mq"] < FF_MIN].copy()
    out["is_trisomy"] = out["label"].map(_is_t)
    out["is_special"] = out["purity"] < PURITY_MIN
    return out


def _stats_row(df: pd.DataFrame, col: str) -> dict:
    n = df.loc[~df["is_trisomy"], col]
    t = df.loc[df["is_trisomy"], col]
    return {
        "n_N": int(n.size),
        "n_T": int(t.size),
        "n_special": int(df["is_special"].sum()),
        "median_N": float(n.median()) if n.size else float("nan"),
        "median_T": float(t.median()) if t.size else float("nan"),
        "mean_N": float(n.mean()) if n.size else float("nan"),
        "mean_T": float(t.mean()) if t.size else float("nan"),
        "n_N_ge_0.5": int((n >= 0.5).sum()) if n.size else 0,
        "n_T_lt_0.5": int((t < 0.5).sum()) if t.size else 0,
    }


def _fmt(x: float) -> str:
    if x != x:
        return "NA"
    return f"{x:.3f}"


def _load_pools(sweep_dir: Path) -> dict[int, pd.DataFrame]:
    by = {}
    for pdir in sorted(sweep_dir.glob("pool_*")):
        tsv = pdir / "abnormality_signal_ratio.tsv"
        if tsv.is_file():
            by[int(pdir.name.split("_")[1])] = pd.read_csv(tsv, sep="\t")
    return by


def _mode_section(mode: str, result_root: Path) -> list[str]:
    sweep = result_root / mode
    by = _load_pools(sweep)
    if not by:
        return [f"## {mode}", "", f"No pool TSVs under `{sweep}`.", ""]
    pools = sorted(by)
    cfg_path = sweep / f"pool_{pools[0]}" / "run_config.json"
    repeats = json.loads(cfg_path.read_text())["total_repeats"] if cfg_path.is_file() else "?"
    lines = [
        f"## {mode}",
        "",
        f"Pools {pools[0]}–{pools[-1]} (n={len(pools)}, step={pools[1] - pools[0] if len(pools) > 1 else '?'}), "
        f"{repeats} repeats/pool. Combo: "
        f"ep {MODES[mode]['ep_threshold']}/{MODES[mode]['ep_recall']}, "
        f"z {MODES[mode]['z_threshold']}/{MODES[mode]['z_recall']}.",
        "",
    ]
    focus = DEFAULT_POOL if DEFAULT_POOL in by else pools[len(pools) // 2]
    for high, tag in ((True, "ff≥1%"), (False, "ff<1%")):
        d0 = _prep(by[focus], high_ff=high)
        if d0.empty:
            continue
        lines.append(f"### {tag}")
        lines.append("")
        n_n = int((~d0["is_trisomy"]).sum())
        n_t = int(d0["is_trisomy"].sum())
        n_sp = int(d0["is_special"].sum())
        lines.append(
            f"At ref {focus // 2}+{focus // 2}: plotted **N={n_n}, T={n_t}** "
            f"({n_sp} with purity<{PURITY_MIN:g}, drawn as blue diamonds; counted in N/T)."
        )
        lines.append("")
        lines.append(
            "| pool | ref | ez | N | T | median N | median T | N≥0.5 (FP) | T<0.5 (FN) |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        show_pools = [p for p in pools if p in {20, 40, 80, 96, 120, 160} or p == focus]
        if not show_pools:
            show_pools = pools[:: max(1, len(pools) // 6)]
        for p in show_pools:
            df = _prep(by[p], high_ff=high)
            for cut in EZ_CUTOFFS:
                st = _stats_row(df, ez_ratio_col(cut))
                lines.append(
                    f"| {p} | {p // 2}+{p // 2} | {cut:g} | {st['n_N']} | {st['n_T']} | "
                    f"{_fmt(st['median_N'])} | {_fmt(st['median_T'])} | "
                    f"{st['n_N_ge_0.5']} | {st['n_T_lt_0.5']} |"
                )
        # trend: median T-N gap at ez=3 vs 4.5 across pools (ff split)
        lines.append("")
        for cut in EZ_CUTOFFS:
            gaps = []
            fps = []
            fns = []
            for p in pools:
                df = _prep(by[p], high_ff=high)
                st = _stats_row(df, ez_ratio_col(cut))
                if st["n_N"] and st["n_T"]:
                    gaps.append((p, st["median_T"] - st["median_N"]))
                    fps.append((p, st["n_N_ge_0.5"]))
                    fns.append((p, st["n_T_lt_0.5"]))
            if gaps:
                best = max(gaps, key=lambda x: x[1])
                lines.append(
                    f"- ez_cutoff={cut:g}: median(T)−median(N) ranges "
                    f"{min(g for _, g in gaps):.3f}–{max(g for _, g in gaps):.3f}; "
                    f"largest gap at pool={best[0]} (ref {best[0] // 2}+{best[0] // 2}, gap={best[1]:.3f}). "
                    f"FP (N ratio≥0.5) {min(v for _, v in fps)}–{max(v for _, v in fps)}; "
                    f"FN (T ratio<0.5) {min(v for _, v in fns)}–{max(v for _, v in fns)}."
                )
        html = f"plots/{mode}_SetA_{'ff_ge_1pct' if high else 'ff_lt_1pct'}_pool_size.html"
        lines.append("")
        lines.append(f"Interactive plot: `{html}`")
        lines.append("")
        if n_sp:
            spec = d0.loc[d0["is_special"], ["orig_sample", "label", "ff_before_mq", "purity"]]
            spec = spec.sort_values("ff_before_mq")
            lines.append(f"purity<{PURITY_MIN:g} in this panel (ref {focus // 2}+{focus // 2}):")
            lines.append("")
            lines.append("| sample | label | ff | purity |")
            lines.append("|---|---|---:|---:|")
            for _, r in spec.iterrows():
                lines.append(
                    f"| {r['orig_sample']} | {r['label']} | {r['ff_before_mq']:.4f} | {r['purity']:.3f} |"
                )
            lines.append("")
    return lines


def _takeaways(result_root: Path) -> list[str]:
    """Short interpretation at ref 40+40 (pool=80) for both modes / ff splits."""
    lines = ["## Takeaways", ""]
    focus = DEFAULT_POOL

    def _st(mode: str, high: bool, cut: float) -> dict | None:
        by = _load_pools(result_root / mode)
        if focus not in by:
            return None
        df = _prep(by[focus], high_ff=high)
        return _stats_row(df, ez_ratio_col(cut))

    a_hi3 = _st("modeA", True, 3.0)
    a_hi45 = _st("modeA", True, 4.5)
    b_hi3 = _st("modeB", True, 3.0)
    b_hi45 = _st("modeB", True, 4.5)
    a_lo3 = _st("modeA", False, 3.0)
    a_lo45 = _st("modeA", False, 4.5)
    b_lo3 = _st("modeB", False, 3.0)
    if not a_hi3:
        return lines

    lines.append(
        f"At **ref {focus // 2}+{focus // 2}**, plotted **N={a_hi3['n_N']}, T={a_hi3['n_T']}** "
        f"({a_hi3['n_special']} with purity<{PURITY_MIN:g}). "
        "Set A is defined as ff≥1%, so there is no ff<1% panel."
    )
    lines.append("")
    lines.append(
        f"- **ff≥1% separates well in both modes.** Median T ez-ratio is "
        f"{_fmt(a_hi3['median_T'])} (modeA) / {_fmt(b_hi3['median_T'])} (modeB) at ez=3, "
        f"with median N {_fmt(a_hi3['median_N'])} / {_fmt(b_hi3['median_N'])}. "
        f"ez_cutoff=4.5 drives FP (Normal ratio≥0.5) to {a_hi45['n_N_ge_0.5']} / {b_hi45['n_N_ge_0.5']} "
        f"but leaves FN={a_hi45['n_T_lt_0.5']} / {b_hi45['n_T_lt_0.5']}; "
        f"ez_cutoff=3 keeps FN at {a_hi3['n_T_lt_0.5']} / {b_hi3['n_T_lt_0.5']} with more FP "
        f"(modeA {a_hi3['n_N_ge_0.5']}, modeB {b_hi3['n_N_ge_0.5']}). "
        "Growing the pool past 40+40 mainly trims remaining FP at ez=3; it does not recover FN at ez=4.5."
    )
    lines.append(
        f"- **modeB vs modeA at ez=3:** FP {b_hi3['n_N_ge_0.5']} vs {a_hi3['n_N_ge_0.5']}; "
        f"FN {b_hi3['n_T_lt_0.5']} vs {a_hi3['n_T_lt_0.5']}. "
        f"At ez=4.5 FP={a_hi45['n_N_ge_0.5']}/{b_hi45['n_N_ge_0.5']}; "
        f"FN modeA {a_hi45['n_T_lt_0.5']} vs modeB {b_hi45['n_T_lt_0.5']}."
    )
    if a_lo3 and a_lo3["n_N"] + a_lo3["n_T"] > 0:
        lines.append(
            f"- **ff<1%** (unexpected in this cohort): modeA ez=3 T {_fmt(a_lo3['median_T'])} "
            f"vs N {_fmt(a_lo3['median_N'])}."
        )
    lines.append(
        "- **N in the title shrinks for pool>96** because nested test-Normal "
        "fillers leave eval. T is unchanged (fillers are Normal)."
    )
    lines.append(
        f"- **{DROP_SAMPLES[0]}** was dropped as requested and is absent from all HTMLs."
    )
    lines.append("")
    return lines


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--result-root",
    default=str(DEFAULT_OUT),
    type=click.Path(file_okay=False, path_type=Path),
)
def main(result_root: Path) -> None:
    result_root = Path(result_root)
    lines = [
        "# Set A pool-size exploration (20260813)",
        "",
        "Combines the 20260810 part1c interactive pool-size view with the 20260811 "
        "batch-QC Set A cohort and modeA/modeB score combos.",
        "",
        f"Result root: `{result_root}`",
        "",
        "## Design",
        "",
        "- **Set A (rebuilt from meta, not mqres-preferred units):** "
        "`meta_samplesheet.csv` with `depth_qc==pass`, `set ∈ {dev,test}`, "
        "`ff_before_mq ≥ 0.01`, `label` not in {Unknown, XO, Twin, M21}, "
        "and `label` does not contain `,`. Actual n = 294 (dev=100, test=194). "
        "Test is 194 rather than 198 because excluding Unknown/XO/Twin/M21 leaves 197 "
        "and dropping 3 comma labels (T21,T22 / T16,T22 / T7,T20) leaves 194.",
        "- **Why the old Set A was too small:** the 20260811 batch-QC Set A was "
        "`Normal|T*` + preferred unit only + **no FF filter**. Multi-batch samples "
        "with missing `preferred_batch_key` (n=18) never got a preferred unit, so they "
        "were only in Set D. Eval then also dropped the 96-ref overlap and samples "
        "without unit ep/z, leaving 158 plotted ff≥1% points vs ~230 after the meta filter.",
        f"- **Dropped from eval/plots:** `{DROP_SAMPLES[0]}`. The 96-dev-Normal reference "
        "pool is not plotted (same as part1c); 63 of those refs also pass the ff≥1% filter.",
        "- **Reference pool:** 96 dev Normal from `20260621-ref_40_rebuild_consider_lib_ng`. "
        "When pool>96, nested test-Normal fillers are added (fill_seed=7) and matching "
        "Set A eval samples leave the plot, so N in the title can shrink.",
        "- **Modes:** modeA ep 0.5/0.65 + z 0.85/0.95; modeB ep 0.1/0.61 + z 0.9/0.92. "
        "Eval scores: batch-QC unit files, falling back to main parquet if unit episcore is missing.",
        "- **Plots:** one HTML per mode (Set A is already ff≥1%). Controls: pool-size slider, "
        "Play/Pause (280 ms/frame, loops), ez_cutoff buttons (3 and 4.5).",
        "- **Style:** Normal = gray circles; trisomy = red circles; **purity<0.8 = blue diamonds**. "
        "Title shows `ref {n}+{n}`, `ez_cutoff`, and `N=…, T=…` only.",
        "",
        "## Plots",
        "",
        "| File | Mode | FF |",
        "|---|---|---|",
        "| `plots/modeA_SetA_ff_ge_1pct_pool_size.html` | modeA | ≥1% |",
        "| `plots/modeB_SetA_ff_ge_1pct_pool_size.html` | modeB | ≥1% |",
        "",
        "## Cohort notes",
        "",
    ]
    filt = result_root / "cohort" / "set_A_filter.txt"
    if filt.is_file():
        lines.append("**Set A filter (from meta_samplesheet):**")
        lines.append("")
        lines.append("```")
        lines.append(filt.read_text().rstrip())
        lines.append("```")
        lines.append("")
    # prepare summaries if present
    for mode in MODES:
        prep = result_root / mode / "input" / "prepare_summary.txt"
        if prep.is_file():
            lines.append(f"**{mode} prepare:**")
            lines.append("")
            lines.append("```")
            lines.append(prep.read_text().rstrip())
            lines.append("```")
            lines.append("")
        miss = result_root / mode / "input" / "missing_units.txt"
        if miss.is_file():
            n_miss = len([x for x in miss.read_text().splitlines() if x.strip()])
            if n_miss:
                lines.append(
                    f"{mode}: {n_miss} Set A samples lacked complete ep+z even after "
                    "main-parquet fallback and were skipped."
                )
                lines.append("")

    for mode in MODES:
        lines.extend(_mode_section(mode, result_root))

    lines.extend(_takeaways(result_root))

    lines.extend(
        [
            "## How to refresh",
            "",
            "```bash",
            "cd /lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/ref_free_batch",
            "bash submit_setA_pool_plots.sh",
            "```",
            "",
            "Scripts: `build_setA_cohort.py`, `prepare_setA_assets.py`, "
            "`pool_size_setA_sweep.py`, `plot_pool_size_setA.py`, `summarize_setA_pool.py`.",
            "",
        ]
    )
    out = result_root / "README.md"
    out.write_text("\n".join(lines))
    console.print(f"[green]wrote[/green] {out}")


if __name__ == "__main__":
    main()
