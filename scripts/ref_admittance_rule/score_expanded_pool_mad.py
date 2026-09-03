#!/usr/bin/env python3
"""MAD-score the expanded Normal candidate pool and overlap with CNV+.

Candidate pool: meta ``set ∈ {dev, test}``, ``depth_qc == pass``, ``label == Normal``.
Per-chr MAD-z is computed **inside this pool** on modeA after-MQ percentage and
hypo/hyper z_intra. Sample score = max(|pct MAD-z|, |z_intra MAD-z|) over chr1–22.

Toxic dividing line: score ≥ 3.5 (same robust-z fence as the 96-pool screen).

CNV+: CNVseq event with length > 10 Mb **and** overlap with the 220k target CpG panel.
"""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rich.console import Console

from common import CHR_LIST, MAD_K, mad_z, try_fisher

console = Console()
DEFAULT_CPG = Path(
    "/lustre1/cqyi/AIPT_2.0/workflow/episcore/assets/220k_cpg_recall_list/220k_cpg_recall_0.65.txt"
)
MIN_CNV_MB = 10.0

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
T_TAG = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"

DEV_OK = "#3D7A5A"
TEST_OK = "#7EB8BE"
DEV_TOXIC = "#9B2226"
TEST_TOXIC = "#E07A3D"
CUTOFF = 3.5

DEFAULT_PARQUET = Path(
    "/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/intermediate_merged_batches_modeA.parquet"
)
DEFAULT_META = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv")
DEFAULT_CNV = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
    "/PTAY_cnvseq_cnv_stat.check.20260814.xlsx"
)
DEFAULT_OUT = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
    "/expanded_pool_mad"
)


def donor_core(sample: object) -> str:
    """Stable donor key: optional J + PTAY + zero-padded integer id."""
    m = re.search(r"(J?PTAY)(\d+)", str(sample), flags=re.I)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2).zfill(4)}"


def _colrow(ref: str) -> tuple[int, int]:
    letters = "".join(c for c in ref if c.isalpha())
    row = int("".join(c for c in ref if c.isdigit()))
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1, row - 1


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    """Minimal first-sheet reader (no openpyxl)."""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join((t.text or "") for t in si.iter(T_TAG)))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        cells: dict[int, dict[int, str]] = {}
        max_c = max_r = 0
        for c in sheet.findall(".//m:c", NS):
            ref = c.get("r")
            if not ref:
                continue
            ci, ri = _colrow(ref)
            max_c, max_r = max(max_c, ci), max(max_r, ri)
            v = c.find("m:v", NS)
            is_el = c.find("m:is", NS)
            ctype = c.get("t")
            if ctype == "s" and v is not None and v.text is not None:
                val = shared[int(v.text)]
            elif ctype == "inlineStr" and is_el is not None:
                val = "".join((t.text or "") for t in is_el.iter(T_TAG))
            elif v is not None:
                val = v.text or ""
            else:
                val = ""
            cells.setdefault(ri, {})[ci] = val
    header = [str(cells.get(0, {}).get(i, f"col{i}")) for i in range(max_c + 1)]
    records = []
    for r in range(1, max_r + 1):
        row = cells.get(r, {})
        if not row:
            continue
        records.append({header[i]: row.get(i, "") for i in range(len(header))})
    return pd.DataFrame(records)


def parse_length_mb(val: object, start: object = None, end: object = None) -> float:
    """Length in Mb from the CNVseq ``Length`` field, else (End-Start)/1e6."""
    s = str(val).strip().replace(",", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*Mb$", s, flags=re.I)
    if m:
        return float(m.group(1))
    try:
        st = float(start)
        en = float(end)
        if np.isfinite(st) and np.isfinite(en) and en > st:
            return (en - st) / 1e6
    except (TypeError, ValueError):
        pass
    return float("nan")


def _chr_token(val: object) -> str:
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    if s.lower().startswith("chr"):
        return "chr" + re.sub(r"\.0$", "", s[3:])
    s = re.sub(r"\.0$", "", s)
    return f"chr{s}"


def load_cpg_positions(path: Path) -> dict[str, np.ndarray]:
    cpg = pd.read_csv(path, sep="\t", usecols=["chr", "start"])
    cpg["chr"] = cpg["chr"].astype(str)
    cpg["start"] = pd.to_numeric(cpg["start"], errors="coerce")
    cpg = cpg.dropna(subset=["start"])
    out: dict[str, np.ndarray] = {}
    for chrom, g in cpg.groupby("chr", sort=False):
        out[str(chrom)] = np.sort(g["start"].to_numpy(dtype=np.int64))
    return out


def n_cpg_in_interval(pos: dict[str, np.ndarray], chrom: str, start: float, end: float) -> int:
    arr = pos.get(chrom)
    if arr is None or arr.size == 0 or not np.isfinite(start) or not np.isfinite(end):
        return 0
    lo = int(start)
    hi = int(end)
    if hi < lo:
        lo, hi = hi, lo
    i0 = int(np.searchsorted(arr, lo, side="left"))
    i1 = int(np.searchsorted(arr, hi, side="right"))
    return max(0, i1 - i0)


def annotate_cnv_events(
    cnv: pd.DataFrame,
    cpg_pos: dict[str, np.ndarray],
    *,
    min_mb: float,
) -> pd.DataFrame:
    out = cnv.copy()
    out["chr"] = out["Chr"].map(_chr_token)
    out["start_bp"] = pd.to_numeric(out["Start"], errors="coerce")
    out["end_bp"] = pd.to_numeric(out["End"], errors="coerce")
    out["length_mb"] = [
        parse_length_mb(length, start, end)
        for length, start, end in zip(out["Length"], out["start_bp"], out["end_bp"])
    ]
    out["n_target_cpgs"] = [
        n_cpg_in_interval(cpg_pos, chrom, st, en)
        for chrom, st, en in zip(out["chr"], out["start_bp"], out["end_bp"])
    ]
    out["pass_length"] = out["length_mb"] > min_mb
    out["pass_target"] = out["n_target_cpgs"] > 0
    out["cnv_event_keep"] = out["pass_length"] & out["pass_target"]
    return out


def _chr_block(df: pd.DataFrame, kind: str) -> np.ndarray:
    cols = [f"{c}_{kind}" for c in CHR_LIST]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise click.ClickException(f"parquet missing columns: {missing[:3]}")
    return df[cols].to_numpy(dtype=float).T  # chr x sample


def _outlier_chrs(abs_z: np.ndarray, cutoff: float) -> str:
    hits = [CHR_LIST[i] for i, v in enumerate(abs_z) if v > cutoff]
    return ",".join(hits)


def score_pool(mat: pd.DataFrame, samples: list[str]) -> pd.DataFrame:
    sub = mat.set_index("sample").reindex(samples)
    if sub.isna().all(axis=None):
        raise click.ClickException("no overlapping samples between meta and parquet")
    pct = _chr_block(sub, "percentage_after_mq")
    hypo = _chr_block(sub, "hypo_z_intra_after_mq")
    hyper = _chr_block(sub, "hyper_z_intra_after_mq")
    pct_z = mad_z(pct, axis=1)
    hypo_z = mad_z(hypo, axis=1)
    hyper_z = mad_z(hyper, axis=1)
    intra = np.maximum(np.abs(hypo_z), np.abs(hyper_z))
    abs_pct = np.abs(pct_z)
    rows = []
    for j, sample in enumerate(samples):
        pz = abs_pct[:, j]
        iz = intra[:, j]
        per_chr = np.maximum(pz, iz)
        peak_i = int(np.nanargmax(per_chr))
        pct_peak = float(pz[peak_i])
        intra_peak = float(iz[peak_i])
        rows.append(
            {
                "sample": sample,
                "max_abs_pct_madz": float(np.nanmax(pz)),
                "max_abs_intra_madz": float(np.nanmax(iz)),
                "mad_score": float(np.nanmax(per_chr)),
                "max_mad_chr": CHR_LIST[peak_i],
                "max_mad_track": "percentage" if pct_peak >= intra_peak else "z_intra",
                "outlier_chrs_pct": _outlier_chrs(pz, CUTOFF),
                "outlier_chrs_intra": _outlier_chrs(iz, CUTOFF),
            }
        )
        for i, chr_name in enumerate(CHR_LIST):
            rows[-1][f"{chr_name}_pct_madz"] = float(pct_z[i, j])
            rows[-1][f"{chr_name}_hypo_madz"] = float(hypo_z[i, j])
            rows[-1][f"{chr_name}_hyper_madz"] = float(hyper_z[i, j])
    return pd.DataFrame(rows)


def _group_color(set_name: str, toxic: bool) -> str:
    if set_name == "dev":
        return DEV_TOXIC if toxic else DEV_OK
    return TEST_TOXIC if toxic else TEST_OK


def plot_score_bars(df: pd.DataFrame, out: Path) -> None:
    ranked = df.sort_values(["mad_score", "sample"], kind="mergesort").reset_index(drop=True)
    colors = [_group_color(s, t) for s, t in zip(ranked["set"], ranked["toxic"])]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(ranked))
    ax.bar(x, ranked["mad_score"], width=1.0, color=colors, linewidth=0, align="edge")
    cnv = ranked["cnv_positive"].to_numpy(dtype=bool)
    if cnv.any():
        ax.scatter(
            x[cnv] + 0.5,
            ranked.loc[cnv, "mad_score"].to_numpy() + 0.08,
            marker="v",
            s=18,
            c="#111",
            zorder=4,
            label="CNV+",
        )
    ax.axhline(CUTOFF, color="#222", ls="--", lw=1.1, zorder=3)
    ax.text(
        0.99,
        CUTOFF + 0.12,
        f"toxic  ≥ {CUTOFF:g}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#222",
    )
    ax.set_xlim(0, len(ranked))
    ax.set_ylim(0, float(ranked["mad_score"].max()) * 1.12)
    ax.set_xlabel("Candidate samples (ordered by ascending MAD score)")
    ax.set_ylabel("MAD score")
    ax.set_title(
        f"Expanded Normal pool (n={len(ranked)}): MAD score by set × toxic"
    )
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def _inset_pie(bounds: list[float], set_name: str, ok_c: str, tox_c: str) -> None:
        sub = ranked.loc[ranked["set"] == set_name]
        n_ok = int((~sub["toxic"]).sum())
        n_tox = int(sub["toxic"].sum())
        n = n_ok + n_tox

        def _count_label(pct: float) -> str:
            return str(int(round(pct / 100.0 * n)))

        inset = ax.inset_axes(bounds)
        _, _, autotexts = inset.pie(
            [n_ok, n_tox],
            colors=[ok_c, tox_c],
            startangle=90,
            counterclock=False,
            autopct=_count_label,
            pctdistance=0.55,
            textprops={"fontsize": 8, "fontweight": "bold"},
            wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
        )
        autotexts[0].set_color("#111")
        autotexts[1].set_color("white")
        inset.set_title(set_name, fontsize=9, pad=2)
        inset.annotate(
            f"ok {n_ok}/{n} ({n_ok / n:.1%})\ntoxic {n_tox}/{n} ({n_tox / n:.1%})",
            xy=(0, 0),
            xytext=(0, -1.32),
            ha="center",
            va="top",
            fontsize=7.5,
            annotation_clip=False,
        )

    _inset_pie([0.05, 0.52, 0.18, 0.38], "dev", DEV_OK, DEV_TOXIC)
    _inset_pie([0.26, 0.52, 0.18, 0.38], "test", TEST_OK, TEST_TOXIC)

    handles = [
        Patch(facecolor=DEV_OK, label="dev_ok"),
        Patch(facecolor=TEST_OK, label="test_ok"),
        Patch(facecolor=DEV_TOXIC, label="dev_toxic"),
        Patch(facecolor=TEST_TOXIC, label="test_toxic"),
        Line2D([0], [0], color="#222", ls="--", lw=1.1, label=f"cutoff {CUTOFF:g}"),
        Line2D(
            [0],
            [0],
            marker="v",
            color="none",
            markerfacecolor="#111",
            markersize=6,
            label="CNV+ (>10 Mb ∩ panel)",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    console.print(f"  wrote {out}")


def _chr_abs_mad(frame: pd.DataFrame, chr_name: str) -> np.ndarray:
    return np.maximum(
        np.abs(frame[f"{chr_name}_pct_madz"].to_numpy(dtype=float)),
        np.maximum(
            np.abs(frame[f"{chr_name}_hypo_madz"].to_numpy(dtype=float)),
            np.abs(frame[f"{chr_name}_hyper_madz"].to_numpy(dtype=float)),
        ),
    )


def peak_chr_uniform_gof(counts: np.ndarray, n_sim: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Monte-Carlo χ² goodness-of-fit of peak-chr counts vs uniform 1/22."""
    counts = np.asarray(counts, dtype=float)
    n = int(counts.sum())
    k = counts.size
    expected = np.full(k, n / k if k else 0.0)
    chi_obs = float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1e-12)))
    if n == 0:
        return chi_obs, float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, np.full(k, 1.0 / k), size=n_sim)
    chi_sim = np.sum((draws - expected) ** 2 / expected, axis=1)
    p = float((1.0 + np.sum(chi_sim >= chi_obs - 1e-12)) / (1.0 + n_sim))
    return chi_obs, p


def toxic_chr_stats(df: pd.DataFrame) -> pd.DataFrame:
    tox = df.loc[df["toxic"]].copy()
    n_tox = len(tox)
    expected = n_tox / len(CHR_LIST) if n_tox else 0.0
    rows = []
    for chr_name in CHR_LIST:
        per = _chr_abs_mad(tox, chr_name) if n_tox else np.array([], dtype=float)
        is_peak = tox["max_mad_chr"] == chr_name if n_tox else pd.Series(dtype=bool)
        ge = per >= CUTOFF if n_tox else np.array([], dtype=bool)
        rows.append(
            {
                "chr": chr_name,
                "n_toxic": n_tox,
                "expected_peak": expected,
                "n_peak": int(is_peak.sum()) if n_tox else 0,
                "n_peak_dev": int((is_peak & (tox["set"] == "dev")).sum()) if n_tox else 0,
                "n_peak_test": int((is_peak & (tox["set"] == "test")).sum()) if n_tox else 0,
                "n_ge_cutoff": int(ge.sum()) if n_tox else 0,
                "n_ge_cutoff_dev": int(((tox["set"] == "dev").to_numpy() & ge).sum()) if n_tox else 0,
                "n_ge_cutoff_test": int(((tox["set"] == "test").to_numpy() & ge).sum()) if n_tox else 0,
                "frac_peak": float(is_peak.mean()) if n_tox else 0.0,
                "frac_ge_cutoff": float(ge.mean()) if n_tox else 0.0,
                "fold_peak": float(is_peak.sum() / expected) if expected else float("nan"),
                "median_mad": float(np.nanmedian(per)) if n_tox else float("nan"),
                "p90_mad": float(np.nanquantile(per, 0.9)) if n_tox else float("nan"),
                "max_mad": float(np.nanmax(per)) if n_tox else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_toxic_chr_mad(df: pd.DataFrame, stats: pd.DataFrame, out: Path) -> None:
    tox = df.loc[df["toxic"]].copy()
    n_tox = len(tox)
    fig = plt.figure(figsize=(11.2, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.55], hspace=0.38, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    axh = fig.add_subplot(gs[1, :])

    x = np.arange(len(stats))
    labels = [c.replace("chr", "") for c in stats["chr"]]
    ax0.bar(x, stats["n_peak_dev"], color=DEV_TOXIC, width=0.8, label="dev")
    ax0.bar(
        x,
        stats["n_peak_test"],
        bottom=stats["n_peak_dev"],
        color=TEST_TOXIC,
        width=0.8,
        label="test",
    )
    ax0.axhline(n_tox / 22.0, color="#555", ls="--", lw=0.9, label="uniform n/22")
    ax0.set_xticks(x, labels, fontsize=8)
    ax0.set_xlabel("Chromosome")
    ax0.set_ylabel("Toxic samples")
    ax0.set_title("Peak chromosome (argmax of the MAD score)")
    ax0.legend(frameon=False, fontsize=8, ncol=3)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1.bar(x, stats["n_ge_cutoff_dev"], color=DEV_TOXIC, width=0.8, label="dev")
    ax1.bar(
        x,
        stats["n_ge_cutoff_test"],
        bottom=stats["n_ge_cutoff_dev"],
        color=TEST_TOXIC,
        width=0.8,
        label="test",
    )
    ax1.set_xticks(x, labels, fontsize=8)
    ax1.set_xlabel("Chromosome")
    ax1.set_ylabel("Toxic samples")
    ax1.set_title(f"|MAD-z| ≥ {CUTOFF:g} on that chr (a sample may hit several)")
    ax1.legend(frameon=False, fontsize=8, ncol=2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    mat = np.column_stack([_chr_abs_mad(tox, c) for c in CHR_LIST]) if n_tox else np.zeros((0, 22))
    chr_idx = {c: i for i, c in enumerate(CHR_LIST)}
    peak_i = tox["max_mad_chr"].map(chr_idx).to_numpy(dtype=int) if n_tox else np.array([], dtype=int)
    order = np.lexsort((-tox["mad_score"].to_numpy(), peak_i)) if n_tox else np.array([], dtype=int)
    mat_o = mat[order] if n_tox else mat
    vmax = float(max(CUTOFF, np.nanpercentile(mat, 98) if n_tox else CUTOFF))
    im = axh.imshow(mat_o, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax, interpolation="nearest")
    if n_tox:
        axh.scatter(
            peak_i[order],
            np.arange(n_tox),
            marker="o",
            s=14,
            facecolors="none",
            edgecolors="#111",
            linewidths=0.7,
            zorder=3,
            label="peak chr",
        )
        ytick = [s.replace("PTAY", "") for s in tox.iloc[order]["sample"]]
        axh.set_yticks(np.arange(n_tox), ytick, fontsize=5.5)
    axh.set_xticks(x, labels, fontsize=8)
    axh.set_xlabel("Chromosome")
    axh.set_ylabel("Toxic sample")
    axh.set_title(f"Per-chr |MAD-z| among toxic samples (open circle = peak; n={n_tox})")
    cbar = fig.colorbar(im, ax=axh, fraction=0.02, pad=0.01)
    cbar.set_label("|MAD-z|")
    axh.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(f"Toxic-sample MAD by chromosome (n={n_tox})", fontsize=12, y=0.98)
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    console.print(f"  wrote {out}")


def plot_cnv_overlap(df: pd.DataFrame, out: Path) -> None:
    a = int((df["toxic"] & df["cnv_positive"]).sum())
    b = int((df["toxic"] & ~df["cnv_positive"]).sum())
    c = int((~df["toxic"] & df["cnv_positive"]).sum())
    d = int((~df["toxic"] & ~df["cnv_positive"]).sum())
    mat = np.array([[d, c], [b, a]], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    im = axes[0].imshow(mat, cmap="Reds", vmin=0)
    axes[0].set_xticks([0, 1], ["CNV−", "CNV+"])
    axes[0].set_yticks([0, 1], ["OK", "toxic"])
    for (i, j), val in np.ndenumerate(mat):
        axes[0].text(j, i, int(val), ha="center", va="center", fontsize=12)
    axes[0].set_title("Candidate pool 2×2")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    cnv = df.loc[df["cnv_positive"]].sort_values("mad_score").reset_index(drop=True)
    if cnv.empty:
        axes[1].text(0.5, 0.5, "no CNV+", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    else:
        colors = [_group_color(s, t) for s, t in zip(cnv["set"], cnv["toxic"])]
        y = np.arange(len(cnv))
        axes[1].barh(y, cnv["mad_score"], color=colors, height=0.72)
        axes[1].axvline(CUTOFF, color="#222", ls="--", lw=1.0)
        axes[1].set_yticks(y, [s.replace("PTAY", "") for s in cnv["sample"]], fontsize=7)
        axes[1].set_xlabel("MAD score")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
    axes[1].set_title(f"CNV+ in pool (n={len(cnv)}; >10 Mb ∩ panel)")

    fig.suptitle("CNV+ vs MAD-toxic in the expanded Normal pool", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    console.print(f"  wrote {out}")


def write_report(
    df: pd.DataFrame,
    cnv_raw: pd.DataFrame,
    meta: pd.DataFrame,
    chr_stats: pd.DataFrame,
    out: Path,
    peak_chi: float | None = None,
    peak_p: float | None = None,
) -> None:
    n = len(df)
    n_tox = int(df["toxic"].sum())
    n_dev_tox = int(((df["set"] == "dev") & df["toxic"]).sum())
    n_test_tox = int(((df["set"] == "test") & df["toxic"]).sum())
    n_events = int(len(cnv_raw))
    n_pass_len = int(cnv_raw["pass_length"].sum()) if "pass_length" in cnv_raw.columns else n_events
    n_pass_tgt = int(cnv_raw["pass_target"].sum()) if "pass_target" in cnv_raw.columns else n_events
    n_keep = int(cnv_raw["cnv_event_keep"].sum()) if "cnv_event_keep" in cnv_raw.columns else n_events
    keep = cnv_raw.loc[cnv_raw["cnv_event_keep"]] if "cnv_event_keep" in cnv_raw.columns else cnv_raw
    n_cnv_tissue = int(keep["donor_core"].nunique())
    n_cnv_meta = int(keep.loc[keep["cf_sample"] != "", "donor_core"].nunique())
    n_cnv_pool = int(df["cnv_positive"].sum())
    a = int((df["toxic"] & df["cnv_positive"]).sum())
    b = int((df["toxic"] & ~df["cnv_positive"]).sum())
    c = int((~df["toxic"] & df["cnv_positive"]).sum())
    d = int((~df["toxic"] & ~df["cnv_positive"]).sum())
    _, p = try_fisher(np.array([[a, b], [c, d]]))
    rate_cnv = a / n_cnv_pool if n_cnv_pool else float("nan")
    rate_other = b / (b + d) if (b + d) else float("nan")

    tox_cnv = df.loc[df["toxic"] & df["cnv_positive"]].sort_values(
        "mad_score", ascending=False
    )
    ok_cnv = df.loc[~df["toxic"] & df["cnv_positive"]].sort_values(
        "mad_score", ascending=False
    )
    unmatched = (
        keep.drop_duplicates("donor_core")
        .loc[lambda x: x["cf_sample"] == "", "cnv_tissue_sample"]
        .tolist()
    )
    in_meta_not_pool = (
        keep.drop_duplicates("donor_core")
        .loc[lambda x: (x["cf_sample"] != "") & ~x["in_candidate_pool"], ["cnv_tissue_sample", "cf_sample", "cf_set", "cf_label", "cf_depth_qc"]]
    )

    def _md_table(sub: pd.DataFrame) -> str:
        cols = [
            "sample",
            "set",
            "mad_score",
            "max_mad_chr",
            "max_mad_track",
            "outlier_chrs_pct",
            "outlier_chrs_intra",
            "cnv_tissue_sample",
            "cnv_labels",
            "cnv_chrs",
            "cnv_length_mb",
            "n_target_cpgs",
            "cnv_chr_overlap",
        ]
        use = [c for c in cols if c in sub.columns]
        if sub.empty:
            return "_none_"
        lines = ["| " + " | ".join(use) + " |", "| " + " | ".join("---" for _ in use) + " |"]
        for _, r in sub.iterrows():
            vals = []
            for k in use:
                v = r[k]
                if isinstance(v, float) and np.isfinite(v):
                    if k in {"mad_score", "cnv_length_mb"}:
                        vals.append(f"{v:.3f}")
                    elif k == "n_target_cpgs":
                        vals.append(f"{v:.0f}")
                    else:
                        vals.append(str(v))
                else:
                    vals.append("" if pd.isna(v) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    if len(in_meta_not_pool):
        lines = [
            "| tissue | cfDNA | set | label | depth_qc |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, r in in_meta_not_pool.iterrows():
            lines.append(
                f"| {r['cnv_tissue_sample']} | {r['cf_sample']} | {r['cf_set']} | {r['cf_label']} | {r['cf_depth_qc']} |"
            )
        in_meta_md = "\n".join(lines)
    else:
        in_meta_md = "_none_"

    odds = (a * d) / (b * c) if b * c else float("nan")
    n_dev = int((df["set"] == "dev").sum())
    n_test = int((df["set"] == "test").sum())
    n_dev_ok = n_dev - n_dev_tox
    n_test_ok = n_test - n_test_tox
    tox = df.loc[df["toxic"]]
    track_pct = int((tox["max_mad_track"] == "percentage").sum()) if n_tox else 0
    track_intra = int((tox["max_mad_track"] == "z_intra").sum()) if n_tox else 0
    n_chrs_peak = int((chr_stats["n_peak"] > 0).sum())
    n_chrs_hit = int((chr_stats["n_ge_cutoff"] > 0).sum())
    top_peak = chr_stats.sort_values(["n_peak", "max_mad"], ascending=False).head(5)
    top_hit = chr_stats.sort_values(["n_ge_cutoff", "max_mad"], ascending=False).head(5)
    expected_peak = n_tox / 22.0 if n_tox else 0.0
    p_txt = f"{peak_p:.4f}" if peak_p is not None and np.isfinite(peak_p) else "NA"
    chi_txt = f"{peak_chi:.1f}" if peak_chi is not None and np.isfinite(peak_chi) else "NA"
    gof_line = (
        f"Peak-chr counts vs uniform 1/22: Monte-Carlo χ² = {chi_txt}, p = {p_txt} "
        f"(expected {expected_peak:.2f} peaks per chr)."
    )
    if peak_p is not None and np.isfinite(peak_p) and peak_p < 0.05:
        gof_line += " Some chromosomes attract toxic peaks more than chance."
    elif peak_p is not None and np.isfinite(peak_p):
        gof_line += " No strong evidence that peak chromosomes are concentrated."

    def _top_list(sub: pd.DataFrame, count_col: str) -> str:
        bits = []
        for _, r in sub.iterrows():
            bits.append(
                f"{r['chr']} n={int(r[count_col])} ({r[count_col] / n_tox:.1%})"
                if n_tox
                else str(r["chr"])
            )
        return ", ".join(bits) if bits else "_none_"

    chr_md_cols = [
        "chr",
        "n_peak",
        "n_peak_dev",
        "n_peak_test",
        "n_ge_cutoff",
        "fold_peak",
        "median_mad",
        "p90_mad",
        "max_mad",
    ]
    chr_lines = [
        "| " + " | ".join(chr_md_cols) + " |",
        "| " + " | ".join("---" for _ in chr_md_cols) + " |",
    ]
    for _, r in chr_stats.sort_values(
        ["n_peak", "n_ge_cutoff", "max_mad"], ascending=False
    ).iterrows():
        vals = []
        for k in chr_md_cols:
            v = r[k]
            if k == "chr":
                vals.append(str(v))
            elif k == "fold_peak":
                vals.append(f"{float(v):.2f}")
            elif k in {"median_mad", "p90_mad", "max_mad"}:
                vals.append(f"{float(v):.2f}")
            else:
                vals.append(str(int(v)))
        chr_lines.append("| " + " | ".join(vals) + " |")
    chr_md = "\n".join(chr_lines)

    report = f"""# Expanded-pool MAD scores and CNVseq overlap

Candidate reference pool from `meta_samplesheet.csv`: **set ∈ {{dev, test}}**, **depth_qc = pass**, **label = Normal**.  
Scores from `intermediate_merged_batches_modeA.parquet` (modeA after-MQ percentage + hypo/hyper z_intra).  
CNVseq calls: `PTAY_cnvseq_cnv_stat.check.20260814.xlsx` (tissue V/B aligned to cfDNA P).

## 1. Pool and MAD score

- n candidates = **{n}** (dev {n_dev}, test {n_test})
- Per chromosome, MAD-z = `0.6745 × (x − median) / MAD` estimated on this pool
- **MAD score** = max over chr1–22 of `|percentage MAD-z|` and `|z_intra MAD-z|` (z_intra = max of |hypo|, |hyper|)

## 2. Toxic dividing line

**Toxic = MAD score ≥ {CUTOFF:g}.** Same robust-z fence as the original 96-dev screen. The ranked distribution is continuous — there is no empty gap at 3.5 — but 3.0 would call {int((df["mad_score"]>=3).sum())}/{n} ({(df["mad_score"]>=3).mean():.0%}) toxic, which is too steep for a reference filter. ≥{CUTOFF:g} calls **{n_tox}/{n} ({n_tox/n:.1%})**: {n_dev_tox} dev_toxic + {n_test_tox} test_toxic.

![MAD score bars](figures/mad_score_bars.png)

*Figure 1.* Candidates ordered by ascending MAD score (sample names hidden). Colours: dev_ok, test_ok, dev_toxic, test_toxic. Dashed line = {CUTOFF:g}. Black triangles mark CNV+ donors (length >10 Mb and overlapping the 220k target panel). Inset pies (blank OK region) show ok/toxic counts: **dev** ok {n_dev_ok}/{n_dev} ({n_dev_ok / n_dev:.1%}), toxic {n_dev_tox}/{n_dev} ({n_dev_tox / n_dev:.1%}); **test** ok {n_test_ok}/{n_test} ({n_test_ok / n_test:.1%}), toxic {n_test_tox}/{n_test} ({n_test_tox / n_test:.1%}).

## 3. Toxic MAD by chromosome

The sample MAD score is the **max** over chr1–22, so an OK sample cannot have any chromosome with |MAD-z| ≥ {CUTOFF:g}. Chromosome “easiness” is therefore a question **among toxic samples**: which chr drives the score (`max_mad_chr`), and which chrs also cross {CUTOFF:g}.

- {n_chrs_peak}/22 chromosomes are a peak at least once; {n_chrs_hit}/22 have ≥1 toxic sample with |MAD-z| ≥ {CUTOFF:g}
- Peak track: **percentage** {track_pct}/{n_tox}, **z_intra** {track_intra}/{n_tox}
- Most frequent peak chrs: {_top_list(top_peak, "n_peak")}
- Most frequent |MAD-z| ≥ {CUTOFF:g} chrs: {_top_list(top_hit, "n_ge_cutoff")}
- {gof_line}

![Toxic MAD by chromosome](figures/toxic_chr_mad.png)

*Figure 2.* Top left: which chromosome is the argmax of each toxic sample’s MAD score (stacked dev/test; dashed line = uniform n/22). Top right: how often each chromosome itself exceeds {CUTOFF:g} (a sample can contribute to several bars). Bottom: per-chr |MAD-z| heatmap for the {n_tox} toxic samples, ordered by peak chromosome; open circles mark the peak.

{chr_md}

`fold_peak` = n_peak / (n_toxic / 22). `n_ge_cutoff` can exceed `n_peak` when a sample is extreme on several chromosomes.

## 4. CNV+ definition

CNVseq is whole-genome; cfDNA episcore uses the targeted 220k CpG panel (`assets/220k_cpg_recall_list/220k_cpg_recall_0.65.txt`). A donor is **CNV+** only if it has ≥1 event that satisfies both:

1. **Length > 10 Mb** (CNVseq `Length` field)
2. **Overlaps ≥1 target CpG** (interval [Start, End] vs panel `chr`/`start`)

| Filter | Events | Unique donors |
| --- | --- | --- |
| All CNVseq calls | {n_events} | {int(cnv_raw["donor_core"].nunique())} |
| Length > 10 Mb | {n_pass_len} | {int(cnv_raw.loc[cnv_raw["pass_length"], "donor_core"].nunique()) if "pass_length" in cnv_raw.columns else n_cnv_tissue} |
| Overlaps 220k panel | {n_pass_tgt} | {int(cnv_raw.loc[cnv_raw["pass_target"], "donor_core"].nunique()) if "pass_target" in cnv_raw.columns else n_cnv_tissue} |
| **CNV+ (both)** | **{n_keep}** | **{n_cnv_tissue}** |

Tissue IDs use **V** (sometimes **B**) where cfDNA uses **P**. Join key = donor core `J?PTAY` + integer id (`PTAY0614B10S1` → `PTAY0614P10S1`; `PTAY0716V` → `PTAY0716P7H1`).

| CNV+ unique donors | Match any meta row | Match the 273-candidate pool |
| --- | --- | --- |
| {n_cnv_tissue} | {n_cnv_meta} | {n_cnv_pool} |

Unmapped CNV+ tissue IDs (no cfDNA row): {", ".join(unmatched) if unmatched else "_none_"}.

In meta but **not** in the candidate pool:

{in_meta_md}

## 5. Overlap of CNV+ with toxic

|  | CNV− | CNV+ | All |
| --- | --- | --- | --- |
| OK | {d} | {c} | {c+d} |
| toxic | {b} | {a} | {a+b} |
| All | {b+d} | {c+a} | {n} |

Toxic rate: **{rate_cnv:.1%}** among CNV+ vs **{rate_other:.1%}** among the rest (odds ratio ~{odds:.2f}, Fisher p = {p:.3f}).

![CNV overlap](figures/cnv_overlap.png)

*Figure 3.* Left: 2×2 counts under the >10 Mb ∩ panel definition. Right: CNV+ candidates ranked by MAD score.

### Toxic and CNV+ (n={a})

{_md_table(tox_cnv) if a else "_none_"}

`cnv_chr_overlap` is true when a MAD-outlier chromosome (score ≥ {CUTOFF:g}) is also a qualifying CNV chromosome.

### CNV+ but MAD-OK (n={c})

{_md_table(ok_cnv) if c else "_none_"}

A large WGS CNV that misses the 220k panel does not enter this CNV+ set. Sub-10 Mb calls are excluded even if they overlap targets.

## 6. Outputs

- `candidate_mad_scores.tsv` — all {n} candidates (`max_mad_chr`, `max_mad_track`)
- `toxic_samplesheet.tsv` — {n_tox} toxic rows with CNV columns
- `toxic_chr_stats.tsv` — per-chr peak / ≥{CUTOFF:g} counts among toxic samples
- `cnvseq_mapped.tsv` — per-event flags (`pass_length`, `pass_target`, `cnv_event_keep`, `n_target_cpgs`)
- `figures/mad_score_bars.png`, `figures/toxic_chr_mad.png`, `figures/cnv_overlap.png`
"""
    out.write_text(report)
    console.print(f"  wrote {out}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--parquet", default=str(DEFAULT_PARQUET), type=click.Path(exists=True, dir_okay=False))
@click.option("--meta", default=str(DEFAULT_META), type=click.Path(exists=True, dir_okay=False))
@click.option("--cnv-xlsx", default=str(DEFAULT_CNV), type=click.Path(exists=True, dir_okay=False))
@click.option("--cpg-list", default=str(DEFAULT_CPG), type=click.Path(exists=True, dir_okay=False))
@click.option("--min-cnv-mb", default=MIN_CNV_MB, show_default=True, type=float)
@click.option("--output-dir", default=str(DEFAULT_OUT), type=click.Path(file_okay=False))
@click.option("--cutoff", default=CUTOFF, show_default=True, type=float)
def main(
    parquet: str,
    meta: str,
    cnv_xlsx: str,
    cpg_list: str,
    min_cnv_mb: float,
    output_dir: str,
    cutoff: float,
) -> None:
    global CUTOFF
    CUTOFF = float(cutoff)
    out = Path(output_dir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(meta)
    meta_df["sample"] = meta_df["sample"].astype(str)
    cand = meta_df.loc[
        meta_df["set"].isin(["dev", "test"])
        & (meta_df["depth_qc"].astype(str) == "pass")
        & (meta_df["label"].astype(str) == "Normal")
    ].copy()
    cand["donor_core"] = cand["sample"].map(donor_core)
    if cand["sample"].duplicated().any():
        raise click.ClickException("duplicate samples in candidate pool")
    console.print(
        f"candidates n={len(cand)} dev={(cand['set']=='dev').sum()} "
        f"test={(cand['set']=='test').sum()}"
    )

    mat = pd.read_parquet(parquet)
    mat["sample"] = mat["sample"].astype(str)
    missing = cand.loc[~cand["sample"].isin(mat["sample"]), "sample"].tolist()
    if missing:
        raise click.ClickException(f"{len(missing)} candidates missing from parquet: {missing[:5]}")

    scores = score_pool(mat, cand["sample"].tolist())
    df = cand.merge(scores, on="sample", how="left")
    df["toxic"] = df["mad_score"] >= CUTOFF
    df["group"] = np.where(
        df["toxic"],
        df["set"].astype(str) + "_toxic",
        df["set"].astype(str) + "_ok",
    )

    cnv_raw = read_xlsx_first_sheet(Path(cnv_xlsx))
    if "sample" not in cnv_raw.columns:
        raise click.ClickException(f"CNV xlsx missing sample column: {list(cnv_raw.columns)}")
    cnv_raw["cnv_tissue_sample"] = cnv_raw["sample"].astype(str)
    cnv_raw["donor_core"] = cnv_raw["cnv_tissue_sample"].map(donor_core)
    meta_by_core = (
        meta_df.assign(donor_core=meta_df["sample"].map(donor_core))
        .drop_duplicates("donor_core")
        .set_index("donor_core")
    )
    cand_by_core = cand.drop_duplicates("donor_core").set_index("donor_core")
    cnv_raw["cf_sample"] = cnv_raw["donor_core"].map(
        lambda k: str(meta_by_core.at[k, "sample"]) if k in meta_by_core.index else ""
    )
    cnv_raw["cf_set"] = cnv_raw["donor_core"].map(
        lambda k: meta_by_core.at[k, "set"] if k in meta_by_core.index else ""
    )
    cnv_raw["cf_label"] = cnv_raw["donor_core"].map(
        lambda k: meta_by_core.at[k, "label"] if k in meta_by_core.index else ""
    )
    cnv_raw["cf_depth_qc"] = cnv_raw["donor_core"].map(
        lambda k: meta_by_core.at[k, "depth_qc"] if k in meta_by_core.index else ""
    )
    cnv_raw["in_candidate_pool"] = cnv_raw["donor_core"].isin(set(cand_by_core.index))
    console.print(f"  loading target CpGs from {cpg_list}")
    cpg_pos = load_cpg_positions(Path(cpg_list))
    n_cpg = int(sum(v.size for v in cpg_pos.values()))
    console.print(f"  panel CpGs n={n_cpg} chr={len(cpg_pos)}")
    cnv_raw = annotate_cnv_events(cnv_raw, cpg_pos, min_mb=float(min_cnv_mb))
    console.print(
        f"  CNV events {len(cnv_raw)}: >{min_cnv_mb:g}Mb={int(cnv_raw['pass_length'].sum())} "
        f"panel-overlap={int(cnv_raw['pass_target'].sum())} "
        f"CNV+ events={int(cnv_raw['cnv_event_keep'].sum())} "
        f"CNV+ donors={int(cnv_raw.loc[cnv_raw['cnv_event_keep'], 'donor_core'].nunique())}"
    )

    keep = cnv_raw.loc[cnv_raw["cnv_event_keep"]].copy()
    agg_cols = [
        "donor_core",
        "cnv_tissue_sample",
        "n_cnv_events",
        "cnv_labels",
        "cnv_chrs",
        "cnv_length_mb",
        "n_target_cpgs",
    ]
    if keep.empty:
        agg = pd.DataFrame(columns=agg_cols)
    else:
        agg = (
            keep.groupby("donor_core", dropna=False)
            .agg(
                cnv_tissue_sample=("cnv_tissue_sample", "first"),
                n_cnv_events=("cnv_tissue_sample", "size"),
                cnv_labels=("Label", lambda s: "; ".join(str(x) for x in s if str(x))),
                cnv_chrs=("chr", lambda s: ",".join(sorted({str(x) for x in s if str(x).strip()}))),
                cnv_length_mb=("length_mb", "max"),
                n_target_cpgs=("n_target_cpgs", "sum"),
            )
            .reset_index()
        )
    raw_n = (
        cnv_raw.groupby("donor_core", dropna=False)
        .size()
        .rename("n_cnv_events_raw")
        .reset_index()
    )
    df = df.merge(raw_n, on="donor_core", how="left")
    df = df.merge(agg, on="donor_core", how="left")
    df["cnv_positive"] = df["n_cnv_events"].fillna(0).astype(int) > 0
    df["n_cnv_events"] = df["n_cnv_events"].fillna(0).astype(int)
    df["n_cnv_events_raw"] = df["n_cnv_events_raw"].fillna(0).astype(int)
    df["n_target_cpgs"] = df["n_target_cpgs"].fillna(0).astype(int)
    df["cnv_tissue_sample"] = df["cnv_tissue_sample"].fillna("")
    df["cnv_labels"] = df["cnv_labels"].fillna("")
    df["cnv_chrs"] = df["cnv_chrs"].fillna("")
    df["cnv_length_mb"] = pd.to_numeric(df["cnv_length_mb"], errors="coerce")

    def _overlap(row: pd.Series) -> bool:
        mad_chrs = set(
            str(row["outlier_chrs_pct"]).split(",") + str(row["outlier_chrs_intra"]).split(",")
        )
        mad_chrs = {c for c in mad_chrs if c}
        cnv_chrs = {c for c in str(row["cnv_chrs"]).split(",") if c}
        return bool(mad_chrs & cnv_chrs)

    df["cnv_chr_overlap"] = df.apply(_overlap, axis=1)
    df = df.sort_values(["mad_score", "sample"], ascending=[False, True]).reset_index(drop=True)
    df["mad_rank"] = np.arange(1, len(df) + 1)

    slim_cols = [
        "mad_rank",
        "sample",
        "set",
        "group",
        "toxic",
        "mad_score",
        "max_mad_chr",
        "max_mad_track",
        "max_abs_pct_madz",
        "max_abs_intra_madz",
        "outlier_chrs_pct",
        "outlier_chrs_intra",
        "ff_before_mq",
        "depth_qc",
        "label",
        "cnv_positive",
        "cnv_tissue_sample",
        "n_cnv_events",
        "n_cnv_events_raw",
        "cnv_length_mb",
        "n_target_cpgs",
        "cnv_labels",
        "cnv_chrs",
        "cnv_chr_overlap",
        "donor_core",
    ]
    slim_cols = [c for c in slim_cols if c in df.columns]
    df[slim_cols].to_csv(out / "candidate_mad_scores.tsv", sep="\t", index=False, float_format="%.6f")
    toxic = df.loc[df["toxic"], slim_cols]
    toxic.to_csv(out / "toxic_samplesheet.tsv", sep="\t", index=False, float_format="%.6f")
    cnv_raw.to_csv(out / "cnvseq_mapped.tsv", sep="\t", index=False)

    summary = {
        "n_candidates": int(len(df)),
        "n_dev": int((df["set"] == "dev").sum()),
        "n_test": int((df["set"] == "test").sum()),
        "cutoff": CUTOFF,
        "n_toxic": int(df["toxic"].sum()),
        "n_dev_toxic": int(((df["set"] == "dev") & df["toxic"]).sum()),
        "n_test_toxic": int(((df["set"] == "test") & df["toxic"]).sum()),
        "n_cnv_tissue": int(cnv_raw.loc[cnv_raw["cnv_event_keep"], "donor_core"].nunique()),
        "n_cnv_events_keep": int(cnv_raw["cnv_event_keep"].sum()),
        "n_cnv_in_pool": int(df["cnv_positive"].sum()),
        "n_toxic_and_cnv": int((df["toxic"] & df["cnv_positive"]).sum()),
        "cnv_min_mb": float(min_cnv_mb),
        "cpg_list": str(Path(cpg_list).resolve()),
        "score": "max(|pct MAD-z|, |z_intra MAD-z|) over chr1-22, after_mq, pool-internal",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    chr_stats = toxic_chr_stats(df)
    peak_chi, peak_p = peak_chr_uniform_gof(chr_stats["n_peak"].to_numpy())
    chr_stats.to_csv(out / "toxic_chr_stats.tsv", sep="\t", index=False, float_format="%.6f")
    summary["peak_chr_chi2"] = peak_chi
    summary["peak_chr_uniform_p"] = peak_p
    summary["n_chrs_with_peak"] = int((chr_stats["n_peak"] > 0).sum())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_score_bars(df, figdir / "mad_score_bars.png")
    plot_toxic_chr_mad(df, chr_stats, figdir / "toxic_chr_mad.png")
    plot_cnv_overlap(df, figdir / "cnv_overlap.png")
    write_report(
        df,
        cnv_raw,
        meta_df,
        chr_stats,
        out / "REPORT.md",
        peak_chi=peak_chi,
        peak_p=peak_p,
    )
    console.print(
        f"[green]OK[/green] toxic={summary['n_toxic']}/{summary['n_candidates']} "
        f"cnv_in_pool={summary['n_cnv_in_pool']} overlap={summary['n_toxic_and_cnv']} -> {out}"
    )


if __name__ == "__main__":
    main()
