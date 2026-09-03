#!/usr/bin/env python3
"""Build SNP pileup samplesheet for current meta samplesheet (merged-batch cohort).

Resolution order per meta sample:
  1. Existing path from the old article pileup samplesheet (if file exists)
  2. Search under DNA_5mC pipeline output (and each mqres bam_root)
  3. Mark for Nextflow ``est_ff_from_bam`` rebuild from mqres clean_bam+deconv

Writes under ``--outdir`` (default samplesheet_summary):
  - snp_pileup_samplesheet.csv          (sample, pileup / pileup_file)
  - snp_pileup_samplesheet_report.md
  - snp_pileup_need_compute.csv         (est_ff_from_bam input, if any)
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_META = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/meta_samplesheet.csv")
DEFAULT_MQRES = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary/mqres_samplesheet.csv")
DEFAULT_OLD = Path("/lustre1/cqyi/syfan/nipt_article_plot/snp_pileup_result_samplesheet.csv")
DEFAULT_OUTDIR = Path("/lustre1/cqyi/AIPT_2.0/results/samplesheet_summary")
DNA_ROOT = Path("/lustre1/cqyi/myli/bert/DNA_5mC_analysis_pipeline/output")
# user-facing alias (symlink to DNA_ROOT)
DNA_ROOT_ALT = Path("/appsnew/home/myli/lustre1/bert/DNA_5mC_analysis_pipeline/output")
_DATE_RE = re.compile(r"(\d{8})")


def yyyymmdd(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    m = _DATE_RE.search(s)
    return m.group(1) if m else None


def bam_root(clean_bam: str) -> Path | None:
    p = Path(str(clean_bam))
    if "bwameth_results" not in p.parts:
        return None
    i = p.parts.index("bwameth_results")
    return Path(*p.parts[: i + 1])


def pileup_at_root(root: Path, sample: str) -> Path | None:
    p = (
        root
        / "zscore_downstream"
        / "beta_zscore"
        / sample
        / "bam_to_pileup"
        / f"{sample}_pileup.tsv.gz"
    )
    return p if p.is_file() else None


def search_dna(sample: str, batches: list[str]) -> Path | None:
    roots = []
    for r in (DNA_ROOT, DNA_ROOT_ALT):
        if r.is_dir():
            roots.append(r.resolve())
    roots = list(dict.fromkeys(roots))

    for root in roots:
        for b in batches:
            hit = pileup_at_root(root / f"{b}-XML" / "bwameth_results", sample)
            if hit is not None:
                return hit
        # batch-agnostic fallback (first hit)
        pattern = f"*/bwameth_results/zscore_downstream/beta_zscore/{sample}/bam_to_pileup/{sample}_pileup.tsv.gz"
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def search_mqres_roots(sample: str, mq_rows: pd.DataFrame) -> Path | None:
    for _, r in mq_rows.iterrows():
        root = bam_root(r["clean_bam"])
        if root is None:
            continue
        hit = pileup_at_root(root, sample)
        if hit is not None:
            return hit
    return None


def pick_mqres_unit(g: pd.DataFrame) -> pd.Series:
    """Prefer non-SE + parquet deconv (same as intermediate units)."""
    g = g.copy()
    g["is_se"] = g["deconv_res"].astype(str).str.contains("single_end", case=False, na=False)
    g2 = g[~g["is_se"]] if (~g["is_se"]).any() else g
    g2 = g2.copy()
    g2["_prio"] = g2["deconv_res"].map(lambda p: 0 if str(p).endswith(".parquet") else 1)
    return g2.sort_values(["_prio", "mqres_batch"]).iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--mqres", type=Path, default=DEFAULT_MQRES)
    ap.add_argument("--old-pileup", type=Path, default=DEFAULT_OLD)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument(
        "--multi-batch-only",
        action="store_true",
        help="Restrict to samples with >1 available_batches.",
    )
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta)
    mq = pd.read_csv(args.mqres)
    old = pd.read_csv(args.old_pileup)
    old_col = "pileup_file" if "pileup_file" in old.columns else "pileup"
    old_map = {
        str(r["sample"]): str(r[old_col])
        for _, r in old.iterrows()
        if pd.notna(r.get(old_col))
    }

    mq = mq.copy()
    mq["mqres_batch"] = mq["mqres_batch"].map(yyyymmdd)
    mq_by = {s: g for s, g in mq.groupby("sample", sort=False)}

    if args.multi_batch_only:
        meta = meta[
            meta["available_batches"].astype(str).map(
                lambda s: len([x for x in s.split(",") if x.strip()]) > 1
            )
        ].copy()

    rows = []
    need_compute = []
    stats = {
        "n_meta": len(meta),
        "from_old": 0,
        "from_dna_search": 0,
        "from_mqres_root": 0,
        "need_compute": 0,
    }

    for r in meta.itertuples(index=False):
        sample = str(r.sample)
        batches = [
            b.strip()
            for b in str(getattr(r, "available_batches", "") or "").split(",")
            if b.strip()
        ]
        source = None
        path = None

        # 1) old samplesheet
        cand = old_map.get(sample)
        if cand and Path(cand).is_file():
            path, source = cand, "old_samplesheet"
            stats["from_old"] += 1
        else:
            # 2a) DNA_5mC search
            hit = search_dna(sample, batches)
            if hit is not None:
                path, source = str(hit), "dna_5mc_search"
                stats["from_dna_search"] += 1
            else:
                # 2b) mqres bam_root
                g = mq_by.get(sample)
                if g is not None:
                    hit = search_mqres_roots(sample, g)
                    if hit is not None:
                        path, source = str(hit), "mqres_bam_root"
                        stats["from_mqres_root"] += 1

        if path is None:
            source = "need_compute"
            stats["need_compute"] += 1
            g = mq_by.get(sample)
            if g is not None:
                # one row per remaining batch for est_ff_from_bam (SPLIT_BAM merges)
                for _, ur in g.groupby("mqres_batch", sort=False):
                    unit = pick_mqres_unit(ur)
                    need_compute.append(
                        {
                            "sample": sample,
                            "clean_bam": unit["clean_bam"],
                            "deconv_res": unit["deconv_res"],
                        }
                    )

        rows.append(
            {
                "sample": sample,
                "pileup": path,
                "pileup_file": path,
                "source": source,
                "available_batches": ",".join(batches),
                "n_batches": len(batches),
            }
        )

    out = pd.DataFrame(rows)
    # NF est_ff_from_pileup wants sample,pileup
    pileup_sheet = out.loc[out["pileup"].notna(), ["sample", "pileup"]].copy()
    pileup_path = outdir / "snp_pileup_samplesheet.csv"
    pileup_sheet.to_csv(pileup_path, index=False)
    # full audit table
    out.to_csv(outdir / "snp_pileup_samplesheet_full.csv", index=False)

    need_df = pd.DataFrame(need_compute)
    need_path = outdir / "snp_pileup_need_compute.csv"
    if len(need_df):
        need_df.to_csv(need_path, index=False)
    elif need_path.is_file():
        need_path.unlink()

    lines = [
        "# SNP pileup samplesheet (meta / merged-batch cohort)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- Meta samples: **{stats['n_meta']}**",
        f"- Resolved pileups: **{len(pileup_sheet)}**",
        f"- From old samplesheet: **{stats['from_old']}**",
        f"- From DNA_5mC search: **{stats['from_dna_search']}**",
        f"- From mqres bam_root: **{stats['from_mqres_root']}**",
        f"- Need Nextflow compute: **{stats['need_compute']}**",
        "",
        f"Output: `{pileup_path}`",
        "",
        "Column `pileup` is ready for `--step est_ff_from_pileup`.",
        "",
    ]
    if stats["need_compute"]:
        lines += [
            f"Rebuild input: `{need_path}`",
            "",
            "```bash",
            "nextflow run /lustre1/cqyi/AIPT_2.0/workflow/episcore/main.nf \\",
            "  -profile early,alioth_slurm,singularity \\",
            f"  --input {need_path} \\",
            "  --outdir /lustre1/cqyi/AIPT_2.0/results/episcore_output/<run_id> \\",
            "  --step est_ff_from_bam \\",
            "  --ff_precision 0.0001",
            "```",
            "",
        ]
    else:
        lines.append("No samples require pileup rebuild.")
    (outdir / "snp_pileup_samplesheet_report.md").write_text("\n".join(lines) + "\n")

    print(stats)
    print(f"Wrote {pileup_path} ({len(pileup_sheet)} rows)")
    if stats["need_compute"]:
        print(f"Need compute: {need_path} ({len(need_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
