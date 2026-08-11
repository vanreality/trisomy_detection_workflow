#!/usr/bin/env python3
"""Prepare samplesheets / meta for 20260731 early+middle analysis.

Writes under INPUT_DIR:
  cohort_labels.csv
  samplesheet_nf_early.csv / samplesheet_nf_middle.csv
  episcore_samples_meta.csv / zscore_samples_meta.csv   (conventional early_ref)
  features_samples_meta.csv                             (all early+middle)
  {male,female}_ref_{episcore,zscore}_meta.csv          (after middle gender exists)
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console

import config as cfg

console = Console()


def _group_deconv(samplesheet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample, grp in samplesheet.groupby("sample", sort=False):
        deconv = sorted({str(p) for p in grp["deconv_res"].tolist() if pd.notna(p)})
        bams = sorted({str(p) for p in grp["clean_bam"].tolist() if pd.notna(p)})
        rows.append(
            {
                "sample": sample,
                "clean_bam": bams[0] if bams else None,
                "deconv_paths": ",".join(deconv),
            }
        )
    return pd.DataFrame(rows)


def _beta_path(sample: str) -> str | None:
    for directory in (
        cfg.OLD_BETA_20260416,
        cfg.OLD_BETA_20260507,
        cfg.OLD_BETA_20260720,
        cfg.EARLY_NF_OUT / "extract_beta_value",
        cfg.MIDDLE_NF_OUT / "extract_beta_value",
        cfg.OUTPUT_DIR / "extract_beta_value",
    ):
        path = directory / f"{sample}_beta_value.tsv.gz"
        if path.is_file():
            return str(path)
    return None


def _build_cohort_labels() -> pd.DataFrame:
    """Prior early (20260416/507/720) + 4 new early + middle placeholders."""
    frames = []
    prior = pd.read_csv(cfg.OLD_LABELS_20260720)
    # Collapse prior new→old for this run (they are background early)
    prior = prior.copy()
    prior["cohort"] = "old_early"
    prior["dataset"] = "early"
    frames.append(prior[["sample", "label", "cohort", "dataset"]])

    new_early = pd.read_csv(cfg.EARLY_META)
    new_early["cohort"] = "new_early"
    new_early["dataset"] = "early"
    frames.append(new_early[["sample", "label", "cohort", "dataset"]])

    middle = _group_deconv(pd.read_csv(cfg.MIDDLE_MQRES))
    mid = pd.DataFrame(
        {
            "sample": middle["sample"],
            "label": "Normal",
            "cohort": "middle",
            "dataset": "middle",
        }
    )
    frames.append(mid)

    labels = pd.concat(frames, ignore_index=True)
    labels = labels.drop_duplicates(subset=["sample"], keep="last")
    return labels.reset_index(drop=True)


def _all_early_mqres() -> pd.DataFrame:
    frames = [
        pd.read_csv(cfg.OLD_SAMPLESHEET_20260416),
        pd.read_csv(cfg.OLD_SAMPLESHEET_20260507),
        pd.read_csv(cfg.OLD_SAMPLESHEET_20260720),
        pd.read_csv(cfg.EARLY_MQRES),
    ]
    return pd.concat(frames, ignore_index=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    cfg.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = _build_cohort_labels()
    # Overlay assigned middle gender if available
    if cfg.MIDDLE_GENDER.is_file():
        mg = pd.read_csv(cfg.MIDDLE_GENDER)
        labels = labels.merge(
            mg[["sample", "assigned_gender"]], on="sample", how="left"
        )
        mid_mask = labels["dataset"] == "middle"
        labels.loc[mid_mask, "label"] = labels.loc[mid_mask, "assigned_gender"].fillna(
            "Normal"
        )
        labels = labels.drop(columns=["assigned_gender"])
    labels.to_csv(cfg.COHORT_LABELS, index=False)
    console.print(f"[green]Wrote[/green] {cfg.COHORT_LABELS}  ({len(labels)} samples)")

    early_mq = pd.read_csv(cfg.EARLY_MQRES)
    early_mq.to_csv(cfg.NF_SAMPLESHEET_EARLY, index=False)
    console.print(
        f"[green]Wrote[/green] {cfg.NF_SAMPLESHEET_EARLY}  "
        f"({early_mq['sample'].nunique()} samples, {len(early_mq)} rows)"
    )

    middle_mq = pd.read_csv(cfg.MIDDLE_MQRES)
    middle_mq.to_csv(cfg.NF_SAMPLESHEET_MIDDLE, index=False)
    console.print(
        f"[green]Wrote[/green] {cfg.NF_SAMPLESHEET_MIDDLE}  "
        f"({middle_mq['sample'].nunique()} samples, {len(middle_mq)} rows)"
    )

    early_deconv = _group_deconv(_all_early_mqres()).merge(
        labels.loc[labels["dataset"] == "early"], on="sample", how="inner"
    )
    middle_deconv = _group_deconv(middle_mq).merge(
        labels.loc[labels["dataset"] == "middle"], on="sample", how="inner"
    )

    # --- Conventional episcore/zscore: 20260416 females = early_ref ---
    female_ref = {
        s
        for s in labels.loc[
            (labels["label"] == "female") & (labels["cohort"] == "old_early"), "sample"
        ]
        if (cfg.OLD_BETA_20260416 / f"{s}_beta_value.tsv.gz").is_file()
    }

    epi_rows = []
    for _, row in early_deconv.iterrows():
        sample = row["sample"]
        beta = _beta_path(sample)
        if beta is None:
            continue
        label = row["label"]
        if pd.isna(label) or label == "male":
            continue
        ref_type = "early_ref" if sample in female_ref else "analyze"
        if ref_type == "early_ref" and not beta.startswith(str(cfg.OLD_BETA_20260416)):
            ref_type = "analyze"
        epi_rows.append(
            {
                "sample": sample,
                "beta_path": beta,
                "ref_type": ref_type,
                "label": label,
                "cohort": row["cohort"],
            }
        )
    epi_df = pd.DataFrame(epi_rows)
    epi_df.to_csv(cfg.EPISCORE_SAMPLES_META, index=False)
    console.print(
        f"[green]Wrote[/green] {cfg.EPISCORE_SAMPLES_META}  "
        f"(early_ref={int((epi_df.ref_type == 'early_ref').sum())}, "
        f"analyze={int((epi_df.ref_type == 'analyze').sum())})"
    )
    missing_new = [
        s
        for s in labels.loc[labels["cohort"] == "new_early", "sample"]
        if _beta_path(s) is None
    ]
    if missing_new:
        console.print(
            f"[yellow]New early missing beta (run nextflow early):[/yellow] "
            f"{', '.join(map(str, missing_new))}"
        )

    z_rows = []
    for _, row in early_deconv.iterrows():
        sample = row["sample"]
        label = row["label"]
        if pd.isna(label) or label == "male" or pd.isna(row.get("deconv_paths")):
            continue
        if sample in female_ref:
            ref_type = "early_ref"
        elif label == "female":
            continue
        else:
            ref_type = "analyze"
        z_rows.append(
            {
                "sample": sample,
                "deconv_paths": row["deconv_paths"],
                "ref_type": ref_type,
                "label": label,
                "cohort": row["cohort"],
            }
        )
    z_df = pd.DataFrame(z_rows)
    z_df.to_csv(cfg.ZSCORE_SAMPLES_META, index=False)
    console.print(
        f"[green]Wrote[/green] {cfg.ZSCORE_SAMPLES_META}  "
        f"(early_ref={int((z_df.ref_type == 'early_ref').sum())}, "
        f"analyze={int((z_df.ref_type == 'analyze').sum())})"
    )

    # --- Features meta: all early + middle with beta/deconv ---
    feat_rows = []
    for _, row in pd.concat([early_deconv, middle_deconv], ignore_index=True).iterrows():
        beta = _beta_path(row["sample"])
        feat_rows.append(
            {
                "sample": row["sample"],
                "deconv_paths": row["deconv_paths"],
                "beta_path": beta,
                "label": row["label"],
                "cohort": row["cohort"],
                "dataset": row["dataset"],
            }
        )
    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(cfg.FEATURES_SAMPLES_META, index=False)
    console.print(
        f"[green]Wrote[/green] {cfg.FEATURES_SAMPLES_META}  "
        f"(n={len(feat_df)}, with_beta={feat_df['beta_path'].notna().sum()})"
    )

    # --- Middle-ref metas (need assigned male/female on middle) ---
    mid_lab = labels.loc[labels["dataset"] == "middle"]
    mid_male = set(mid_lab.loc[mid_lab["label"] == "male", "sample"])
    mid_female = set(mid_lab.loc[mid_lab["label"] == "female", "sample"])
    early_lab = labels.loc[labels["dataset"] == "early"]

    def _write_ref_metas(gender: str, ref_samples: set[str]) -> None:
        if not ref_samples:
            console.print(
                f"[yellow]Skip {gender}_ref metas — no middle {gender} yet "
                f"(run assign_middle_gender after FF/chrY)[/yellow]"
            )
            return
        epi_m = []
        z_m = []
        # refs
        for s in sorted(ref_samples):
            beta = _beta_path(s)
            dpaths = middle_deconv.loc[
                middle_deconv["sample"] == s, "deconv_paths"
            ]
            if beta is None or dpaths.empty:
                continue
            epi_m.append(
                {
                    "sample": s,
                    "beta_path": beta,
                    "ref_type": "early_ref",
                    "label": gender,
                    "cohort": "middle",
                }
            )
            z_m.append(
                {
                    "sample": s,
                    "deconv_paths": dpaths.iloc[0],
                    "ref_type": "early_ref",
                    "label": gender,
                    "cohort": "middle",
                }
            )
        # analyze = all early (incl males for completeness on plots of abnormals+normals)
        for _, row in early_lab.iterrows():
            s = row["sample"]
            beta = _beta_path(s)
            dpaths = early_deconv.loc[early_deconv["sample"] == s, "deconv_paths"]
            if beta is None or dpaths.empty:
                continue
            epi_m.append(
                {
                    "sample": s,
                    "beta_path": beta,
                    "ref_type": "analyze",
                    "label": row["label"],
                    "cohort": row["cohort"],
                }
            )
            z_m.append(
                {
                    "sample": s,
                    "deconv_paths": dpaths.iloc[0],
                    "ref_type": "analyze",
                    "label": row["label"],
                    "cohort": row["cohort"],
                }
            )
        epi_out = (
            cfg.MALE_REF_EPISCORE_META
            if gender == "male"
            else cfg.FEMALE_REF_EPISCORE_META
        )
        z_out = (
            cfg.MALE_REF_ZSCORE_META if gender == "male" else cfg.FEMALE_REF_ZSCORE_META
        )
        pd.DataFrame(epi_m).to_csv(epi_out, index=False)
        pd.DataFrame(z_m).to_csv(z_out, index=False)
        console.print(
            f"[green]Wrote[/green] {epi_out.name} / {z_out.name}  "
            f"(ref={len(ref_samples)}, analyze_early={len(early_lab)})"
        )

    _write_ref_metas("male", mid_male)
    _write_ref_metas("female", mid_female)
    console.print("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
