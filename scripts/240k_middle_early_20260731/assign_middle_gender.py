#!/usr/bin/env python3
"""Assign fetal gender to middle samples using early male/female FF–chrY fits.

Fits separate linear models on early male and early female (ff_before_mq → chrY_ratio).
Each middle sample is assigned to the closer line (vertical residual in chrY).
"""

from __future__ import annotations

import click
import numpy as np
import pandas as pd
from rich.console import Console

import config as cfg

console = Console()


def _fit_line(df: pd.DataFrame) -> tuple[float, float] | None:
    sub = df.dropna(subset=["ff_before_mq", "chrY_ratio"])
    if len(sub) < 2:
        return None
    x = sub["ff_before_mq"].to_numpy(dtype=float)
    y = sub["chrY_ratio"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _pred(fit: tuple[float, float], x: float) -> float:
    return fit[0] * x + fit[1]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--min-delta",
    type=float,
    default=0.0,
    show_default=True,
    help="If |resid_male−resid_female| < min_delta, mark ambiguous (keep Normal).",
)
def main(min_delta: float) -> None:
    if not cfg.CHRY_FF_TSV.is_file():
        raise click.ClickException(f"Missing {cfg.CHRY_FF_TSV}; run collect_chry_ff.py")

    df = pd.read_csv(cfg.CHRY_FF_TSV, sep="\t")
    early = df[df["dataset"] == "early"].copy()
    middle = df[df["dataset"] == "middle"].copy()

    male_fit = _fit_line(early[early["label"] == "male"])
    female_fit = _fit_line(early[early["label"] == "female"])
    if male_fit is None or female_fit is None:
        raise click.ClickException(
            "Need ≥2 early male and ≥2 early female with FF+chrY to fit lines"
        )

    console.print(
        f"male fit   : chrY = {male_fit[0]:.6g} * FF + {male_fit[1]:.6g}"
    )
    console.print(
        f"female fit : chrY = {female_fit[0]:.6g} * FF + {female_fit[1]:.6g}"
    )

    rows = []
    for _, row in middle.iterrows():
        ff = row["ff_before_mq"]
        cy = row["chrY_ratio"]
        if pd.isna(ff) or pd.isna(cy):
            rows.append(
                {
                    "sample": row["sample"],
                    "ff_before_mq": ff,
                    "chrY_ratio": cy,
                    "resid_male": np.nan,
                    "resid_female": np.nan,
                    "assigned_gender": "ambiguous",
                    "male_slope": male_fit[0],
                    "male_intercept": male_fit[1],
                    "female_slope": female_fit[0],
                    "female_intercept": female_fit[1],
                }
            )
            continue
        rm = abs(cy - _pred(male_fit, float(ff)))
        rf = abs(cy - _pred(female_fit, float(ff)))
        if abs(rm - rf) < min_delta:
            gender = "ambiguous"
        else:
            gender = "male" if rm < rf else "female"
        rows.append(
            {
                "sample": row["sample"],
                "ff_before_mq": ff,
                "chrY_ratio": cy,
                "resid_male": rm,
                "resid_female": rf,
                "assigned_gender": gender,
                "male_slope": male_fit[0],
                "male_intercept": male_fit[1],
                "female_slope": female_fit[0],
                "female_intercept": female_fit[1],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(cfg.MIDDLE_GENDER, index=False)
    console.print(f"[green]Wrote[/green] {cfg.MIDDLE_GENDER}")
    console.print(out["assigned_gender"].value_counts().to_string())


if __name__ == "__main__":
    main()
