#!/usr/bin/env python3
"""Per-sample FP/FN detail across fixed-combo / fixed-ez flag repeats.

For each eval sample (ff≥ff_min, not blacklisted), count how often it is a
false positive (Normal flagged) or false negative (Trisomy missed). Emits:

  * ``fp_fn_sample_detail_{score}.tsv`` — every sample that errs at least once
  * ``fp_fn_shared_errors_{score}.tsv`` — samples wrong in *all* repeats
  * ``fp_fn_sample_detail_summary.json`` — compact summary

Columns: sample, set, label, ff_before_mq, error_type, pred_label,
n_error_repeats, n_repeats, frac_error.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console

from separation import is_trisomy_label

console = Console()

DEFAULT_BLACKLIST = (
    "PTAY0577P9S1",
    "PTAY0599P8S1",
    "PTAY0666P7S1",
    "PTAY0682P7S1",
    "PTAY0689P8H1",
)


def _load_flags(flag_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    def _start(p: Path) -> int:
        return int(p.stem.split("_")[1])

    shards = sorted(flag_dir.glob("flags_*.npz"), key=_start)
    if not shards:
        raise click.ClickException(f"No flags_*.npz under {flag_dir}")
    ep, z, ez = [], [], []
    for p in shards:
        d = np.load(p)
        ep.append(d["flags_ep"])
        z.append(d["flags_z"])
        ez.append(d["flags_ez"])
    cfg = {}
    cfg_path = flag_dir / "run_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
    return {
        "episcore": np.concatenate(ep, axis=0),
        "zscore": np.concatenate(z, axis=0),
        "ezscore": np.concatenate(ez, axis=0),
    }, cfg


def _detail_for_score(
    flags: np.ndarray,
    eval_info: pd.DataFrame,
    *,
    keep: np.ndarray,
    y_all: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all ever-wrong, always-wrong) detail tables."""
    sub = eval_info.loc[keep].reset_index(drop=True)
    y = y_all[keep]
    pred = flags[:, keep].astype(bool)
    n_rep = int(pred.shape[0])
    rows = []
    for i, row in sub.iterrows():
        n_flagged = int(pred[:, i].sum())
        truth_t = bool(y[i])
        if not truth_t:
            n_err = n_flagged  # FP when flagged
            error_type = "FP"
            pred_when_err = "Abnormal"
        else:
            n_err = n_rep - n_flagged  # FN when not flagged
            error_type = "FN"
            pred_when_err = "Normal"
        if n_err <= 0:
            continue
        rows.append(
            {
                "sample": str(row["sample"]),
                "set": str(row["set"]),
                "label": str(row["label"]),
                "ff_before_mq": float(row["ff_before_mq"])
                if pd.notna(row["ff_before_mq"])
                else float("nan"),
                "error_type": error_type,
                "pred_label": pred_when_err,
                "n_error_repeats": n_err,
                "n_repeats": n_rep,
                "frac_error": float(n_err) / float(n_rep),
                "n_pred_abnormal": n_flagged,
            }
        )
    cols = [
        "sample",
        "set",
        "label",
        "ff_before_mq",
        "error_type",
        "pred_label",
        "n_error_repeats",
        "n_repeats",
        "frac_error",
        "n_pred_abnormal",
    ]
    detail = pd.DataFrame(rows, columns=cols)
    if detail.empty:
        shared = detail.copy()
    else:
        detail = detail.sort_values(
            ["frac_error", "error_type", "sample"], ascending=[False, True, True]
        ).reset_index(drop=True)
        shared = detail.loc[detail["frac_error"] >= 1.0 - 1e-12].copy()
    return detail, shared


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--flag-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path))
@click.option("--ff-min", default=0.01, show_default=True, type=float)
@click.option(
    "--score",
    default="ezscore",
    type=click.Choice(["ezscore", "episcore", "zscore", "all"]),
    show_default=True,
)
@click.option(
    "--blacklist",
    default=",".join(DEFAULT_BLACKLIST),
    show_default=True,
)
def main(
    flag_dir: Path,
    output_dir: Path | None,
    ff_min: float,
    score: str,
    blacklist: str,
) -> None:
    out = output_dir or (flag_dir.parent / "fp_fn_density_fixed_ez")
    if not out.name.startswith("fp_fn"):
        # when caller passes parent OUT_BASE, write beside flags
        cand = flag_dir.parent / "fp_fn_density_fixed_ez"
        out = cand if cand.is_dir() else (flag_dir.parent / "fp_fn_sample_detail")
    out.mkdir(parents=True, exist_ok=True)

    eval_info = pd.read_csv(flag_dir / "eval_samples.tsv", sep="\t")
    score_map, cfg = _load_flags(flag_dir)

    bl = {s.strip() for s in blacklist.split(",") if s.strip()}
    ff = pd.to_numeric(eval_info["ff_before_mq"], errors="coerce").to_numpy()
    labels = eval_info["label"].astype(str)
    samples = eval_info["sample"].astype(str)
    y_all = labels.map(is_trisomy_label).to_numpy()
    keep = (
        (ff >= ff_min)
        & (labels.eq("Normal").to_numpy() | y_all)
        & ~samples.isin(bl).to_numpy()
    )

    names = list(score_map) if score == "all" else [score]
    summary: dict = {
        "flag_dir": str(flag_dir),
        "ff_min": ff_min,
        "blacklist": sorted(bl),
        "n_repeats": int(next(iter(score_map.values())).shape[0]),
        "n_eval_kept": int(keep.sum()),
        "ez_cutoff": cfg.get("ez_cutoff"),
        "scores": {},
    }

    for name in names:
        detail, shared = _detail_for_score(
            score_map[name], eval_info, keep=keep, y_all=y_all
        )
        detail_path = out / f"fp_fn_sample_detail_{name}.tsv"
        shared_path = out / f"fp_fn_shared_errors_{name}.tsv"
        detail.to_csv(detail_path, sep="\t", index=False, float_format="%.6f")
        shared.to_csv(shared_path, sep="\t", index=False, float_format="%.6f")

        n_fp = int((detail["error_type"] == "FP").sum()) if len(detail) else 0
        n_fn = int((detail["error_type"] == "FN").sum()) if len(detail) else 0
        summary["scores"][name] = {
            "n_ever_fp_samples": n_fp,
            "n_ever_fn_samples": n_fn,
            "n_shared_all_repeats": int(len(shared)),
            "shared_samples": shared["sample"].tolist() if len(shared) else [],
            "detail_tsv": str(detail_path),
            "shared_tsv": str(shared_path),
        }
        console.print(
            f"[bold]{name}[/bold]: ever-FP={n_fp} ever-FN={n_fn} "
            f"shared-in-all-repeats={len(shared)}"
        )
        if len(detail):
            show = detail.head(20)
            console.print(show.to_string(index=False))
        else:
            console.print("  (no FP/FN samples)")

    (out / "fp_fn_sample_detail_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    console.print(f"[green]OK[/green] Wrote detail under {out}")


if __name__ == "__main__":
    main()
