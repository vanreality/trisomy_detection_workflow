#!/usr/bin/env python3
"""Map expanded-pool Normals to DB label_1..label_9 and audit Normal evidence.

Pulls ``样本总表`` via ``notebooks/aipt_1.0/tools/db_helper``, renames follow-up
fields with ``update_samplesheet.rename_from_db_to_feishu``, then classifies
where each sample's ``Normal`` label comes from. Weak DB classes that match
Feishu CNV (`read_cnv_df`) become ``cnv_seq``. Published classes are
``birth_outcome`` / ``cnv_seq`` / ``other``.

Writes under ``--outdir`` (default: expanded_pool_mad):

- ``pool_label_source.tsv`` — all candidates + label columns + source_class
- ``toxic_samplesheet_with_db_labels.tsv`` — toxic subset
- ``ok_samplesheet_with_db_labels.tsv`` — OK subset
- ``toxic_label_source_summary.tsv`` — counts by group × source_class
- ``toxic_label_source_report.md`` — distribution report
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
AIPT10_TOOLS = REPO_ROOT / "notebooks" / "aipt_1.0"
if str(AIPT10_TOOLS) not in sys.path:
    sys.path.insert(0, str(AIPT10_TOOLS))

from tools.db_helper import AIPTDatabase  # noqa: E402
from tools.feishu_helper import read_cnv_df  # noqa: E402
from tools import update_samplesheet as us  # noqa: E402

console = Console()

DEFAULT_OUT_BASE = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule"
)
DEFAULT_TOXIC = DEFAULT_OUT_BASE / "expanded_pool_mad" / "toxic_samplesheet.tsv"
DEFAULT_CANDIDATES = DEFAULT_OUT_BASE / "expanded_pool_mad" / "candidate_mad_scores.tsv"
DEFAULT_OUTDIR = DEFAULT_OUT_BASE / "expanded_pool_mad"

LABEL_COL_TO_DB = {
    "label_1": "检测结果_简述",
    "label_2": "随访结果_出生_",
    "label_3": "随访结果_染色体倍性",
    "label_4": "随访情况_羊穿_绒穿",
    "label_5": "随访情况_备注",
    "label_6": "J_随访情况_染色体倍性",
    "label_7": "J_随访情况_羊穿_绒穿",
    "label_8": "J_随访情况_备注",
    "label_9": "安医随访结果_20260126白底_",
}
LABEL_COLS = [f"label_{i}" for i in range(1, 10)]

# CNV-seq overrides the weaker DB-only classes.
CNV_RECLASS_CLASSES = frozenset(
    {"nowhere", "karyotype_46", "normal_keyword_context", "same_sample_link"}
)
SOURCE_PRIORITY = ("birth_outcome", "cnv_seq", "other")
FINE_SOURCE_PRIORITY = (
    "birth_outcome",
    "karyotype_46",
    "normal_keyword_context",
    "same_sample_link",
    "nowhere",
)
SOURCE_DESC = {
    "birth_outcome": (
        "Healthy birth / delivery narrative "
        "(顺产/剖宫产/孕…周 + 女/男/体重, etc.)"
    ),
    "cnv_seq": (
        "Present in Feishu CNV sheet (`read_cnv_df` / `merge_cnv_df`); "
        "Normal comes from CNV-seq rather than clinical follow-up"
    ),
    "other": (
        "No birth narrative and not in `cnv_df` "
        "(bare Normal, karyotype text, keyword-only, or same-sample link)"
    ),
}


def _empty_to_nan(val: Any) -> Any:
    if us._is_empty_label_value(val):
        return np.nan
    text = str(val).strip()
    if text.lower() in {"na", "none", "null"}:
        return np.nan
    return val


def classify_normal_evidence(text: str) -> str:
    """Classify a single follow-up cell that parses as (or looks) Normal."""
    text = us._normalize_label_text(text)
    if not text:
        return "empty"
    if us._SAME_SAMPLE_LINK_RE.search(text):
        return "same_sample_link"
    if us._DISOMY_RE.search(text):
        return "karyotype_46"
    if any(k in text for k in ("顺产", "剖宫产", "活产", "足月")):
        return "birth_outcome"
    if ("孕" in text) and any(k in text for k in ("女", "男", "斤", "kg", "g", "重")):
        return "birth_outcome"
    if any(k in text for k in us._NORMAL_BIRTH_KEYWORDS):
        return "birth_outcome"
    bare = {
        "normal",
        "n",
        "阴性",
        "无异常",
        "未见异常",
        "未见明显异常",
        "正常",
        "单胎",
    }
    if text.strip().lower() in bare or re.fullmatch(
        r"N(?:ormal)?", text, flags=re.IGNORECASE
    ):
        return "nowhere"
    if us._is_normal_text(text):
        return "normal_keyword_context"
    return "non_normal"


def _extract_same_sample_link(texts: list[str]) -> Optional[str]:
    for text in texts:
        m = us._SAME_SAMPLE_LINK_RE.search(us._normalize_label_text(text))
        if m:
            return us.normalize_sample_name(m.group(1))
    return None


def _collect_normal_hits(row: pd.Series) -> list[tuple[str, str, str]]:
    """Return (col, evidence_kind, text) for Normal-like filled label cols."""
    hits: list[tuple[str, str, str]] = []
    for col in LABEL_COLS:
        val = row.get(col)
        text = us._normalize_label_text(val)
        if not text:
            continue
        parsed = us._parse_ploidy_label_from_text(val)
        kind = classify_normal_evidence(text)
        if kind == "same_sample_link":
            hits.append((col, kind, text))
            continue
        if parsed == "Normal" or us._is_normal_text(text) or kind != "non_normal":
            if kind == "non_normal":
                continue
            hits.append((col, kind, text))
    return hits


def _pick_best(hits: list[tuple[str, str, str]]) -> Optional[tuple[str, str, str]]:
    for kind in FINE_SOURCE_PRIORITY:
        for col, k, text in hits:
            if k == kind:
                return col, k, text
    return None


def _resolve_source_for_row(
    row: pd.Series,
    by_sample: dict[str, pd.Series],
    depth: int = 0,
) -> dict[str, Any]:
    hits = _collect_normal_hits(row)
    best = _pick_best(hits)
    link = _extract_same_sample_link(
        [us._normalize_label_text(row.get(c)) for c in LABEL_COLS]
    )

    # Prefer own clinical evidence over a same-sample link.
    own_clinical = [
        h for h in hits if h[1] in {"birth_outcome", "karyotype_46", "normal_keyword_context"}
    ]
    if own_clinical:
        best = _pick_best(own_clinical)
        assert best is not None
        return {
            "source_class": best[1],
            "source_col": best[0],
            "source_db_field": LABEL_COL_TO_DB[best[0]],
            "source_text": best[2],
            "same_sample_link": link or "",
            "source_via_link": False,
            "linked_source_class": "",
        }

    nowhere_hits = [h for h in hits if h[1] == "nowhere"]
    if nowhere_hits and not link:
        best = nowhere_hits[0]
        return {
            "source_class": "nowhere",
            "source_col": best[0],
            "source_db_field": LABEL_COL_TO_DB[best[0]],
            "source_text": best[2],
            "same_sample_link": "",
            "source_via_link": False,
            "linked_source_class": "",
        }

    if link and depth < 2:
        linked = by_sample.get(link)
        if linked is not None:
            linked_res = _resolve_source_for_row(linked, by_sample, depth=depth + 1)
            # If the only local signal is a link (or bare + link), inherit.
            if not nowhere_hits or linked_res["source_class"] != "nowhere":
                return {
                    "source_class": linked_res["source_class"],
                    "source_col": linked_res["source_col"],
                    "source_db_field": linked_res["source_db_field"],
                    "source_text": linked_res["source_text"],
                    "same_sample_link": link,
                    "source_via_link": True,
                    "linked_source_class": linked_res["source_class"],
                }
            # Linked sample also nowhere → still nowhere, but record the link.
            return {
                "source_class": "nowhere",
                "source_col": nowhere_hits[0][0],
                "source_db_field": LABEL_COL_TO_DB[nowhere_hits[0][0]],
                "source_text": nowhere_hits[0][2],
                "same_sample_link": link,
                "source_via_link": True,
                "linked_source_class": "nowhere",
            }

    if best is not None:
        return {
            "source_class": best[1],
            "source_col": best[0],
            "source_db_field": LABEL_COL_TO_DB[best[0]],
            "source_text": best[2],
            "same_sample_link": link or "",
            "source_via_link": False,
            "linked_source_class": "",
        }

    return {
        "source_class": "nowhere",
        "source_col": "",
        "source_db_field": "",
        "source_text": "",
        "same_sample_link": link or "",
        "source_via_link": bool(link),
        "linked_source_class": "",
    }


def fetch_db_label_table() -> pd.DataFrame:
    with AIPTDatabase() as db:
        main = db.fetch_table("样本总表").to_pandas()
    renamed = us.rename_from_db_to_feishu(main)
    renamed["_sample"] = renamed["sample"].map(us.normalize_sample_name)
    for col in LABEL_COLS:
        renamed[col] = renamed[col].map(_empty_to_nan)
    # Keep one row per normalized sample (first).
    renamed = renamed.drop_duplicates(subset=["_sample"], keep="first")
    return renamed


def build_toxic_with_labels(
    toxic: pd.DataFrame,
    db_labels: pd.DataFrame,
) -> pd.DataFrame:
    tox = toxic.copy()
    tox["_sample"] = tox["sample"].map(us.normalize_sample_name)
    label_keep = ["_sample", "sample"] + LABEL_COLS + [
        c
        for c in (
            "week",
            "age",
            "state",
            "reproductive_history",
            "conception_mode",
            "HCG",
            "timepoint",
        )
        if c in db_labels.columns
    ]
    db_slim = db_labels[label_keep].rename(columns={"sample": "db_sample"})
    merged = tox.merge(db_slim, on="_sample", how="left")
    missing = merged.loc[merged["db_sample"].isna(), "sample"].astype(str).tolist()
    if missing:
        raise RuntimeError(f"Samples missing from 样本总表: {missing}")

    by_sample = {
        us.normalize_sample_name(r["_sample"]): r
        for _, r in db_labels.iterrows()
    }
    # Ensure toxic rows themselves are resolvable even if only in tox merge.
    for _, r in merged.iterrows():
        by_sample[us.normalize_sample_name(r["_sample"])] = r

    resolved = merged.apply(
        lambda row: pd.Series(_resolve_source_for_row(row, by_sample)),
        axis=1,
    )
    out = pd.concat([merged.reset_index(drop=True), resolved], axis=1)
    out["n_filled_label_cols"] = out[LABEL_COLS].notna().sum(axis=1)
    out["filled_label_cols"] = out.apply(
        lambda r: ",".join(c for c in LABEL_COLS if pd.notna(r.get(c))),
        axis=1,
    )
    # Drop helper keys from the published sheet.
    drop = [c for c in ("_sample",) if c in out.columns]
    return out.drop(columns=drop)


def attach_cnv_and_reclassify(sheet: pd.DataFrame, cnv_df: pd.DataFrame) -> pd.DataFrame:
    """If a weak DB source is in ``cnv_df``, the Normal label is CNV-seq."""
    out = sheet.copy()
    tmp = pd.DataFrame({"sample": out["sample"].map(us.normalize_sample_name)})
    attached = us.merge_cnv_df(tmp, cnv_df)
    for src, dst in (
        ("cnv_label", "cnv_label"),
        ("cnv_gender", "cnv_gender"),
        ("purity", "cnv_purity"),
    ):
        out[dst] = attached[src].values if src in attached.columns else np.nan

    cnv_ids = (
        set(cnv_df["sample"].map(us.normalize_sample_name).astype(str))
        if not cnv_df.empty and "sample" in cnv_df.columns
        else set()
    )
    norm = out["sample"].map(us.normalize_sample_name).astype(str)
    in_cnv = out["cnv_label"].notna() & ~out["cnv_label"].map(us._is_empty_label_value)
    out["in_cnv_df"] = in_cnv
    out["cnv_match"] = np.where(
        ~in_cnv,
        "",
        np.where(norm.isin(cnv_ids), "exact", "prefix"),
    )

    out["db_source_class"] = out["source_class"]
    out["db_source_text"] = out["source_text"]
    reclass = in_cnv & out["source_class"].isin(CNV_RECLASS_CLASSES)
    out.loc[reclass, "source_class"] = "cnv_seq"
    out.loc[reclass, "source_col"] = "cnv_df"
    out.loc[reclass, "source_db_field"] = "CNV sheet (read_cnv_df / merge_cnv_df)"
    out.loc[reclass, "source_text"] = out.loc[reclass, "cnv_label"].map(
        us._normalize_label_text
    )
    return out


def _norm_cnv_gender(val: Any) -> str:
    if us._is_empty_label_value(val):
        return ""
    text = str(val).strip().lower()
    if text in {"male", "m", "男", "boy", "男胎"}:
        return "male"
    if text in {"female", "f", "女", "girl", "女胎"}:
        return "female"
    return text


def collapse_source_class(sheet: pd.DataFrame) -> pd.DataFrame:
    """Keep birth_outcome / cnv_seq; fold remaining fine classes into other."""
    out = sheet.copy()
    if "source_class_fine" not in out.columns:
        out["source_class_fine"] = out["source_class"]
    out["cnv_gender"] = out["cnv_gender"].map(_norm_cnv_gender)
    out["source_class"] = np.where(
        out["source_class"].isin(["birth_outcome", "cnv_seq"]),
        out["source_class"],
        "other",
    )
    return out


def _md_escape(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    text = str(val).replace("\n", " ").replace("|", "\\|")
    return text


def _md_table(df: pd.DataFrame, cols: Optional[list[str]] = None) -> list[str]:
    use = list(cols) if cols is not None else list(df.columns)
    header = "| " + " | ".join(use) + " |"
    sep = "| " + " | ".join("---" for _ in use) + " |"
    rows = [
        "| " + " | ".join(_md_escape(r[c]) for c in use) + " |"
        for _, r in df.iterrows()
    ]
    return [header, sep, *rows]


def _group_counts(df: pd.DataFrame) -> pd.Series:
    return (
        df["source_class"]
        .value_counts()
        .reindex(list(SOURCE_PRIORITY), fill_value=0)
    )


def _gender_counts(df: pd.DataFrame) -> pd.DataFrame:
    cnv = df.loc[df["source_class"] == "cnv_seq"].copy()
    gender = cnv["cnv_gender"].replace("", "unknown").fillna("unknown")
    gender = gender.where(gender.isin(["male", "female"]), "unknown")
    tab = (
        gender.value_counts()
        .reindex(["male", "female", "unknown"], fill_value=0)
        .rename_axis("cnv_gender")
        .reset_index(name="n")
    )
    n = int(tab["n"].sum())
    tab["pct"] = np.where(n > 0, tab["n"] / n, 0.0)
    return tab


def write_report(pool: pd.DataFrame, out_md: Path) -> None:
    toxic = pool.loc[pool["toxic"].astype(bool)].copy()
    ok = pool.loc[~pool["toxic"].astype(bool)].copy()
    n_tox = len(toxic)
    n_ok = len(ok)
    tox_counts = _group_counts(toxic)
    ok_counts = _group_counts(ok)

    lines: list[str] = []
    lines.append("# Normal label-source audit (toxic vs OK)")
    lines.append("")
    lines.append(
        f"Expanded Normal pool (n={len(pool)}: toxic {n_tox}, OK {n_ok}) from "
        "`candidate_mad_scores.tsv`, joined to NocoDB `样本总表` via `db_helper` "
        "+ `rename_from_db_to_feishu` (`label_1`…`label_9`). Weak DB classes "
        "(`nowhere` / `karyotype_46` / `normal_keyword_context` / "
        "`same_sample_link`) that match Feishu CNV via `read_cnv_df` + "
        "`merge_cnv_df` are `cnv_seq`. Published classes are "
        "`birth_outcome` / `cnv_seq` / `other`."
    )
    lines.append("")
    lines.append("## Source classes")
    lines.append("")
    lines.append(
        "Birth outcome wins over CNV-seq. Remaining non-birth samples in "
        "`cnv_df` are `cnv_seq`. Everything else is `other`."
    )
    lines.append("")
    for key, desc in SOURCE_DESC.items():
        lines.append(f"- **`{key}`** — {desc}")
    lines.append("")

    lines.append("## Toxic vs OK")
    lines.append("")
    lines.append("| source_class | toxic n | toxic % | OK n | OK % |")
    lines.append("|--------------|--------:|--------:|-----:|-----:|")
    for cls in SOURCE_PRIORITY:
        tn = int(tox_counts.get(cls, 0))
        on = int(ok_counts.get(cls, 0))
        lines.append(
            f"| `{cls}` | {tn} | {100 * tn / n_tox:.1f}% | {on} | {100 * on / n_ok:.1f}% |"
        )
    lines.append(
        f"| **total** | **{n_tox}** | **100%** | **{n_ok}** | **100%** |"
    )
    lines.append("")
    lines.append("### Toxic by set")
    lines.append("")
    tox_set = (
        pd.crosstab(toxic["set"], toxic["source_class"])
        .reindex(columns=list(SOURCE_PRIORITY), fill_value=0)
        .reset_index()
    )
    lines.extend(_md_table(tox_set))
    lines.append("")
    lines.append("### OK by set")
    lines.append("")
    ok_set = (
        pd.crosstab(ok["set"], ok["source_class"])
        .reindex(columns=list(SOURCE_PRIORITY), fill_value=0)
        .reset_index()
    )
    lines.extend(_md_table(ok_set))
    lines.append("")

    lines.append("## CNV-seq gender")
    lines.append("")
    lines.append(
        "Fetal sex from `cnv_df` (`预测性别` via `merge_cnv_df`) among "
        "`source_class == cnv_seq` samples."
    )
    lines.append("")
    for title, sub in (("toxic", toxic), ("OK", ok)):
        gtab = _gender_counts(sub)
        n_cnv = int((sub["source_class"] == "cnv_seq").sum())
        lines.append(f"### {title} cnv_seq (n={n_cnv})")
        lines.append("")
        if n_cnv == 0:
            lines.append("_none_")
            lines.append("")
            continue
        lines.extend(_md_table(gtab, ["cnv_gender", "n", "pct"]))
        lines.append("")
        by_g = (
            sub.loc[sub["source_class"] == "cnv_seq"]
            .assign(
                cnv_gender=lambda d: d["cnv_gender"]
                .replace("", "unknown")
                .fillna("unknown")
            )
        )
        gset = (
            pd.crosstab(by_g["set"], by_g["cnv_gender"])
            .reindex(columns=["male", "female", "unknown"], fill_value=0)
            .reset_index()
        )
        lines.extend(_md_table(gset))
        lines.append("")

    def _section(title: str, frame: pd.DataFrame, mask: pd.Series, cols: list[str]) -> None:
        sub = frame.loc[mask].sort_values(["mad_rank", "sample"])
        lines.append(f"## {title} (n={len(sub)})")
        lines.append("")
        if sub.empty:
            lines.append("_none_")
            lines.append("")
            return
        use = [c for c in cols if c in sub.columns]
        lines.extend(_md_table(sub, use))
        lines.append("")

    _section(
        "Toxic cnv_seq",
        toxic,
        toxic["source_class"] == "cnv_seq",
        [
            "mad_rank",
            "sample",
            "set",
            "cnv_gender",
            "cnv_match",
            "cnv_label",
            "source_class_fine",
            "db_source_class",
            "db_source_text",
        ],
    )
    _section(
        "Toxic birth_outcome",
        toxic,
        toxic["source_class"] == "birth_outcome",
        [
            "mad_rank",
            "sample",
            "set",
            "source_col",
            "source_text",
        ],
    )
    _section(
        "Toxic other",
        toxic,
        toxic["source_class"] == "other",
        [
            "mad_rank",
            "sample",
            "set",
            "source_class_fine",
            "db_source_class",
            "source_col",
            "source_text",
            "same_sample_link",
            "in_cnv_df",
        ],
    )

    lines.append("## Takeaway")
    lines.append("")
    t_birth = int(tox_counts["birth_outcome"])
    t_cnv = int(tox_counts["cnv_seq"])
    t_other = int(tox_counts["other"])
    o_birth = int(ok_counts["birth_outcome"])
    o_cnv = int(ok_counts["cnv_seq"])
    o_other = int(ok_counts["other"])
    tox_g = _gender_counts(toxic)
    tox_male = int(tox_g.loc[tox_g["cnv_gender"] == "male", "n"].sum())
    tox_female = int(tox_g.loc[tox_g["cnv_gender"] == "female", "n"].sum())
    tox_unk = int(tox_g.loc[tox_g["cnv_gender"] == "unknown", "n"].sum())
    ok_g = _gender_counts(ok)
    ok_male = int(ok_g.loc[ok_g["cnv_gender"] == "male", "n"].sum())
    ok_female = int(ok_g.loc[ok_g["cnv_gender"] == "female", "n"].sum())
    ok_unk = int(ok_g.loc[ok_g["cnv_gender"] == "unknown", "n"].sum())
    lines.append(
        f"Toxic (n={n_tox}): **{t_birth}** birth_outcome "
        f"({100 * t_birth / n_tox:.1f}%), **{t_cnv}** cnv_seq "
        f"({100 * t_cnv / n_tox:.1f}%), **{t_other}** other "
        f"({100 * t_other / n_tox:.1f}%). "
        f"CNV-seq gender: male {tox_male}, female {tox_female}, unknown {tox_unk}."
    )
    lines.append("")
    lines.append(
        f"OK (n={n_ok}): **{o_birth}** birth_outcome "
        f"({100 * o_birth / n_ok:.1f}%), **{o_cnv}** cnv_seq "
        f"({100 * o_cnv / n_ok:.1f}%), **{o_other}** other "
        f"({100 * o_other / n_ok:.1f}%). "
        f"CNV-seq gender: male {ok_male}, female {ok_female}, unknown {ok_unk}."
    )
    lines.append("")
    lines.append(
        "OK samples are more often `birth_outcome`; toxics are relatively "
        "enriched for `cnv_seq` (miscarriage / embryo-karyotype Normals without "
        "a live-birth record)."
    )
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


FRONT_COLS = [
    "mad_rank",
    "sample",
    "db_sample",
    "set",
    "group",
    "toxic",
    "mad_score",
    "max_mad_chr",
    "max_mad_track",
    "ff_before_mq",
    "depth_qc",
    "label",
    "source_class",
    "source_class_fine",
    "db_source_class",
    "source_col",
    "source_db_field",
    "source_text",
    "source_via_link",
    "same_sample_link",
    "linked_source_class",
    "in_cnv_df",
    "cnv_match",
    "cnv_label",
    "cnv_gender",
    "cnv_purity",
    "db_source_text",
    "n_filled_label_cols",
    "filled_label_cols",
]


def _order_cols(df: pd.DataFrame) -> pd.DataFrame:
    front = [c for c in FRONT_COLS if c in df.columns]
    mid = [c for c in LABEL_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in front + mid]
    return df[front + mid + rest]


@click.command()
@click.option(
    "--candidates",
    "cand_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=str(DEFAULT_CANDIDATES),
    show_default=True,
)
@click.option(
    "--outdir",
    type=click.Path(path_type=Path, file_okay=False),
    default=str(DEFAULT_OUTDIR),
    show_default=True,
)
def main(cand_path: Path, outdir: Path) -> None:
    """Build DB-mapped pool samplesheet + 3-class Normal label-source report."""
    outdir.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(cand_path, sep="\t")
    if "toxic" not in cand.columns:
        raise click.ClickException(f"{cand_path} missing column toxic")
    cand["toxic"] = cand["toxic"].astype(str).str.lower().isin(["true", "1", "yes"])
    console.print(
        f"pool n={len(cand)} toxic={int(cand['toxic'].sum())} "
        f"ok={int((~cand['toxic']).sum())} from {cand_path}"
    )

    console.print("fetching 样本总表 …")
    db_labels = fetch_db_label_table()
    console.print(f"DB samples n={len(db_labels)}")

    sheet = build_toxic_with_labels(cand, db_labels)
    console.print("fetching CNV sheet (read_cnv_df) …")
    cnv_df = read_cnv_df()
    console.print(f"cnv_df n={len(cnv_df)}")
    sheet = attach_cnv_and_reclassify(sheet, cnv_df)
    sheet = collapse_source_class(sheet)
    if "toxic" not in sheet.columns:
        sheet = sheet.merge(cand[["sample", "toxic"]], on="sample", how="left")

    old_path = outdir / "toxic_samplesheet_with_db_labels.tsv"
    if old_path.is_file():
        old = pd.read_csv(old_path, sep="\t")
        extra = [c for c in old.columns if c not in sheet.columns and c != "sample"]
        if extra:
            sheet = sheet.merge(old[["sample"] + extra], on="sample", how="left")

    sheet = _order_cols(sheet)
    pool_path = outdir / "pool_label_source.tsv"
    sheet.to_csv(pool_path, sep="\t", index=False)

    toxic = sheet.loc[sheet["toxic"].astype(bool)].copy()
    ok = sheet.loc[~sheet["toxic"].astype(bool)].copy()
    tox_path = outdir / "toxic_samplesheet_with_db_labels.tsv"
    ok_path = outdir / "ok_samplesheet_with_db_labels.tsv"
    toxic.to_csv(tox_path, sep="\t", index=False)
    ok.to_csv(ok_path, sep="\t", index=False)

    rows = []
    for group_name, sub in (("toxic", toxic), ("ok", ok)):
        n = len(sub)
        for cls in SOURCE_PRIORITY:
            n_cls = int((sub["source_class"] == cls).sum())
            rows.append(
                {
                    "group": group_name,
                    "source_class": cls,
                    "n": n_cls,
                    "pct": n_cls / n if n else 0.0,
                }
            )
        if group_name == "toxic" or True:
            gtab = _gender_counts(sub)
            for _, g in gtab.iterrows():
                rows.append(
                    {
                        "group": f"{group_name}_cnv_seq_gender",
                        "source_class": str(g["cnv_gender"]),
                        "n": int(g["n"]),
                        "pct": float(g["pct"]),
                    }
                )
    summary = pd.DataFrame(rows)
    summary_path = outdir / "toxic_label_source_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.4f")

    report_path = outdir / "toxic_label_source_report.md"
    write_report(sheet, report_path)

    meta = {
        "n_pool": int(len(sheet)),
        "n_toxic": int(len(toxic)),
        "n_ok": int(len(ok)),
        "cand_path": str(cand_path.resolve()),
        "outdir": str(outdir.resolve()),
        "source_counts_toxic": {
            cls: int((toxic["source_class"] == cls).sum()) for cls in SOURCE_PRIORITY
        },
        "source_counts_ok": {
            cls: int((ok["source_class"] == cls).sum()) for cls in SOURCE_PRIORITY
        },
        "cnv_seq_gender_toxic": _gender_counts(toxic).set_index("cnv_gender")["n"].to_dict(),
        "cnv_seq_gender_ok": _gender_counts(ok).set_index("cnv_gender")["n"].to_dict(),
        "label_col_to_db": LABEL_COL_TO_DB,
    }
    (outdir / "toxic_label_source_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    console.print(summary.to_string(index=False))
    console.print(f"[green]OK[/green] wrote {pool_path}")
    console.print(f"[green]OK[/green] wrote {tox_path}")
    console.print(f"[green]OK[/green] wrote {ok_path}")
    console.print(f"[green]OK[/green] wrote {report_path}")


if __name__ == "__main__":
    main()
