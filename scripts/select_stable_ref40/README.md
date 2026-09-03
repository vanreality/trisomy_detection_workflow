# Select stable ref_40

Offline toolkit to (1) summarize episcore / zscore calculation sources, (2) fetch
missing zscore percentages, and (3) choose a Normal+dev `ref_40` that preserves
early_ref mean/std and ezscore `pred_label` (cutoff 4.5).

## Inputs

| Input | Path |
|-------|------|
| Episcore samplesheet | `/lustre1/cqyi/syfan/nipt_article_plot/episcore_result_samplesheet.csv` |
| Percentage (partial) | `/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260607-ref_40/percentage.csv` |
| Meta samplesheet | `/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet.csv` |
| Ezscore refs (n=25) | `/lustre1/cqyi/myli/bert/analysis_nipt/multiomics/chr_stats_reference_samples.txt` |

Combos:

- Episcore: early.config recall=0.65, threshold=0.5
- Zscore: recall=0.95, cutoff=0.85

## Run

```bash
bash /lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/select_stable_ref40/run_all.sh

# or step-wise via singularity + common_tools.sif:
# build_source_tables.py → select_ref40.py → write_updated_meta.py → render_plots.py
```

Useful knobs for `select_ref40.py`:

- `--n-random` / `--n-swap-rounds` / `--seed`
- `--exclude-sample ''` to allow all Normal+dev (default historically excluded `PTAY0586P8S1`)

### Re-select vs meta (recommended when meta ≠ recomputed baseline)

`select_ref40.py` optimizes against a **recomputed** early_ref baseline. That can
disagree with stored meta `final_zscores` near the 4.5 cutoff (e.g. PTAY1472P9S1:
meta Gray_T16 at 4.441 vs recompute T16 at 4.554). To keep meta/ref_17 labels:

```bash
singularity exec -B /lustre1/cqyi:/lustre1/cqyi \
  /lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif \
  python3 /lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/select_stable_ref40/reselect_vs_meta.py
```

This matches meta `final_zscores` pred masks (excluding emergency), hard-protects
borderline Gray samples (`PTAY1472P9S1`, `PTAY1253P6H1`, `PTAY0704P7H1`), and
rewrites `temporary_updated_samplesheet_ref40.csv`.

## Outputs

Directory: `/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260812-stable_ref40/`
(prior run: `20260730-stable_ref40/`)

| File | Description |
|------|-------------|
| `beta.csv` | Episcore source table (all meta samples) |
| `percentage.csv` | Zscore percentages at cutoff=0.85 |
| `ref40_samples.txt` | Selected ref_40 list |
| `baseline_score.tsv` / `ref40_score.tsv` | Recalculated scores |
| `reference_meanstd_compare.tsv` | early_ref vs ref_40 mean/std |
| `pred_label_compare.tsv` | Pred-label diffs vs meta |
| `selection_summary.json` | Search metrics |
| `temporary_updated_samplesheet_ref40.csv` | Updated meta (`ref_type` + `*_zscores` + `pred_label`) |
| `ref40_episcore_matrix.tsv` | Episcore reference (wide; early_reference format) |
| `ref40_zscore_matrix.csv` | Zscore reference (long; percentage + adj_percentage) |
| `ref40_ezscore_matrix.csv` | EZscore chr mu/sigma over fixed 25 ezscore refs |

Build matrices only:

```bash
singularity exec -B /lustre1/cqyi:/lustre1/cqyi \
  /lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif \
  python3 scripts/select_stable_ref40/build_ref40_matrices.py \
  --output-dir /lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260812-stable_ref40
```

Also copied: `/lustre1/cqyi/syfan/nipt_article_plot/temporary_updated_samplesheet_ref40.csv`

## Notebooks

- `notebooks/aipt_2.0/summarize_score_sources.ipynb`
- `notebooks/aipt_2.0/select_stable_ref40.ipynb`
