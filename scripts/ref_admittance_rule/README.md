# Reference-pool admittance rules

Offline analysis of why some 40+40 ref_free draws are perfect (FP+FN=0) and
others are bad (FP+FN≥5). Replays the 20260810 `fixed_flags_ez45` experiment
(seed=42, ez=4.5, ep 0.5/0.65, z 0.85/0.95) and stores compact membership so
we can QC the 96-sample **dev Normal** pool.

Does **not** change `main.nf`. Results live on lustre (gitignored).

## Questions

1. Are some pool samples persistently in perfect vs bad draws? Do they have
   extreme `ff_before_mq` or per-chr `percentage` / `z_intra` MAD outliers?
2. Do perfect vs bad **sets** differ in FF spread or ez-ref SD (the ezscore
   denominator — FN-dominated errors suggest inflated SD)?
3. Can we write admittance rules, and prove they move ref_free FP+FN vs a
   **matched-N random drop** control?

## Layout

| Script | Role |
|--------|------|
| `score_repeats.py` | Replay 40+40; write `repeats_*.npz` (FP/FN + membership + set stats) |
| `analyze_perfect_vs_bad.py` | Q1 enrichment + Q2 set-level distributions |
| `derive_and_prove_rules.py` | Q3 rules, retrospective proof, admitted list |
| `run_admitted_redraw.py` | Prospective 40+40 from a pool file (smoke / local) |
| `plot_admittance.py` | HTML plots |

## Run (Alioth)

```bash
cd /lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/ref_admittance_rule

# 1. 100k compact rescoring (same seed as 20260810, first 100k of 1e6)
bash submit_score_repeats.sh

# 2. after array finishes
bash submit_analyze.sh \
  /lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/baseline96

# 3. if admitted n>=80, redraw 20k from admitted + random-N control
bash submit_redraw.sh \
  /lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/baseline96/analysis/proof
```

Smoke (20 repeats, no slurm):

```bash
singularity exec -B /lustre1,/lustre2,/appsnew \
  /lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif \
  python3 score_repeats.py --total-repeats 20 --tag smoke
```

## Inputs / outputs

- Input grids: `/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260621-ref_40_rebuild_consider_lib_ng`
- Output root: `/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/`
- Density sanity check vs: `.../20260810-ref_free_pool_size/fp_fn_density/fp_fn_density_ezscore.tsv`

Eval mask matches 20260810: dev trisomy + test, `ff_before_mq≥0.01`, blacklist
`PTAY0577P9S1,PTAY0599P8S1,PTAY0666P7S1,PTAY0682P7S1,PTAY0689P8H1`.

## Rules evaluated

- `ff_tail_5_95` — drop pool FF outside 5th–95th percentile
- `pct_mad_3_5` — drop any-chr \|percentage MAD-z\| > 3.5
- `intra_mad_3_5` — drop any-chr \|hypo/hyper z_intra MAD-z\| > 3.5
- `mad_or_ff` — union of MAD + FF tails
- `toxic_heldout` — toxic membership signature on **even** repeats, evaluated on odd
- `toxic_keep80` — drop highest even-split `toxic_score` until n=80 remain (used for redraw)

Proof always includes a matched-N random drop (K=20). Because a 40+40
draw uses 80/96 samples, “all 80 pass QC” is rare after dropping even a
few pool members — the primary retrospective test is therefore
**Spearman ρ(n_fail_members, FP+FN)** and the dose-response curve, not
the all-80-pass subset. Prospective redraw (new 40+40 from the admitted
pool vs a same-size random pool) is the causal test.

## Results (20260813, 100k seed=42)

Density matches 20260810 1e6 (frac_perfect 0.111 vs 0.112). Errors remain FN-dominated.

- **Q1:** 23 toxic / 41 protective pool samples (Fisher p<0.05, role=either). Top toxic often have percentage MAD outliers (chr14/chr15) or extreme FF, but not all do — membership toxicity is stronger than any single MAD cut.
- **Q2:** Bad 40+40 sets have **larger ez-ref SD** (Cliff's δ ≈ 0.79 on `ez_sd_mean`; chr14 δ ≈ 0.96). That shrinks ezscores and produces FN.
- **Q3:** `toxic_keep80` Spearman ρ(n_fail, FP+FN)=0.26 vs ~0 for matched-N random. Prospective 20k redraw (seed=7):

| pool | n | frac_perfect | mean FP+FN |
|------|---|--------------|------------|
| baseline 96 | 100k | 0.111 | 2.67 |
| admitted toxic_keep80 | 20k | **0.919** | **0.088** |
| random drop 16 | 20k | 0.165 | 2.06 |

Plots: `/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/baseline96/analysis/plots/`

Write-up with PNG figures: `scripts/ref_admittance_rule/report/REPORT.md`
(regenerate figures with `python3 render_report_figures.py`).

Expanded Normal pool (dev+test, depth pass) MAD ranking + CNVseq overlap:

```bash
singularity exec -B /lustre1,/lustre2,/appsnew \
  /lustre1/cqyi/AIPT_2.0/workflow/episcore/containers/common_tools.sif \
  python3 score_expanded_pool_mad.py
```

Outputs: `.../20260813-ref_admittance_rule/expanded_pool_mad/`

Toxic Normal label sources (DB `label_1`…`label_9`):

```bash
# needs notebooks/aipt_1.0 conda env (SSH tunnel / paramiko), not common_tools.sif
python3 audit_toxic_label_sources.py
```

Writes `pool_label_source.tsv`, `toxic_samplesheet_with_db_labels.tsv`, `ok_samplesheet_with_db_labels.tsv`, `toxic_label_source_report.md`. Classes: `birth_outcome` / `cnv_seq` / `other`.

Mqres batch enrichment vs the expanded Normal pool:

```bash
python3 audit_toxic_batches.py
```

Writes `toxic_batch_report.md` + `toxic_batch_enrichment.tsv` (also adds `mqres_batches` onto the DB-mapped toxic sheet).

Per-batch MAD for the 2 multi-batch toxics + pool commonality:

```bash
python3 audit_toxic_multibatch.py
```

Writes `toxic_multibatch_units.tsv` + `toxic_commonality_report.md`.

## Independent test (`ref_admittance_check`)

Replay the **toxic_keep80** procedure on 96 randomly chosen **test** Normals
(`label==Normal`, `depth_qc==pass`, seed=13), excluding those 96 from eval so
`all_96_test` / `toxic_16_excluded` / `random_16_excluded` share the same eval.

Also Monte-Carlo (10k) ez-ref mean/SD vs pool size 20…160 step 10, 22 chr panels.

```bash
cd /lustre1/cqyi/AIPT_2.0/workflow/episcore/scripts/ref_admittance_rule
bash submit_ref_admittance_check.sh
# or only the pool-size ez μ/σ plots:
bash submit_pool_size_ez_stats.sh
```

Outputs: `/lustre1/cqyi/AIPT_2.0/results/episcore_output/20260813-ref_admittance_rule/ref_admittance_check/`
