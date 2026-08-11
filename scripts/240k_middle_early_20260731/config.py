"""Paths for 20260731 early-allosomes + middle-normal 240k analysis."""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]  # workflow/episcore

INPUT_DIR = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260731-240k_middle_normal_samples_plus_early_allosomes_samples"
)
OUTPUT_DIR = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260731-240k_middle_normal_samples_plus_early_allosomes_samples"
)

# Prior early cohorts (background for conventional plots)
OLD_SAMPLESHEET_20260416 = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260416-240k.csv"
)
OLD_SAMPLESHEET_20260507 = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260507-240k_XO_samples.csv"
)
OLD_SAMPLESHEET_20260720 = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260720-240k_early_allosomes_samples/mqres.csv"
)
OLD_LABELS_20260720 = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/20260720-240k_early_allosomes_samples/cohort_labels.csv"
)
OLD_FF_20260416 = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260416-early_samples-240krecall0/collect_reports/summary_report.tsv"
)
OLD_FF_20260507 = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260507-240k_XO_samples-240krecall0/collect_reports/summary_report.tsv"
)
OLD_FF_20260720 = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260720-240k_early_allosomes_samples/collect_reports/summary_report.tsv"
)
OLD_BETA_20260416 = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260416-early_samples-240krecall0/extract_beta_value"
)
OLD_BETA_20260507 = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260507-240k_XO_samples-240krecall0/extract_beta_value"
)
OLD_BETA_20260720 = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260720-240k_early_allosomes_samples/extract_beta_value"
)
# Prior conventional recall tables (merge new-early curves onto these)
PRIOR_EPISCORE_TSV = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260720-240k_early_allosomes_samples/tables/chrX_episcore_vs_recall.tsv"
)
PRIOR_ZSCORE_TSV = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260720-240k_early_allosomes_samples/tables/chrX_zscore_vs_recall.tsv"
)
PRIOR_CHRY_FF = Path(
    "/lustre1/cqyi/AIPT_2.0/results/episcore_output/"
    "20260720-240k_early_allosomes_samples/tables/chry_ff.tsv"
)

# Input sheets
EARLY_MQRES = INPUT_DIR / "early_allosomes_mqres.csv"
EARLY_META = INPUT_DIR / "early_allosomes_meta.csv"
MIDDLE_MQRES = INPUT_DIR / "middle_normal_mqres.csv"

# Recall CpG lists (240k panel)
CPG_RECALL_DIR = Path(
    "/lustre1/cqyi/AIPT_2.0/data/meta/episcore/"
    "20260525-grid_search_240k_panel_240k_model/recall_list_240k"
)
FULL_240K_CPG_LIST = PROJECT_DIR / "assets" / "cpgs_in_240k_probes.txt"

EPISCORE_THRESHOLD = 0.5
ZSCORE_CUTOFF = 0.85

# Derived under INPUT_DIR
COHORT_LABELS = INPUT_DIR / "cohort_labels.csv"
MIDDLE_GENDER = INPUT_DIR / "middle_gender_assigned.csv"
NF_SAMPLESHEET_EARLY = INPUT_DIR / "samplesheet_nf_early.csv"
NF_SAMPLESHEET_MIDDLE = INPUT_DIR / "samplesheet_nf_middle.csv"
EPISCORE_SAMPLES_META = INPUT_DIR / "episcore_samples_meta.csv"  # conventional early_ref
ZSCORE_SAMPLES_META = INPUT_DIR / "zscore_samples_meta.csv"
FEATURES_SAMPLES_META = INPUT_DIR / "features_samples_meta.csv"  # early+middle deconv/beta
MALE_REF_EPISCORE_META = INPUT_DIR / "male_ref_episcore_meta.csv"
FEMALE_REF_EPISCORE_META = INPUT_DIR / "female_ref_episcore_meta.csv"
MALE_REF_ZSCORE_META = INPUT_DIR / "male_ref_zscore_meta.csv"
FEMALE_REF_ZSCORE_META = INPUT_DIR / "female_ref_zscore_meta.csv"

# Outputs
EARLY_NF_OUT = OUTPUT_DIR / "early_nf"
MIDDLE_NF_OUT = OUTPUT_DIR / "middle_nf"
EPISCORE_RECALL_DIR = OUTPUT_DIR / "episcore_recall_conventional"
ZSCORE_RECALL_DIR = OUTPUT_DIR / "zscore_recall_conventional"
FEATURES_DIR = OUTPUT_DIR / "features_recall"
MALE_REF_EPISCORE_DIR = OUTPUT_DIR / "male_ref_episcore_recall"
FEMALE_REF_EPISCORE_DIR = OUTPUT_DIR / "female_ref_episcore_recall"
MALE_REF_ZSCORE_DIR = OUTPUT_DIR / "male_ref_zscore_recall"
FEMALE_REF_ZSCORE_DIR = OUTPUT_DIR / "female_ref_zscore_recall"
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"

CHRY_FF_TSV = TABLES_DIR / "chry_ff.tsv"
FEATURE_MATRIX_TSV = TABLES_DIR / "early_middle_feature_matrix.tsv"
EPISCORE_COLLECTED = TABLES_DIR / "chrX_episcore_vs_recall.tsv"
ZSCORE_COLLECTED = TABLES_DIR / "chrX_zscore_vs_recall.tsv"
MALE_REF_EPISCORE_COLLECTED = TABLES_DIR / "male_ref_chrX_episcore_vs_recall.tsv"
MALE_REF_ZSCORE_COLLECTED = TABLES_DIR / "male_ref_chrX_zscore_vs_recall.tsv"
FEMALE_REF_EPISCORE_COLLECTED = TABLES_DIR / "female_ref_chrX_episcore_vs_recall.tsv"
FEMALE_REF_ZSCORE_COLLECTED = TABLES_DIR / "female_ref_chrX_zscore_vs_recall.tsv"

SINGULARITY_IMAGE = PROJECT_DIR / "containers" / "common_tools.sif"
MAIN_NF = PROJECT_DIR / "main.nf"
