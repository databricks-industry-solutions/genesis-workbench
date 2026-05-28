# GWAS Analysis

## Introduction

Genome-Wide Association Study (GWAS) analysis identifies statistical associations between genetic variants and a phenotype (trait or disease). This workflow uses the Glow library to perform logistic regression GWAS on VCF genotype data against a user-supplied phenotype file.

## What It Achieves

- Reads a multi-sample VCF file and filters to samples with phenotype data
- Normalizes variants against the GRCh38 reference genome
- Computes Hardy-Weinberg equilibrium (HWE) statistics and filters variants
- Runs logistic regression with approximate Firth correction across specified chromosomes
- Produces a results table with per-variant p-values for Manhattan plot visualization

## How to Use

1. Navigate to **Disease Biology > GWAS Analysis** tab
2. Select VCF source: enter a path directly or pick from a completed Variant Calling run
3. Enter the phenotype file path (CSV or TSV on a UC Volume)
4. Configure analysis parameters:
   - **Phenotype column**: Column name containing case/control labels (default: `phenotype`)
   - **Contigs**: Chromosomes to analyze, comma-separated (default: `6`)
   - **HWE cutoff**: Hardy-Weinberg p-value filter (default: `0.01`)
   - **Firth p-value threshold**: Triggers Firth correction for rare variants (default: `0.01`)
5. Set MLflow experiment and run names
6. Click **Start GWAS Analysis**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| VCF file path | Multi-sample VCF with genotype data | `/Volumes/catalog/schema/gwas_data/sample_vcf/ALL.chr6...vcf.gz` |
| Phenotype file | CSV/TSV with sample IDs and phenotype labels | `/Volumes/catalog/schema/gwas_data/sample_phenotype/breast_cancer_phenotype.tsv` |
| Phenotype column | Column name for case (1) / control (0) labels | `phenotype` |
| Contigs | Chromosomes to analyze | `6` or `1,2,3` |

### Phenotype File Format

The phenotype file must contain:
- A sample ID column (`Sample name`, `sampleId`, `sample_id`, or first column)
- A phenotype column with binary labels (integer 0/1 or string labels that get auto-mapped)

### Outputs

- **GWAS results table**: Per-variant p-values, stored as a Delta table
- **Manhattan plot**: Visualized in the results viewer (significant hits highlighted)
- **Summary metrics**: Total variants tested, significant/suggestive hits, minimum p-value

## How It's Implemented

### Architecture

1. **UI** (`modules/core/app/views/disease_biology.py`): GWAS Analysis tab with VCF source selector and form
2. **Backend** (`modules/core/app/utils/disease_biology.py`): `start_gwas_analysis()` creates MLflow run and triggers job
3. **Job** (`modules/disease_biology/gwas/gwas_v1/resources/gwas_analysis.job.yml`): Multi-task workflow

### Workflow Pipeline

```
prepare_phenotype → run_gwas → save_results → mark_success/mark_failure
```

1. **Prepare Phenotype** (`03_prepare_phenotype.py`): Reads phenotype CSV/TSV, normalizes column names, maps string labels to integers, writes Delta table
2. **Run GWAS** (`04_gwas_analysis.py`): Loads VCF with Glow, filters to phenotyped samples, normalizes variants, computes HWE, runs logistic regression
3. **Save Results** (`05_save_results.py`): Computes summary statistics, logs metrics and final status to MLflow

### MLflow Tracking

- Logs parameters: VCF path, phenotype path, contigs, HWE cutoff
- Logs metrics: total variants tested, significant hits, suggestive hits, min p-value
- Tags: `feature=gwas`, `job_status=gwas_complete` or `failed`

### Key Files

- `modules/core/app/views/disease_biology.py` — UI (GWAS Analysis tab)
- `modules/core/app/utils/disease_biology.py` — `start_gwas_analysis()`, `pull_gwas_results()`
- `modules/disease_biology/gwas/gwas_v1/resources/gwas_analysis.job.yml` — Job definition
- `modules/disease_biology/gwas/gwas_v1/notebooks/03_prepare_phenotype.py` — Phenotype prep
- `modules/disease_biology/gwas/gwas_v1/notebooks/04_gwas_analysis.py` — GWAS execution
- `modules/disease_biology/gwas/gwas_v1/notebooks/05_save_results.py` — Results and MLflow
