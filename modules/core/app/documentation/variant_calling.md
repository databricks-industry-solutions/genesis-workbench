# Variant Calling with Parabricks

## Introduction

Variant Calling uses NVIDIA Parabricks to perform GPU-accelerated germline variant calling from paired-end whole-genome sequencing data. It takes raw sequencing reads (FASTQ files) and produces aligned reads (BAM) and a variant call file (VCF) identifying genetic differences from a reference genome.

## What It Achieves

- Aligns paired-end FASTQ reads to a reference genome (GRCh38) using GPU-accelerated BWA-MEM
- Calls germline variants (SNPs and indels) using GPU-accelerated HaplotypeCaller
- Produces a BAM file (aligned reads) and VCF file (variant calls) ready for downstream GWAS or annotation workflows

## How to Use

1. Navigate to **Disease Biology > Variant Calling** tab
2. Enter the UC Volume paths for your paired-end FASTQ files (Read 1 and Read 2)
3. Select the reference genome (GRCh38 is pre-staged during setup, or provide a custom path)
4. Specify the output volume path where BAM and VCF files will be written
5. Configure MLflow experiment and run names for tracking
6. Click **Start Variant Calling**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| FASTQ Read 1 | Path to paired-end read 1 | `/Volumes/catalog/schema/gwas_data/sample_fastq/sample_1.fq.gz` |
| FASTQ Read 2 | Path to paired-end read 2 | `/Volumes/catalog/schema/gwas_data/sample_fastq/sample_2.fq.gz` |
| Reference Genome | GRCh38 (pre-staged) or custom path | Pre-staged during module setup |
| Output Volume Path | UC Volume for output files | `/Volumes/catalog/schema/gwas_data` |

### Outputs

- **BAM file**: Aligned reads in BAM format
- **VCF file**: Germline variant calls (SNPs + indels)
- Output VCF can be used directly in GWAS Analysis or VCF Ingestion workflows

## How It's Implemented

### Architecture

1. **UI** (`modules/core/app/views/disease_biology.py`): Streamlit form collects inputs and triggers the workflow
2. **Backend** (`modules/core/app/utils/disease_biology.py`): `start_parabricks_alignment()` creates an MLflow run and triggers the Databricks job
3. **Job** (`modules/disease_biology/gwas/gwas_v1/resources/parabricks_alignment.job.yml`): Defines the Databricks workflow with a GPU cluster
4. **Notebook** (`modules/disease_biology/gwas/gwas_v1/notebooks/02_parabricks_germline.py`): Executes the Parabricks germline pipeline

### Workflow Pipeline

```
FASTQ R1 + R2 → Parabricks fq2bam (alignment) → BAM
                 Parabricks germline (variant calling) → VCF
```

### MLflow Tracking

- Creates an MLflow run with input parameters (FASTQ paths, reference genome)
- Tags the run with `feature=gwas_alignment` and `origin=genesis_workbench`
- Updates `job_status` tag: `started` → `alignment_complete` or `failed`

### Key Files

- `modules/core/app/views/disease_biology.py` — UI (Variant Calling tab)
- `modules/core/app/utils/disease_biology.py` — `start_parabricks_alignment()`
- `modules/disease_biology/gwas/gwas_v1/resources/parabricks_alignment.job.yml` — Job definition
- `modules/disease_biology/gwas/gwas_v1/notebooks/02_parabricks_germline.py` — Execution notebook
