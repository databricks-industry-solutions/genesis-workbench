# VCF Ingestion

## Introduction

VCF Ingestion converts Variant Call Format (VCF) files into Delta tables for efficient querying and downstream analysis. It uses the Glow library to parse VCF files and persist the structured variant data in Unity Catalog.

## What It Achieves

- Reads VCF files (including compressed `.vcf.gz`) using Glow's VCF reader
- Writes variant data as a Delta table with full schema (contig, position, ref, alt, genotypes, INFO fields)
- Enables SQL-based querying of variant data for annotation and analysis workflows

## How to Use

1. Navigate to **Disease Biology > VCF Ingestion** tab
2. Enter the VCF file path on a UC Volume
3. Specify the output Delta table name
4. Set MLflow experiment and run names
5. Click **Start VCF Ingestion**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| VCF file path | Path to VCF/VCF.GZ file | `/Volumes/catalog/schema/vcf_ingestion_data/my_variants.vcf.gz` |
| Output table name | Name for the Delta table | `my_variants` |

### Outputs

- **Delta table**: `{catalog}.{schema}.{output_table_name}` containing all variant records
- Variant count logged to MLflow

## How It's Implemented

### Architecture

1. **UI** (`modules/core/app/views/disease_biology.py`): VCF Ingestion tab
2. **Backend** (`modules/core/app/utils/disease_biology.py`): `start_vcf_ingestion()` creates MLflow run and triggers job
3. **Job** (`modules/disease_biology/vcf_ingestion/vcf_ingestion_v1/resources/vcf_ingestion_workflow.job.yml`): Single-task workflow with Glow JAR library

### Workflow Pipeline

```
vcf_to_delta → mark_success/mark_failure
```

1. **VCF to Delta** (`01_vcf_to_delta.py`): Reads VCF with `spark.read.format("vcf")`, counts variants, writes to Delta table

### MLflow Tracking

- Logs parameters: VCF path, output table name, variant count
- Tags: `feature=vcf_ingestion`, `job_status=ingestion_complete` or `failed`
- Output table name stored as tag for downstream reference

### Key Files

- `modules/core/app/views/disease_biology.py` — UI (VCF Ingestion tab)
- `modules/core/app/utils/disease_biology.py` — `start_vcf_ingestion()`
- `modules/disease_biology/vcf_ingestion/vcf_ingestion_v1/resources/vcf_ingestion_workflow.job.yml` — Job definition
- `modules/disease_biology/vcf_ingestion/vcf_ingestion_v1/notebooks/01_vcf_to_delta.py` — Execution notebook
