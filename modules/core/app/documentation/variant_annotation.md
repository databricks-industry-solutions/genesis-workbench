# Variant Annotation

## Introduction

Variant Annotation enriches genetic variants with clinical significance data from ClinVar, enabling identification of pathogenic mutations. It cross-references ingested VCF data against the ClinVar database and filters for clinically relevant variants in specified gene regions.

## What It Achieves

- Cross-references variant data against the ClinVar database (downloaded during module setup)
- Filters variants by user-specified gene regions
- Identifies pathogenic and likely-pathogenic variants with clinical significance annotations
- Produces annotated results with disease associations, zygosity, and clinical classifications
- Includes an interactive Lakeview dashboard for exploring results

## How to Use

1. Navigate to **Disease Biology > Variant Annotation** tab
2. Select a completed VCF Ingestion run (provides the variants table) or enter the table name
3. Specify gene regions of interest (e.g., `BRCA1,BRCA2,TP53`)
4. Optionally provide a pathogenic VCF path for additional annotation
5. Set MLflow experiment and run names
6. Click **Start Variant Annotation**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Variants table | Delta table from VCF Ingestion | `catalog.schema.my_variants` |
| Gene regions | Comma-separated gene symbols | `BRCA1,BRCA2,TP53` |
| Pathogenic VCF path | Optional additional pathogenic variants | `/Volumes/.../brca_pathogenic_corrected.vcf` |

### Outputs

- **Annotated variants table**: `{catalog}.{schema}.variant_annotation_pathogenic`
- Fields: gene, chromosome, position, ref, alt, zygosity, clinical significance, disease name
- **Lakeview dashboard**: Interactive visualization of pathogenic variants

## How It's Implemented

### Architecture

1. **UI** (`modules/core/app/views/disease_biology.py`): Variant Annotation tab with VCF Ingestion run picker
2. **Backend** (`modules/core/app/utils/disease_biology.py`): `start_variant_annotation()` creates MLflow run and triggers job
3. **Job** (`modules/disease_biology/variant_annotation/variant_annotation_v1/resources/variant_annotation_workflow.job.yml`): Multi-task workflow

### Workflow Pipeline

```
filter_and_annotate → save_results → mark_success/mark_failure
```

1. **Filter and Annotate** (`02_filter_and_annotate.py`): Joins variant data with ClinVar, filters by gene region and pathogenicity
2. **Save Results** (`03_save_results.py`): Persists annotated results and updates MLflow

### Setup (One-Time)

During module initialization:
- **ClinVar download** (`00_download_clinvar.py`): Downloads ClinVar GRCh38 VCF from NCBI, loads into `clinvar_variants` Delta table
- **Dashboard setup**: Configures Lakeview dashboard with correct catalog/schema references
- **Demo data**: Copies sample pathogenic VCF for testing

### MLflow Tracking

- Logs parameters: variants table, gene regions
- Tags: `feature=variant_annotation`, `job_status=annotation_complete` or `failed`

### Key Files

- `modules/core/app/views/disease_biology.py` — UI (Variant Annotation tab)
- `modules/core/app/utils/disease_biology.py` — `start_variant_annotation()`, `pull_annotation_results()`
- `modules/disease_biology/variant_annotation/variant_annotation_v1/resources/variant_annotation_workflow.job.yml` — Job definition
- `modules/disease_biology/variant_annotation/variant_annotation_v1/notebooks/00_download_clinvar.py` — ClinVar setup
- `modules/disease_biology/variant_annotation/variant_annotation_v1/notebooks/02_filter_and_annotate.py` — Annotation logic
