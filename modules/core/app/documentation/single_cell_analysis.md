# Single Cell Analysis

## Introduction

Single Cell Analysis provides end-to-end processing of single-cell RNA sequencing (scRNA-seq) data. Two processing backends are available: **Scanpy** (CPU-based, standard) and **RAPIDS-SingleCell** (GPU-accelerated, for large datasets). Both produce the same outputs: quality-filtered, normalized, and clustered cell populations with marker gene identification.

## What It Achieves

- Quality control filtering (minimum genes/cells, mitochondrial content)
- Gene name harmonization across species references
- Normalization and highly variable gene selection
- Dimensionality reduction (PCA → UMAP)
- Cell clustering with configurable resolution
- Marker gene identification per cluster
- Interactive results viewer with UMAP visualization and marker dotplots

## How to Use

### Processing

1. Navigate to **Single Cell Studies > Raw Single Cell Processing** tab
2. Select processing mode: **Scanpy** (CPU) or **RAPIDS-SingleCell** (GPU)
3. Enter the h5ad file path on a UC Volume
4. Configure gene name mapping (column name or species reference)
5. Set filtering parameters:
   - Minimum genes per cell
   - Minimum cells per gene
   - Mitochondrial content cutoff
6. Set normalization parameters:
   - Target sum for normalization
   - Number of highly variable genes
7. Set dimensionality reduction parameters:
   - Number of principal components
   - Cluster resolution
8. Configure MLflow tracking
9. Click **Start Processing**

### Viewing Results

1. Navigate to **Single Cell Studies > Results Viewer** tab
2. Filter runs by experiment, processing mode, or time period
3. Select a completed run
4. Explore:
   - **UMAP plot**: Color by cluster, marker gene expression, or QC metrics
   - **Marker dotplot**: Top markers per cluster
   - **Dataset summary**: Cell/gene counts, clusters identified
   - **Export**: Download results as CSV

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| h5ad file | AnnData format single-cell data | `/Volumes/.../my_data.h5ad` |
| Processing mode | Scanpy (CPU) or RAPIDS (GPU) | Scanpy |
| Min genes/cell | QC filter | 200 |
| Min cells/gene | QC filter | 3 |
| Mito cutoff | Max mitochondrial % | 20 |
| HVG count | Highly variable genes | 2000 |
| PCs | Principal components | 50 |
| Resolution | Clustering resolution | 0.5 |

### Outputs

- **Processed AnnData**: Filtered, normalized data with UMAP embeddings and cluster labels
- **Marker genes**: Top differentially expressed genes per cluster
- **MLflow artifacts**: Full results stored for reproducibility

## How It's Implemented

### Processing Pipeline

```
h5ad input
  ↓
QC filtering (min genes, min cells, mito %)
  ↓
Gene name harmonization
  ↓
Normalization + log transform
  ↓
Highly variable gene selection
  ↓
PCA (dimensionality reduction)
  ↓
Neighbor graph construction
  ↓
UMAP embedding
  ↓
Leiden clustering
  ↓
Marker gene identification (rank_genes_groups)
  ↓
Save to MLflow
```

### Supporting Models

- **SCimilarity**: Cell type annotation using ~23M reference profiles (Genentech)
  - Three endpoints: GeneOrder, GetEmbedding, SearchNearest
  - Finds similar cells across disease datasets (Adams et al. lung, etc.)
- **scGPT**: Transformer language model for cell annotation and embedding
  - Pre-trained on 33M cells (whole-human model)
  - Supports zero-shot cell type prediction

### Key Files

- `modules/core/app/views/single_cell.py` — UI (Processing + Results tabs)
- `modules/single_cell/scanpy/scanpy_v0.0.1/` — Scanpy processing notebooks
- `modules/single_cell/rapidssinglecell/rapidssinglecell_v0.0.1/` — RAPIDS processing notebooks
- `modules/single_cell/scimilarity/` — SCimilarity model registration
- `modules/single_cell/scgpt/` — scGPT model registration
