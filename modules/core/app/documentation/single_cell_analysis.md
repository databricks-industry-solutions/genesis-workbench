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

1. Navigate to **Single Cell Studies > Raw Single Cell Processing > View Analysis Results** tab
2. Filter runs by experiment, processing mode, or time period
3. Select a completed run and click **Load**
4. The dataset summary and download/MLflow buttons appear at the top. Below, explore results across tabs:
   - **UMAP**: Interactive scatter plot — color by cluster, marker gene expression, or QC metrics
   - **Marker Genes**: Dot plot of top marker gene expression by cluster (z-scored or raw)
   - **Differential Expression**: Compare two clusters using Mann-Whitney U test; volcano plot of log2 fold-change vs adjusted p-value with labeled significant genes
   - **Pathway Enrichment**: Select a cluster and gene set databases (GO, KEGG, Reactome); runs local Over-Representation Analysis (Fisher's exact test) against GMT files to find enriched biological pathways
   - **Trajectory**: UMAP colored by diffusion pseudotime and gene expression trends along the trajectory (requires pseudotime enabled during processing)
   - **QC & Outputs**: Link to the full MLflow run with all artifacts
   - **Raw Data**: Filterable data table with column selection

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

### Differential Expression Details

Compares gene expression between two selected clusters:
1. Runs a Mann-Whitney U test per gene across the two cell populations
2. Computes log2 fold-change and Benjamini-Hochberg adjusted p-values
3. Displays a volcano plot with significance thresholds (|log2FC| > 1, adj. p < 0.05)
4. Lists significant genes in a sortable table

### Pathway Enrichment Details

Identifies overrepresented biological pathways in a cluster's marker genes:
1. Retrieves the cluster's top marker genes (from MLflow artifact or top-50 by expression)
2. Loads GMT gene set files from the `scanpy_reference` volume (downloaded during deployment)
3. Runs Fisher's exact test for each gene set term against the dataset's gene background
4. Applies Benjamini-Hochberg correction and displays top enriched terms as a bar chart

Available databases: GO Biological Process, GO Molecular Function, GO Cellular Component, KEGG, Reactome.

### Supporting Models

- **SCimilarity**: Cell type annotation and similarity search using ~23M reference profiles (Genentech)
  - Three endpoints: GeneOrder, GetEmbedding, SearchNearest
  - Finds similar cells across disease datasets (Adams et al. lung, etc.)
- **scGPT**: Transformer language model for cell annotation, embedding, and perturbation prediction
  - Pre-trained on 33M cells (whole-human model)
  - Supports zero-shot cell type prediction and gene perturbation analysis

### Key Files

- `modules/core/app/views/single_cell_workflows/processing.py` — Processing + Results Viewer UI
- `modules/core/app/views/single_cell_workflows/cell_type_annotation.py` — Cell Type Annotation UI
- `modules/core/app/views/single_cell_workflows/cell_similarity.py` — Cell Similarity Search UI
- `modules/core/app/views/single_cell_workflows/perturbation.py` — Gene Perturbation Prediction UI
- `modules/single_cell/scanpy/scanpy_v0.0.1/` — Scanpy processing notebooks + gene set downloads
- `modules/single_cell/rapidssinglecell/rapidssinglecell_v0.0.1/` — RAPIDS processing notebooks
- `modules/single_cell/scimilarity/` — SCimilarity model registration
- `modules/single_cell/scgpt/` — scGPT model registration (embedding + perturbation)
