# Cell Type Annotation

## Introduction

Cell Type Annotation automatically identifies the cell types present in each cluster of a processed single-cell dataset using SCimilarity, a deep learning model trained on ~23 million reference cell profiles from Genentech.

## What It Achieves

- Predicts a cell type label for each cluster in the dataset
- Maps predictions onto the UMAP embedding for visual validation
- Leverages a large reference database spanning healthy and diseased tissues

## How to Use

1. Navigate to **Single Cell Studies > Cell Type Annotation** tab
2. Select a completed processing run (Scanpy or RAPIDS)
3. Click **Annotate**
4. Review the UMAP plot colored by predicted cell type
5. Inspect per-cluster predictions in the results table

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Processing run | A completed Scanpy or RAPIDS run | Select from dropdown |

### Outputs

- **Annotated UMAP**: Scatter plot colored by predicted cell type
- **Predictions table**: Cluster-to-cell-type mapping with confidence

## How It's Implemented

1. Loads the marker gene expression matrix from the selected MLflow run
2. Calls the SCimilarity annotation pipeline (`annotate_clusters`) which:
   - Aligns gene names to the SCimilarity gene order
   - Computes cell embeddings via the SCimilarity model endpoint
   - Matches each cluster's embedding against the 23M-cell reference
   - Returns predicted cell type labels per cluster
3. Maps predictions back onto the UMAP coordinates for visualization

### Key Files

- `modules/core/app/views/single_cell_workflows/cell_type_annotation.py` — UI
- `modules/core/app/utils/scimilarity_tools.py` — `annotate_clusters()`
- `modules/single_cell/scimilarity/` — SCimilarity model registration

### Dependencies

- SCimilarity model serving endpoints (GeneOrder, GetEmbedding, SearchNearest)
