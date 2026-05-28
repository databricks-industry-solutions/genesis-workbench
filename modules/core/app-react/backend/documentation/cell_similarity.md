# Cell Similarity Search

## Introduction

Cell Similarity Search finds cells in SCimilarity's 23 million-cell reference database that are most similar to cells in a selected cluster from your dataset. This helps identify matching cell populations across published studies, tissues, and disease conditions.

## What It Achieves

- Finds the most similar reference cells to a query cluster
- Shows the distribution of matching cell types and diseases across reference datasets
- Enables cross-study comparison to validate cluster identity

## How to Use

1. Navigate to **Single Cell Studies > Cell Similarity Search** tab
2. Select a completed processing run
3. Click **Load Run**
4. Select a cluster to query
5. Click **Search**
6. Review results:
   - **Neighbor Cell Types**: Bar chart of cell type distribution among similar cells
   - **Neighbor Disease Distribution**: Bar chart of disease labels among similar cells
   - **Study Sources**: Table of contributing datasets

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Processing run | A completed Scanpy or RAPIDS run | Select from dropdown |
| Cluster | Cluster ID to query | 0, 1, 2, ... |

### Outputs

- **Cell type distribution**: Bar chart showing which cell types the reference matches belong to
- **Disease distribution**: Bar chart of disease contexts from matched reference cells
- **Study sources**: Table listing which published datasets contributed matched cells

## How It's Implemented

1. Loads the marker gene expression matrix from the selected MLflow run
2. Retrieves the SCimilarity gene order from the GeneOrder endpoint
3. Aligns the cluster's expression profile to the reference gene order
4. Normalizes expression values (log-normalization)
5. Computes cell embeddings via the GetEmbedding endpoint
6. Searches for nearest neighbors in the reference database via the SearchNearest endpoint
7. Aggregates and displays the results by cell type, disease, and study

### Key Files

- `modules/core/app/views/single_cell_workflows/cell_similarity.py` — UI
- `modules/core/app/utils/scimilarity_tools.py` — `get_gene_order()`, `align_to_gene_order()`, `get_cell_embeddings()`, `search_nearest_cells()`
- `modules/single_cell/scimilarity/` — SCimilarity model registration

### Dependencies

- SCimilarity model serving endpoints (GeneOrder, GetEmbedding, SearchNearest)
