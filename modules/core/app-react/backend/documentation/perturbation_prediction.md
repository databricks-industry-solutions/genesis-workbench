# Gene Perturbation Prediction

## Introduction

Gene Perturbation Prediction uses scGPT's transformer architecture to predict how knocking out or overexpressing a gene affects the expression profile of a cell population. This enables in-silico perturbation experiments without wet-lab work.

## What It Achieves

- Predicts post-perturbation expression profiles for gene knockouts or overexpression
- Identifies the genes most affected by the perturbation (largest expression deltas)
- Provides visual comparison of original vs predicted expression

## How to Use

1. Navigate to **Single Cell Studies > Gene Perturbation Prediction** tab
2. Select a completed processing run
3. Click **Load Run**
4. Select a cluster and perturbation type (Knockout or Overexpress)
5. Choose gene(s) to perturb from the ranked list or type additional gene names
6. Click **Predict Perturbation Effect**
7. Review results:
   - **Top Affected Genes**: Horizontal bar chart of the 20 genes with largest expression change
   - **Original vs Predicted**: Scatter plot comparing control and perturbed expression (deviation from diagonal indicates perturbation effect)
   - **Summary metrics**: Total genes analyzed, significantly affected count, max delta

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Processing run | A completed Scanpy or RAPIDS run | Select from dropdown |
| Cluster | Cell cluster to perturb | 0, 1, 2, ... |
| Perturbation type | Knockout (set to 0) or Overexpress (set to max) | Knockout |
| Gene(s) to perturb | One or more gene names | TP53, BRCA1 |

### Outputs

- **Top affected genes chart**: Bar chart ranked by absolute expression delta
- **Original vs predicted scatter**: Points off the diagonal indicate perturbation effects
- **Summary**: Count of significantly affected genes and maximum delta
- **Full results table**: Per-gene original expression, predicted expression, and delta

## How It's Implemented

1. Extracts the mean expression profile for the selected cluster from the markers data
2. Sends the expression vector, gene names, perturbation gene(s), and perturbation type to the scGPT perturbation endpoint
3. The scGPT model:
   - Maps genes to vocabulary token IDs
   - Creates control (original) and perturbed expression vectors
   - Computes gene + value embeddings for both vectors
   - Runs both through the transformer encoder and expression decoder
   - Returns per-gene predicted expression and delta (perturbed - control)
4. Results are sorted by absolute delta and displayed as charts and a table

### Key Files

- `modules/core/app/views/single_cell_workflows/perturbation.py` — UI and endpoint call
- `modules/single_cell/scgpt/scgpt_v0.2.4/notebooks/03_register_scgpt_perturbation.py` — Model wrapper and registration

### Dependencies

- scGPT Perturbation model serving endpoint (GPU)
