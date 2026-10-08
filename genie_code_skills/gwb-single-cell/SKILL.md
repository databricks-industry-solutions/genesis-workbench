---
name: gwb-single-cell
description: Annotate or embed single cells in a Databricks notebook using Genesis Workbench single-cell foundation models (TEDDY, scGPT, SCimilarity) and the cell Vector Search indices — cell-type annotation by reference KNN, cell embeddings, and in-silico gene perturbation. Use for scRNA-seq / AnnData life-sciences workflows. Triggers on "annotate my cells / what cell types are these", "embed single cells", "scRNA-seq", "AnnData", "TEDDY / scGPT / SCimilarity", "predict a gene knockout / overexpression".
---

# Single-cell workflows in Genesis Workbench

Three single-cell foundation models, all consumed as serving endpoints (see **gwb-serving-endpoints** for
the raw contracts) and paired with cell Vector Search indices (see **gwb-vector-search**):

| Model | Endpoint slug | Embedding dim | Cell VS index | Best for |
|---|---|---|---|---|
| **TEDDY** | `teddy` | 1024 (400M) | `teddy_cell_index` (~2M ref cells) | cell-type **annotation** by KNN |
| **SCimilarity** | `scimilarity_get_embedding` | 128 | `scimilarity_cell_index` (~23M ref cells) | cell search / annotation |
| **scGPT** | `scgpt` / `scgpt_perturbation` | — | — | embeddings + in-silico **perturbation** |

## Annotate cells by reference KNN (TEDDY)
Encode each query cell with the TEDDY endpoint, query `teddy_cell_index`, and majority-vote the neighbors'
`cell_type`. The endpoint takes an AnnData-shaped payload (dense expression + obs/var as `orient="split"`
JSON; `adata_var` **must** carry gene names in its index).
```python
import json, numpy as np, pandas as pd, scanpy as sc
from databricks.sdk import WorkspaceClient
from collections import Counter
w = WorkspaceClient()

adata = sc.read_h5ad("/Volumes/.../my_cells.h5ad")          # your query cells
X = adata.X
Xd = X.toarray() if hasattr(X, "toarray") else np.asarray(X)   # dense cells × genes
var = adata.var.copy(); var.index = adata.var_names             # gene names in the index

def teddy_embed(expr_rows, obs_df, var_df):
    payload = [{
        "adata_sparsematrix": np.asarray(expr_rows, dtype=float).tolist(),
        "adata_obs": obs_df.to_json(orient="split"),
        "adata_var": var_df.to_json(orient="split"),
    }]
    preds = w.serving_endpoints.query(name=TEDDY_EP, inputs=payload).predictions
    return [np.asarray(p["embedding"], dtype=float) for p in preds]

TEDDY_EP = next(e.name for e in w.serving_endpoints.list() if "teddy" in e.name)
embs = teddy_embed(Xd, adata.obs, var)                          # one 1024-d vector per cell

labels = []
for emb in embs:
    r = w.vector_search_indexes.query_index(
        index_name="main.genesis_workbench.teddy_cell_index",
        columns=["cell_type", "tissue", "disease"],
        query_vector=emb.tolist(), num_results=25)
    neigh = [row[0] for row in (r.result.data_array or []) if row and row[0]]
    labels.append(Counter(neigh).most_common(1)[0][0] if neigh else "unknown")
adata.obs["predicted_cell_type"] = labels
display(adata.obs[["predicted_cell_type"]].value_counts())
```
SCimilarity is the same shape with its own payload (`celltype_sample` as `orient="split"` JSON with a
`celltype_subsample` column) → 128-d embedding → `scimilarity_cell_index`.

## In-silico perturbation (scGPT)
```python
res = w.serving_endpoints.query(
    name=next(e.name for e in w.serving_endpoints.list() if "scgpt_perturbation" in e.name),
    dataframe_split={"columns": ["expression", "gene_names", "genes_to_perturb", "perturbation_type"],
                     "data": [[expr_list, json.dumps(gene_names), "TP53", "knockout"]]}).predictions
# → {gene_name, original_expression, predicted_expression, delta, abs_delta}; sort by abs_delta for top responders
```

## Notes
- **Gene vocabulary matters.** Match `adata_var` gene identifiers to what the model expects (HGNC symbols
  / Ensembl IDs). TEDDY/scGPT rank genes by expression internally; a mismatched vocab degrades results.
- **Batch size:** send cells in chunks (hundreds–few-thousand per call); for a very large AnnData, embed at
  scale with Ray (**gwb-ray-batch-inference**) and KNN-query the index in bulk.
- **Honest scope:** annotation is reference KNN — only as good/complete as the reference (TEDDY ~2M,
  SCimilarity ~23M cells); novel/rare states may have no good neighbor. Perturbation is a model prediction,
  not a measured readout.
- Warm the endpoints + VS endpoints first (GWB reaps idle ones).
