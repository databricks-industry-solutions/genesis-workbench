# TEDDY joint cell-type + disease annotation (Single Cell)

**Module:** single_cell
**Status:** GA
**Added:** 2026-05-20

## What it does

Annotates the clusters from a Scanpy run with both **cell type** and **disease** labels using Merck's TEDDY-G 400M foundation model (Apache 2.0, encoder-only). For each cluster, the app embeds representative cells through the TEDDY endpoint, queries a 2-million-cell CELLxGENE Census reference index for k-nearest neighbors, and votes — with optional inverse-frequency bias correction — on the cell type and disease labels. The result is shown side-by-side with SCimilarity's prediction on the same UMAP tab; both run from a single click.

## How to use (UI walkthrough)

1. Open the GWB app → **Single Cell** → run a **Scanpy** analysis (or open a past run from *Search Past Runs*).
2. Open the **UMAP** tab on the result dialog.
3. Under **Annotate clusters**, the checkboxes default to both **SCimilarity** and **TEDDY** = ON, plus **Bias-correct vote tally (IDF)** = ON for TEDDY.
4. Click **Annotate**. The app embeds each cluster's representative cells via the `gwb_teddy_endpoint`, queries `teddy_cell_index`, weighted-votes the labels, and logs the annotation table to the existing MLflow run.
5. Two per-cluster tables appear side-by-side: SCimilarity's predicted cell type, TEDDY's joint (cell type + disease) labels, plus confidence + top-3 similar references.
6. Wall-clock: ~30-60 s for a 10-cluster run; the model is endpoint-served so cluster cold-start does not apply.

## Inputs

- A completed Scanpy run logged under `scanpy_genesis_workbench`. The annotator reads `markers_flat.parquet` (per-cluster top marker genes, ~68 dense genes, mostly non-zero in the cells they were picked from).
- **Why markers_flat and not the HVG matrix?** The wider HVG matrix (~2000 genes) is ~96 % zero per cell. With TEDDY's rank-value encoding, the zero-valued tail of each cell's topk sequence is filled in gene-index order (PyTorch tie-break), producing an identical "filler tail" across cells. The model's uniform attention mask attends equally to that identical 96 % filler and the 4 % real signal — embeddings collapse and every cluster lands on the same KNN region. markers_flat's higher signal density matches what TEDDY was pretrained on. (Once the long-term attention-mask fix lands in the wrapper + the reference is rebuilt with the same masking, the HVG path can be reinstated for richer context.)
- Required gene-name convention: HGNC symbols. The app translates to ENSG IDs via `gene_mapping.json` (60,530 entries) on the UC Volume before calling the endpoint. **Reads via the SDK Files API**, not POSIX `open()` — Databricks Apps' sandbox has no FUSE mount for `/Volumes`, so `open()` would silently fail and leave the mapping empty, sending raw HGNC symbols to the endpoint which then maps everything to `<unk>` and returns identical embeddings.

## Outputs

For each annotated run, logged to the run's MLflow artifacts directory under `cell_type_annotation/`:

- `teddy_cluster_annotation.json` — `[{cluster_id, predicted_cell_type, confidence, top3_similar, predicted_disease, disease_confidence, top3_diseases}, ...]`
- The result dialog renders these alongside SCimilarity's `cluster_annotation.json` in two columns.
- No new tables; the annotation is per-MLflow-run, not stored in a shared Delta.

## Underlying models / endpoints

- **Model:** TEDDY-G 400M, registered at `${catalog}.${schema}.teddy` (Unity Catalog) — Apache 2.0.
- **Serving endpoint:** name looked up from the `model_deployments` table at call time via `genesis_workbench.models.get_endpoint_name_for_uc_model("teddy")`. (Replaces the prior pattern of constructing `gwb_{DEV_USER_PREFIX}_teddy_endpoint` client-side — that approach silently 404'd whenever the env var didn't make it through the deploy chain.) The endpoint runs on GPU_MEDIUM (A10), single replica, embeddings only (no classification head — public TEDDY-G has `n_cls=0`).
- **Reference table:** `${catalog}.${schema}.teddy_cells` — 2 M cells stratified across 197 (tissue_general, disease) strata from CELLxGENE Census 2024-07-01 LTS. Includes healthy + disease cells.
- **Vector Search index:** `${catalog}.${schema}.teddy_cell_index` on `gwb_teddy_vs_endpoint`. Delta Sync, TRIGGERED, embedding dim 1024.
- **Gene mapping artifact:** `/Volumes/${catalog}/${schema}/teddy/gene_mapping.json` — HGNC → ENSG translation (60,530 entries from Census var). Read via SDK Files API in the app (see Inputs).

Detailed model + deploy reference: [`modules/single_cell/teddy/teddy_g_v1/README.md`](../../../single_cell/teddy/teddy_g_v1/README.md).

## Limitations and known issues

- **No rare-disease retrieval guarantee.** Diseases with < 1000 cells in the 2 M reference are voted but with low confidence; the UI shows confidence in the result table.
- **Human only.** Census slice is `homo_sapiens`. Mouse runs fall back to SCimilarity-only annotation (TEDDY checkbox is greyed out when the Scanpy run's species ≠ `hsapiens`).
- **markers_flat is the active input today.** The HVG-matrix input is currently disabled in the app — it triggers an attention-pattern collapse with the current wrapper. A long-term fix (per-cell attention mask that ignores zero-expression positions on both query and reference sides, plus a reference rebuild) will re-enable HVG for richer context.
- **400M-only.** The earlier 70M variant collapsed immune cell types (NK ↔ plasma cosine 0.98 in inspection); the TEDDY paper validates zero-shot retrieval only on 400M. The bundle variable `teddy_model_size` defaults to `400M` and should not be lowered for production.
- **Endpoint cold-start.** First annotation after idle (~30 min) takes 30-60 s extra for endpoint scale-up.
