# TEDDY (Merck) — Single-Cell Foundation Model

**TEDDY** (Transformer for Enabling Drug Discovery) is a family of foundation models for single-cell biology from Merck, trained on 116M cells (human + mouse). Licensed under **Apache 2.0**.

Source: <https://huggingface.co/Merck/TEDDY>

## How GWB uses TEDDY

The publicly released TEDDY-G checkpoints are **encoder-only** (`n_cls=0`, no classification heads). GWB wraps the encoder in a serving endpoint that returns a per-cell embedding, then performs annotation via:

1. **TEDDY embedding endpoint** (`gwb_teddy_endpoint`) — per-cell embedding (512-d for 70M / 768-d for 160M / 1024-d for 400M).
2. **TEDDY reference Delta table** (`teddy_cells`) — ~2M cells from CELLxGENE Census, each labeled with `cell_type`, `disease`, `tissue`, embedded with the same TEDDY variant.
3. **TEDDY Vector Search index** (`teddy_cell_index`) — Delta-Sync index over `teddy_cells`.
4. **Annotation pipeline** (in the GWB app) — embed user cells → KNN against `teddy_cell_index` → majority-vote on both `cell_type` and `disease` per cluster.

Both labels come from one neighbor lookup — the joint annotation view always shows both, no model selector.

## Variants

| Variant | Parameters | Default? |
|---|---|---|
| TEDDY-G 70M | 70M | ✓ (smallest, fastest serving) |
| TEDDY-G 160M | 160M |  |
| TEDDY-G 400M | 400M |  |

Select via the `teddy_model_size` bundle variable. Only TEDDY-G is released; TEDDY-X is not currently public.

## Deploy

From the GWB repo root:

```bash
./deploy.sh single_cell <aws|azure|gcp> --only-submodule teddy/teddy_g_v1
```

Or from this directory directly:

```bash
./deploy.sh aws --var="core_catalog_name=genesis_workbench,core_schema_name=dev_yyang_genesis_workbench"
```

To pin a specific HuggingFace revision (recommended for reproducibility):

```bash
./deploy.sh aws --var="teddy_hf_revision=<commit_sha>,..."
```

## What the deploy does

1. Creates a managed Volume `${core_catalog}.${core_schema}.teddy` for caching weights.
2. Runs `notebooks/01_register_teddy.py` on a T4 GPU cluster:
   - Downloads the TEDDY HF snapshot (weights + source code) into the Volume.
   - Wraps the encoder in an MLflow PyFunc that returns per-cell embeddings.
   - Logs the model to UC: `${catalog}.${schema}.teddy`.
3. Runs `notebooks/02_import_model_gwb.py` on serverless to import into GWB and deploy a `gwb_teddy_endpoint` serving endpoint (GPU_SMALL).

## One-time post-deploy: build the annotation reference

The deploy gets you the embedding endpoint. To make the annotation tab functional, run two more notebooks one time per workspace (these are heavy and intentionally not part of the auto-deploy job):

```bash
# 1. Pull ~2M cells from CELLxGENE Census, embed with TEDDY, write teddy_cells Delta.
#    ~2-6 hours on a T4 GPU cluster. Idempotent — safe to re-run if interrupted.
databricks bundle run --target prod_aws \
  reembed_teddy_reference \
  --var="..." \
  --no-wait

# 2. Create gwb_teddy_vs_endpoint + teddy_cell_index over the Delta table.
#    Index sync takes ~30-60 min after rows are written.
databricks bundle run --target prod_aws \
  create_teddy_vs_index \
  --var="..." \
  --no-wait
```

Or run them manually as notebooks (`notebooks/03_reembed_reference.py`, `notebooks/04_create_teddy_vs_index.py`).

After all four steps the **TEDDY Annotation** tab in the Single Cell page is fully live.

The `gwb_teddy_vs_endpoint` and `teddy_cell_index` are **preserved across GWB destroys** per the project's destroy-preservation policy (they're expensive to rebuild — same rule as the SCimilarity VS resources).

## Notes

- All pip deps in `requirements.txt` are exact-pinned per the GWB project rule.
- The HF revision is variable-controlled — override `teddy_hf_revision` to pin a specific commit SHA.
- Apache 2.0 license — satisfies the project rule that only permissive-licensed models (Apache/MIT/BSD) ship in GWB.
