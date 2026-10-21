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

| Variant | Parameters | Embedding dim | Default? |
|---|---|---|---|
| TEDDY-G 70M | 70M | 512 |  ⚠ inadequate — see below |
| TEDDY-G 160M | 160M | 768 | not benchmarked for zero-shot retrieval |
| TEDDY-G 400M | 400M | 1024 | ✓ **default** |

Select via the `teddy_model_size` bundle variable. Only TEDDY-G is released; TEDDY-X is not currently public.

**Why 400M is the default and not 70M:** the public TEDDY-G 70M encoder collapses immune cell types — in our internal inspection on a `scanpy_20260507_1950` cluster known to be NK by marker genes, the 70M encoder placed it at cosine 0.98 to plasma cells, making KNN retrieval indistinguishable between the two. The TEDDY paper benchmarks zero-shot retrieval **only** on the 400M variant. Variant size matters for this workflow; lowering to 70M for cost savings will regress annotation quality.

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

`deploy.sh` runs a single multi-task DAG (`register_teddy` bundle job). Tasks:

| # | Task | Cluster | Notebook | Description |
|---|---|---|---|---|
| 1 | `register_teddy_task` | GPU (serverless) | `01_register_teddy.py` | Downloads HF snapshot, wraps encoder in MLflow PyFunc, registers in UC. |
| 2 | `import_teddy_model_task` | serverless | `02_import_model_gwb.py` | Imports model into GWB, deploys `gwb_teddy_endpoint` (**GPU_MEDIUM / A10**). |
| 3 | `extract_gene_mapping_task` | serverless | `06_extract_gene_mapping.py` | Pulls HGNC→ENSG mapping from CELLxGENE Census var; writes JSON to Volume. |
| 4 | `stage_reference_task` | GPU (serverless) | `03a_stage_reference.py` | **Network-bound.** Fetches ~2M Census cells and precomputes the TEDDY top-k gene **token IDs** (GPU `torch.topk`), writing resumable chunked parquet to the cache Volume. |
| 5 | `reembed_reference_task` | **Ray on serverless GPU** | `03b_embed_reference_ray.py` | **GPU-bound.** Reads the staged tokens (no Census), runs the TEDDY forward at bf16+batch=48 across Ray A10 workers, writes `teddy_cells` Delta. |
| 6 | `create_vs_index_task` | serverless | `04_create_teddy_vs_index.py` | Creates `gwb_teddy_vs_endpoint` + `teddy_cell_index` (Delta Sync) — or syncs existing. |

Tasks 2, 3, 4 fan out after 1; 5 depends on 4; 6 depends on 5.

After the DAG succeeds, the **TEDDY Annotation** workflow under the GWB app's UMAP tab is fully live.

### Reembed configuration (tasks #4–#5)

The 2 M-cell reference embed runs on **serverless GPU via Ray**, split into a network-bound STAGE (03a) and a GPU-bound EMBED (03b) so the A10s never idle on Census I/O:

- **STAGE (`03a_stage_reference.py`)** — opens CELLxGENE Census, builds a deterministic stratified obs sample, and across Ray A10 workers fetches X via `cellxgene_census.get_anndata(obs_coords=...)` and precomputes the TEDDY input (top-k gene **token IDs** via GPU `torch.topk` over the 60,530-gene matrix). Writes one resumable `chunk_<ts>/` parquet sub-dir per run — an interrupted stage never loses progress. Batches are sorted by `soma_joinid` so each Ray batch reads a contiguous joinid range from TileDB-SOMA on S3 (sequential reads, not random point reads — the critical perf knob).
- **EMBED (`03b_embed_reference_ray.py`)** — Ray reads the LOCAL staged tokens (no Census), runs the pure TEDDY forward at bf16 + batch=48 across the A10 workers, mean-pools, and promotes the parquet output into the managed `teddy_cells` Delta via CTAS. With the network fetch gone this phase is GPU-bound → scales ~linearly with workers.
- **Ray workers × GPU_1xA10 serverless** — default **4** (`--var=num_gpu_workers=N`); serverless GPU bypasses the EC2 GPU vCPU quota (critical where GPU quota = 0).
- **Reading the chunked stage with Spark** needs `.option("recursiveFileLookup","true")` — the per-run `chunk_<ts>/` sub-dirs are not `key=value` partitions, so Spark's default parquet reader misses them (`UNABLE_TO_INFER_SCHEMA`). Ray's `read_parquet` recurses on its own.
- Wall-clock on the validated 4-worker serverless build: **stage ~30 min + a fast GPU embed for 2 M cells** — well under the old single-notebook ~5.5 h bottleneck (the full DAG end-to-end also includes endpoint deploy + VS index initial sync).

### Cost transparency

Defaults:
- 4 × serverless GPU (GPU_1xA10) Ray workers, for ~1-2 hours (stage ~30 min + embed).
- One-time cost per workspace (the post-deploy idempotency check makes re-deploys a no-op when the reference is already built — see below).

Override via:
- `--var=teddy_reembed_target_n_cells=500000` for a quick install (~30-45 min)
- `--var=teddy_reembed_per_stratum_cap=10000` for a more balanced (but smaller) sample
- `--var=teddy_reembed_census_version=<lts-tag>` to pin a different Census release
- `--var=num_gpu_workers=8` to scale up Ray workers for faster embedding (default 4)
- `--var=teddy_model_size=70M` if you accept the immune cell-type collapse (NOT recommended, see Variants section)

### Idempotency — re-deploys are no-ops when the reference is complete

Both heavy tasks pre-flight-check before doing work:

- **Notebook 03** (`reembed_reference_task`): reads `teddy_cells` row count + embedding dimension. If `rows ≥ 0.95 × target` AND `dim == expected_dim_for_variant`, it logs `Looks complete and dim matches — skipping rebuild.` and exits without touching the table.
- **Notebook 04** (`create_vs_index_task`): if `teddy_cells` is complete AND the index already exists at matching dim, it calls `dbutils.notebook.exit("skipped...")` immediately — no VS API calls at all.

The dim check is what makes variant switches safe: deploying with `teddy_model_size=400M` on a workspace that previously ran 70M sees the dim=512 stale table, logs `Embedding dim mismatch (have 512, want 1024 for 400M) — rebuilding.`, drops, and rebuilds. Notebook 04 mirrors this: if the existing index is dim=512 and the variant wants dim=1024, it drops the index and creates fresh (sync can't change index dim).

### Preserved across `databricks bundle destroy`

These artifacts are **procedurally created by the notebooks**, not declared as bundle resources, so `databricks bundle destroy` does NOT touch them:

- `gwb_teddy_vs_endpoint` (Vector Search endpoint)
- `${catalog}.${schema}.teddy_cell_index` (VS index)
- `${catalog}.${schema}.teddy_cells` (Delta reference, 2M rows)
- `/Volumes/${catalog}/${schema}/teddy/gene_mapping.json` (HGNC→ENSG mapping)

Rebuilding them is expensive (the A10-hours of #4 above). Preserving across destroy is by design — same policy as SCimilarity's VS resources. The destroy wizard (`./destroy.sh`) and any future cleanup tooling MUST NOT propose deleting these as a default step.

If you genuinely want to rebuild the reference: `DROP TABLE ${catalog}.${schema}.teddy_cells` manually, then re-run `databricks bundle run register_teddy`. The notebook's idempotency check sees the missing table and embeds fresh.

## Notes

- All pip deps in `requirements.txt` are exact-pinned per the GWB project rule.
- The HF revision is variable-controlled — override `teddy_hf_revision` to pin a specific commit SHA.
- Apache 2.0 license — satisfies the project rule that only permissive-licensed models (Apache/MIT/BSD) ship in GWB.
