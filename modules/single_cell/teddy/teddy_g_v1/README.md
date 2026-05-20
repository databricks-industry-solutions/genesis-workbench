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
| 1 | `register_teddy_task` | T4 GPU | `01_register_teddy.py` | Downloads HF snapshot, wraps encoder in MLflow PyFunc, registers in UC. |
| 2 | `import_teddy_model_task` | serverless | `02_import_model_gwb.py` | Imports model into GWB, deploys `gwb_teddy_endpoint` (**GPU_MEDIUM / A10** — 400M won't fit T4). |
| 3 | `extract_gene_mapping_task` | serverless | `06_extract_gene_mapping.py` | Pulls HGNC→ENSG mapping from CELLxGENE Census var; writes JSON to Volume. |
| 4 | `reembed_reference_task` | **8× A10 multinode** | `03_reembed_reference.py` | **Heavy.** Pulls ~2M Census cells, embeds with bf16+batch=48 via `mapInPandas`, writes `teddy_cells` Delta. |
| 5 | `create_vs_index_task` | serverless | `04_create_teddy_vs_index.py` | Creates `gwb_teddy_vs_endpoint` + `teddy_cell_index` (Delta Sync) — or syncs existing. |

Tasks 1, 3 run in parallel; 2, 4 fan out after 1; 5 depends on 4.

After the DAG succeeds, the **TEDDY Annotation** workflow under the GWB app's UMAP tab is fully live.

### Reembed cluster shape (task #4)

The 2 M-cell reference embed runs on a **multi-node** GPU cluster, not the single-node A10 used in earlier revisions:

- **8 workers × 1× A10 (g5.16xlarge or per-cloud equivalent) + A10 driver.**
- **On-demand for all 8 workers + driver**, no spot fallback (the bundle declares `availability: ON_DEMAND` per cloud target). A10 spot reclamation rate at multi-hour run length is too high.
- `spark.task.resource.gpu.amount=1` pins exactly one Spark task per GPU — no CUDA-context contention.
- Inside the UDF: each Spark worker loads TEDDY-G 400M once (per process), opens its own CELLxGENE Census handle, fetches X for its partition via `cellxgene_census.get_anndata(obs_coords=...)`, embeds at batch=48 with bf16 autocast, yields embeddings. GPU-side `torch.topk` over the 60,530-gene matrix (CPU topk was a bottleneck).
- **Partitioning:** `repartitionByRange("soma_joinid", 40)` — 40 partitions of ~50 k cells, sorted by Census joinid so each partition reads a contiguous joinid range from TileDB-SOMA on S3 (sequential reads, not random point reads). This is the critical perf knob — see the CHANGELOG for the 256-partition bug it fixed.
- Wall-clock on the validated 8-A10 build: **~3 h 15 min for 2 M cells** (the full DAG end-to-end is ~5 h including endpoint deploy + VS index initial sync).

### Cost transparency

Defaults:
- 8 × A10 g5.16xlarge worker nodes + 1 driver, on-demand, for ~3-4 hours.
- One-time cost per workspace (the post-deploy idempotency check makes re-deploys a no-op when the reference is already built — see below).

Override via:
- `--var=teddy_reembed_target_n_cells=500000` for a quick install (~30-45 min)
- `--var=teddy_reembed_per_stratum_cap=10000` for a more balanced (but smaller) sample
- `--var=teddy_reembed_census_version=<lts-tag>` to pin a different Census release
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
