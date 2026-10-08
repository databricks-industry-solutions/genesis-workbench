# Databricks notebook source
# MAGIC %md
# MAGIC ### TEDDY reference embed — scalable Ray GPU forward from staged tokens (Phase 2 of 2)
# MAGIC
# MAGIC Reads the **local** precomputed top-k gene token IDs written by `03a_stage_reference` (NO Census
# MAGIC calls), runs the TEDDY forward on serverless-GPU Ray workers, mean-pools, and writes `teddy_cells`.
# MAGIC Because the network fetch is gone, this phase is **GPU-bound → scales ~linearly with
# MAGIC `num_gpu_workers`**. The embed math is identical to the former inline reembed: the stored tokens
# MAGIC are the top-k `gene_ids`; `gene_values` (linspace rank), the CLS prefix, and the attention mask
# MAGIC are reconstructed here deterministically from k.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt pyarrow==15.0.2 "ray[data]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "teddy", "Cache dir")
dbutils.widgets.text("teddy_model_size", "400M", "TEDDY-G variant")
dbutils.widgets.text("target_n_cells", "2000000", "Target reference cell count (for the idempotency check)")
dbutils.widgets.text("num_gpu_workers", "4", "Number of single-A10 serverless-GPU Ray workers")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
model_size = dbutils.widgets.get("teddy_model_size")
target_n_cells = int(dbutils.widgets.get("target_n_cells"))
num_gpu_workers = int(dbutils.widgets.get("num_gpu_workers"))

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"

SNAPSHOT_DIR = f"{cache_full_path}/snapshots/main"
MODEL_DIR = f"{SNAPSHOT_DIR}/teddy/models/teddy_g/{model_size}"
MODEL_SIZE = model_size
NUM_GPU_WORKERS = num_gpu_workers
BATCH_EMB = 48                 # per-forward GPU batch (A10 + bf16 + 400M ≈ 22-24 GB; drop to 40 on OOM)
RAY_BATCH = 10_000             # cells per map_batches call
TOKENS_STAGE_DIR = f"{cache_full_path}/reembed_tokens_stage"       # written by 03a
OUTPUT_STAGE_PARQUET = f"{cache_full_path}/reembed_output_stage_parquet"
CELLS_TABLE = f"{catalog}.{schema}.teddy_cells"
EXPECTED_DIM = {"70M": 512, "160M": 768, "400M": 1024}.get(model_size, 1024)

print(f"Reading staged tokens from {TOKENS_STAGE_DIR}")
print(f"Target table {CELLS_TABLE} (dim {EXPECTED_DIM}), {num_gpu_workers} A10 Ray workers")

# COMMAND ----------

# DBTITLE 1,Idempotency check (same contract as the former reembed)
already_done = False
if spark.catalog.tableExists(CELLS_TABLE):
    existing_rows = spark.table(CELLS_TABLE).count()
    existing_dim = (spark.table(CELLS_TABLE).selectExpr("size(embedding) as d").limit(1).collect()[0]["d"]
                    if existing_rows > 0 else None)
    print(f"{CELLS_TABLE} already has {existing_rows:,} rows, dim={existing_dim}")
    if existing_rows >= target_n_cells * 0.95 and existing_dim == EXPECTED_DIM:
        print("Looks complete and dim matches — skipping. Drop the table to force re-run.")
        already_done = True

# Guard: 03a must have produced the token stage.
if not already_done:
    try:
        # recursiveFileLookup: 03a writes nested chunk_<ts>/ sub-dirs; Ray's read_parquet below
        # recurses on its own, but this Spark guard needs it to avoid UNABLE_TO_INFER_SCHEMA.
        _staged = spark.read.option("recursiveFileLookup", "true").parquet(TOKENS_STAGE_DIR).count()
    except Exception as e:
        raise RuntimeError(f"No staged tokens at {TOKENS_STAGE_DIR} — run 03a_stage_reference first. ({e})")
    print(f"Staged tokens available: {_staged:,} cells")

# COMMAND ----------

# DBTITLE 1,Ray: N single-A10 workers — pure TEDDY forward over the staged tokens (no Census)
if not already_done:
    from serverless_gpu.ray import ray_launch

    @ray_launch(gpus=NUM_GPU_WORKERS, gpu_type="a10", remote=True)
    def embed_with_ray():
        import inspect, sys
        import numpy as np
        import ray
        import torch

        print(f"Ray cluster resources: {ray.cluster_resources()}", flush=True)

        class TeddyEmbedActor:
            """One per GPU. Loads TEDDY from the pre-staged snapshot Volume once; each batch is the
            precomputed top-k gene token IDs → reconstruct rank values + CLS + attn → forward → pool.
            No Census handle, no get_anndata, no top-k (all done in 03a)."""

            def __init__(self):
                if SNAPSHOT_DIR not in sys.path:
                    sys.path.insert(0, SNAPSHOT_DIR)
                from teddy.models.model_directory import get_architecture, model_dict

                arch = get_architecture(MODEL_DIR)
                config = model_dict[arch]["config_cls"].from_pretrained(MODEL_DIR)
                self.model = model_dict[arch]["model_cls"].from_pretrained(MODEL_DIR, config=config)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device).eval()

                self.fwd_params = set(inspect.signature(self.model.forward).parameters.keys())
                self.add_cls = bool(getattr(config, "add_cls", False))
                self.cls_token_id = int(getattr(config, "cls_token_id", 0))
                self.d_model = int(config.d_model)
                self.use_bf16 = (self.device == "cuda")
                print(f"TEDDY-G {MODEL_SIZE} on {self.device}, d_model={self.d_model}", flush=True)

            def _assemble(self, gene_ids_np):
                # gene_ids_np: (b, k) int top-k token ids (from 03a). Rebuild the exact former inputs.
                gene_ids_b = torch.tensor(gene_ids_np, dtype=torch.long, device=self.device)
                b, k = gene_ids_b.shape
                rank_vec = torch.linspace(1.0, -1.0, steps=k, device=self.device)
                gene_values = rank_vec.unsqueeze(0).expand(b, -1).clone()
                if self.add_cls:
                    cls_col = torch.full((b, 1), self.cls_token_id, dtype=gene_ids_b.dtype, device=self.device)
                    gene_ids_b = torch.cat([cls_col, gene_ids_b], dim=1)
                    ones_col = torch.ones(b, 1, dtype=gene_values.dtype, device=self.device)
                    gene_values = torch.cat([ones_col, gene_values], dim=1)
                attn = torch.ones_like(gene_ids_b, dtype=torch.long)
                return gene_ids_b, gene_values, attn

            def _forward(self, gene_ids_b, gene_values, attn):
                kw = {}
                for n in ("gene_ids", "input_ids"):
                    if n in self.fwd_params: kw[n] = gene_ids_b; break
                for n in ("gene_values", "values", "expression_values", "gene_value"):
                    if n in self.fwd_params: kw[n] = gene_values; break
                for n in ("attention_mask", "mask"):
                    if n in self.fwd_params: kw[n] = attn; break
                with torch.no_grad():
                    if self.use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return self.model(**kw)
                    return self.model(**kw)

            @staticmethod
            def _extract_hidden(out):
                def _get(o, n):
                    return o.get(n) if isinstance(o, dict) else getattr(o, n, None)
                for n in ("cell_emb", "cell_embedding", "pooled_output", "pooler_output"):
                    v = _get(out, n)
                    if isinstance(v, torch.Tensor) and v.dim() == 2:
                        return v, True
                for n in ("last_hidden_state", "hidden_states", "encoder_last_hidden_state"):
                    v = _get(out, n)
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        return v, v.dim() == 2
                if isinstance(out, (tuple, list)):
                    for v in out:
                        if isinstance(v, torch.Tensor) and v.dim() in (2, 3):
                            return v, v.dim() == 2
                if isinstance(out, torch.Tensor) and out.dim() in (2, 3):
                    return out, out.dim() == 2
                return None, False

            def __call__(self, batch):
                cols = ("cell_type", "disease", "tissue", "tissue_general", "dataset_id")
                toks = batch["gene_token_ids"]
                gene_ids_np = np.stack([np.asarray(t, dtype=np.int64) for t in toks])  # (n, k)
                n = gene_ids_np.shape[0]
                embeddings = np.zeros((n, self.d_model), dtype=np.float32)
                for s in range(0, n, BATCH_EMB):
                    e = min(s + BATCH_EMB, n)
                    gids, gvals, attn = self._assemble(gene_ids_np[s:e])
                    hidden, is_pooled = self._extract_hidden(self._forward(gids, gvals, attn))
                    if hidden is None:
                        raise RuntimeError("Could not extract hidden state from TEDDY forward")
                    emb = hidden if is_pooled else hidden.mean(dim=1)
                    embeddings[s:e] = emb.detach().cpu().float().numpy()
                return {
                    "cell_id":   np.array([str(x) for x in batch["soma_joinid"]], dtype=object),
                    "embedding": [row.tolist() for row in embeddings],
                    **{c: np.asarray(batch[c], dtype=object) for c in cols},
                }

        ds = ray.data.read_parquet(TOKENS_STAGE_DIR)
        print(f"Dataset rows: {ds.count()}", flush=True)
        result = ds.map_batches(
            TeddyEmbedActor,
            batch_size=RAY_BATCH,
            num_gpus=1,
            concurrency=NUM_GPU_WORKERS,
            batch_format="numpy",
        )
        result.write_parquet(OUTPUT_STAGE_PARQUET)
        print(f"Ray write_parquet complete → {OUTPUT_STAGE_PARQUET}", flush=True)

# COMMAND ----------

if not already_done:
    dbutils.fs.rm(OUTPUT_STAGE_PARQUET, recurse=True)
    import time as _time
    _t0 = _time.time()
    embed_with_ray.distributed()
    print(f"Ray embedding complete in {(_time.time()-_t0)/60:.1f} min")

# COMMAND ----------

# DBTITLE 1,Promote parquet → managed UC Delta table, enable CDF, verify
if not already_done:
    spark.sql(f"DROP TABLE IF EXISTS {CELLS_TABLE}")
    spark.sql(f"CREATE TABLE {CELLS_TABLE} USING DELTA AS SELECT * FROM parquet.`{OUTPUT_STAGE_PARQUET}`")
    print(f"Managed UC Delta table {CELLS_TABLE} created with {spark.table(CELLS_TABLE).count():,} rows")
    dbutils.fs.rm(OUTPUT_STAGE_PARQUET, recurse=True)

spark.sql(f"ALTER TABLE {CELLS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
from pyspark.sql import functions as F
stats = (spark.table(CELLS_TABLE).select(
    F.count("*").alias("n_rows"),
    F.countDistinct("cell_type").alias("n_cell_types"),
    F.min(F.size("embedding")).alias("min_dim"),
    F.max(F.size("embedding")).alias("max_dim"),
).collect()[0].asDict())
print(stats)
display(spark.table(CELLS_TABLE).limit(5))
