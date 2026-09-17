# Databricks notebook source
# MAGIC %md
# MAGIC ### TEDDY reference re-embed — Serverless GPU + Ray Data
# MAGIC
# MAGIC Ray Data port of the former 4×A10 **multi-node Spark `mapInPandas`** reembed.
# MAGIC Uses `serverless_gpu.ray.ray_launch` to bring up a Ray cluster across N
# MAGIC single-A10 **serverless-GPU** nodes, then
# MAGIC `ray.data.read_parquet → map_batches(TeddyEmbedActor) → write_parquet`.
# MAGIC This replaces the classic multi-node GPU cluster (blocked by the workshop's
# MAGIC EC2 GPU vCPU quota=0) with serverless GPU — no Spark GPU cluster, no
# MAGIC `spark.task.resource.gpu.amount`, no manual partition sharding.
# MAGIC
# MAGIC **Data path.** UC *managed tables* can't be read/written by the non-Spark
# MAGIC Ray workers (UC vends storage creds only to Spark). UC *Volumes* are
# MAGIC FUSE-accessible from any client, so they bridge both ends:
# MAGIC   1. **Orchestrator (Spark/driver):** open Census, build the stratified obs
# MAGIC      sample, discover the Census gene vocab, and write `soma_joinid`+metadata
# MAGIC      (sorted by `soma_joinid` for contiguous TileDB-SOMA reads) as parquet to
# MAGIC      a Volume input stage; write the gene vocab to a Volume JSON.
# MAGIC   2. **`@ray_launch`:** each actor loads TEDDY from the pre-staged snapshot
# MAGIC      Volume + opens its own Census handle, reads the input parquet, fetches X
# MAGIC      per batch via `get_anndata`, embeds (bf16), and `write_parquet` to a
# MAGIC      Volume output stage — each worker writes its own blocks in parallel.
# MAGIC   3. **Orchestrator (Spark):** CTAS the parquet into the **managed** UC Delta
# MAGIC      table `teddy_cells`, enable CDF, and clean up the stages.
# MAGIC
# MAGIC The embed math (Census `get_anndata` → densify → top-k gene ranking →
# MAGIC TEDDY forward (bf16) → mean-pool) is identical to the former `mapInPandas`
# MAGIC worker. TEDDY weights come from the pre-staged snapshot Volume (serverless
# MAGIC egress can't reach HF's LFS CDN — see 01_register_teddy).

# COMMAND ----------

# MAGIC # cellxgene-census (tiledbsoma) ships compiled extensions built against numpy<2;
# MAGIC # pin numpy==1.26.4 for this task so the Census read doesn't hit
# MAGIC # "numpy.core.multiarray failed to import". torch 2.7.1 / transformers 4.44
# MAGIC # run fine on numpy 1.26. (This pin is local to the reembed task — the
# MAGIC # register/serving path keeps the runtime numpy 2.x.)
# MAGIC # ray[data] is NOT preinstalled on this serverless GPU runtime (serverless_gpu.ray
# MAGIC # is, but it just does `import ray`), so install it here — ray_launch bootstraps
# MAGIC # the Ray cluster and ray.data does the distributed map_batches. pyarrow==15.0.2
# MAGIC # matches the reference's known-good combo.
# MAGIC %pip install -q -r ../requirements.txt cellxgene-census==1.17.0 pyarrow==15.0.2 numpy==1.26.4 "ray[data]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_yyang_genesis_workbench", "Schema")
dbutils.widgets.text("cache_dir", "teddy", "Cache dir")
dbutils.widgets.text("teddy_model_size", "400M", "TEDDY-G variant")
dbutils.widgets.text("target_n_cells", "2000000", "Target reference cell count")
dbutils.widgets.text("per_stratum_cap", "30000", "Max cells per (tissue, disease) stratum")
dbutils.widgets.text("census_version", "2024-07-01", "CELLxGENE Census version (LTS tag or 'latest')")
dbutils.widgets.text("num_gpu_workers", "4", "Number of single-A10 serverless-GPU Ray workers")
dbutils.widgets.text("obs_limit", "0", "Cap Census obs scan to first N soma_joinids (0 = full scan; >0 for fast tests / bounded driver memory)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
model_size = dbutils.widgets.get("teddy_model_size")
target_n_cells = int(dbutils.widgets.get("target_n_cells"))
per_stratum_cap = int(dbutils.widgets.get("per_stratum_cap"))
census_version = dbutils.widgets.get("census_version")
num_gpu_workers = int(dbutils.widgets.get("num_gpu_workers"))
obs_limit = int(dbutils.widgets.get("obs_limit"))

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# Module-level constants (ALL_CAPS) — @ray_launch serializes the referenced
# globals to the remote Ray workers, so everything the actor needs is here.
SNAPSHOT_DIR = f"{cache_full_path}/snapshots/main"
MODEL_DIR = f"{SNAPSHOT_DIR}/teddy/models/teddy_g/{model_size}"
CENSUS_VERSION = census_version
MODEL_SIZE = model_size
NUM_GPU_WORKERS = num_gpu_workers
# Cells per get_anndata / map_batches call (contiguous soma_joinids → sequential
# TileDB-SOMA S3 reads). Inside each batch the GPU forward sub-batches by BATCH_EMB.
CENSUS_FETCH_BATCH = 10_000
# Per-forward GPU batch. A10 + bf16 + 400M: batch=48 ≈ 22-24 GB. Drop to 40 on OOM.
BATCH_EMB = 48
# UC Volume bridge stages (FUSE-accessible from Ray workers).
INPUT_STAGE_DIR = f"{cache_full_path}/reembed_input_stage"
OUTPUT_STAGE_PARQUET = f"{cache_full_path}/reembed_output_stage_parquet"
GENE_VOCAB_PATH = f"{cache_full_path}/reembed_census_gene_vocab.json"

CELLS_TABLE = f"{catalog}.{schema}.teddy_cells"
EXPECTED_DIM = {"70M": 512, "160M": 768, "400M": 1024}.get(model_size, 1024)

print(f"Cache: {cache_full_path}")
print(f"Model: TEDDY-G {model_size} at {MODEL_DIR}")
print(f"Target table: {CELLS_TABLE}")
print(f"Target cells: {target_n_cells:,} (cap {per_stratum_cap:,}/stratum), {num_gpu_workers} A10 Ray workers")

# COMMAND ----------

# DBTITLE 1,Idempotency check
already_done = False
if spark.catalog.tableExists(CELLS_TABLE):
    existing_rows = spark.table(CELLS_TABLE).count()
    existing_dim = (
        spark.table(CELLS_TABLE).selectExpr("size(embedding) as d").limit(1).collect()[0]["d"]
        if existing_rows > 0 else None
    )
    print(f"{CELLS_TABLE} already has {existing_rows:,} rows, embedding dim={existing_dim}")
    if existing_rows >= target_n_cells * 0.95 and existing_dim == EXPECTED_DIM:
        print("Looks complete and dim matches — skipping rebuild. Drop the table to force re-run.")
        already_done = True
    elif existing_dim is not None and existing_dim != EXPECTED_DIM:
        print(f"Embedding dim mismatch (have {existing_dim}, want {EXPECTED_DIM} for {model_size}) — rebuilding.")

# COMMAND ----------

# DBTITLE 1,Orchestrator: open Census and build stratified obs sample
if not already_done:
    import cellxgene_census
    import numpy as np
    import pandas as pd

    print(f"Opening CELLxGENE Census {census_version}…")
    census = cellxgene_census.open_soma(census_version=census_version)
    obs = census["census_data"]["homo_sapiens"].obs

    # Include healthy + disease cells (filtering disease!='normal' biased KNN
    # annotation toward disease cells and under-represented NK/naive T etc.).
    _read_kwargs = dict(
        column_names=[
            "soma_joinid", "cell_type", "disease", "tissue_general",
            "tissue", "dataset_id", "assay", "is_primary_data",
        ],
        value_filter="is_primary_data == True",
    )
    # obs_limit>0 caps the scan to the first N soma_joinids — fast tests + bounded
    # driver memory (the full primary-obs scan is ~50M rows into driver pandas).
    if obs_limit > 0:
        _read_kwargs["coords"] = (slice(0, obs_limit - 1),)
        print(f"obs_limit={obs_limit:,} — capping Census obs scan (test / bounded-memory mode)")
    obs_df = obs.read(**_read_kwargs).concat().to_pandas()
    print(f"Census cells (primary, healthy + disease): {len(obs_df):,}")

    _tissue = obs_df["tissue_general"].astype("object").where(obs_df["tissue_general"].notna(), "unknown")
    _disease = obs_df["disease"].astype("object").where(obs_df["disease"].notna(), "unknown")
    obs_df["__stratum"] = _tissue.astype(str) + " | " + _disease.astype(str)
    strata_counts = obs_df["__stratum"].value_counts()
    print(f"Distinct (tissue_general, disease) strata: {len(strata_counts)}")
    print(f"Top 10 strata:\n{strata_counts.head(10)}")

    rng = np.random.default_rng(seed=42)
    sampled_parts = []
    for stratum, idx in obs_df.groupby("__stratum").groups.items():
        idx = np.array(idx)
        if len(idx) <= per_stratum_cap:
            sampled_parts.append(idx)
        else:
            sampled_parts.append(rng.choice(idx, size=per_stratum_cap, replace=False))
    sampled_idx = np.concatenate(sampled_parts)
    rng.shuffle(sampled_idx)
    if len(sampled_idx) > target_n_cells:
        sampled_idx = sampled_idx[:target_n_cells]

    obs_sample = obs_df.iloc[sampled_idx].reset_index(drop=True)
    n_to_embed = len(obs_sample)
    print(f"Selected {n_to_embed:,} cells across {obs_sample['__stratum'].nunique()} strata")
    del census

# COMMAND ----------

# DBTITLE 1,Discover Census gene vocab (once) → write to a Volume JSON for the workers
if not already_done:
    import cellxgene_census as _cc_for_var
    import json, os

    _c2 = _cc_for_var.open_soma(census_version=census_version)
    _var = (
        _c2["census_data"]["homo_sapiens"]
        .ms["RNA"]
        .var.read(column_names=["soma_joinid", "feature_id"])
        .concat()
        .to_pandas()
    )
    # Census ordering: var rows match the gene axis order get_anndata returns.
    census_gene_ids = _var.sort_values("soma_joinid")["feature_id"].astype(str).tolist()
    print(f"Census gene vocab size: {len(census_gene_ids):,}")
    del _c2

    os.makedirs(cache_full_path, exist_ok=True)
    with open(GENE_VOCAB_PATH, "w") as f:
        json.dump(census_gene_ids, f)
    print(f"Wrote gene vocab → {GENE_VOCAB_PATH}")

# COMMAND ----------

# DBTITLE 1,Stage soma_joinid + metadata as parquet (sorted for contiguous Census reads)
if not already_done:
    # Sort by soma_joinid so the parquet — and therefore each Ray batch — holds a
    # contiguous joinid range. CELLxGENE Census is TileDB-SOMA backed; contiguous
    # obs_coords make get_anndata do sequential S3 reads instead of scattered
    # point reads (the dominant cost on the old run). Written via pandas/pyarrow
    # (not Spark) to preserve a single global sort order in the parquet.
    obs_sample_sorted = obs_sample.sort_values("soma_joinid").reset_index(drop=True)
    stage_pdf = pd.DataFrame({
        "soma_joinid":     obs_sample_sorted["soma_joinid"].astype("int64").to_numpy(),
        "cell_type":       obs_sample_sorted["cell_type"].astype(str).fillna("").to_numpy(),
        "disease":         obs_sample_sorted["disease"].astype(str).fillna("").to_numpy(),
        "tissue":          obs_sample_sorted["tissue"].astype(str).fillna("").to_numpy(),
        "tissue_general":  obs_sample_sorted["tissue_general"].astype(str).fillna("").to_numpy(),
        "dataset_id":      obs_sample_sorted["dataset_id"].astype(str).fillna("").to_numpy(),
    })

    dbutils.fs.rm(INPUT_STAGE_DIR, recurse=True)
    os.makedirs(INPUT_STAGE_DIR, exist_ok=True)
    # A handful of row-group-ordered files keeps Ray read parallelism up while
    # preserving order; one file is simplest and fine at this scale.
    stage_pdf.to_parquet(f"{INPUT_STAGE_DIR}/obs_sample.parquet", index=False, row_group_size=CENSUS_FETCH_BATCH)
    print(f"Staged {len(stage_pdf):,} rows → {INPUT_STAGE_DIR}")

# COMMAND ----------

# DBTITLE 1,Ray: provision N single-A10 serverless-GPU workers + embed
if not already_done:
    from serverless_gpu.ray import ray_launch

    @ray_launch(gpus=NUM_GPU_WORKERS, gpu_type="a10", remote=True)
    def embed_with_ray():
        import inspect, json, sys, time
        import numpy as np
        import ray
        import torch
        import cellxgene_census

        print(f"Ray cluster resources: {ray.cluster_resources()}", flush=True)

        class TeddyEmbedActor:
            """Stateful Ray Data actor — one per GPU. Loads TEDDY from the
            pre-staged snapshot Volume once, opens its own Census handle, and
            embeds each batch of cells fetched from Census by soma_joinid."""

            def __init__(self):
                if SNAPSHOT_DIR not in sys.path:
                    sys.path.insert(0, SNAPSHOT_DIR)
                from teddy.models.model_directory import get_architecture, model_dict
                from teddy.tokenizer.gene_tokenizer import GeneTokenizer

                arch = get_architecture(MODEL_DIR)
                config = model_dict[arch]["config_cls"].from_pretrained(MODEL_DIR)
                self.model = model_dict[arch]["model_cls"].from_pretrained(MODEL_DIR, config=config)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device).eval()

                tokenizer = GeneTokenizer.from_pretrained(MODEL_DIR)
                unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
                gene_names = json.load(open(GENE_VOCAB_PATH))
                ids = tokenizer.convert_tokens_to_ids(list(gene_names))
                ids = [unk_id if i is None else i for i in ids]
                self.token_array = torch.tensor(ids, dtype=torch.long).to(self.device)

                self.fwd_params = set(inspect.signature(self.model.forward).parameters.keys())
                self.add_cls = bool(getattr(config, "add_cls", False))
                self.cls_token_id = int(getattr(config, "cls_token_id", 0))
                self.d_model = int(config.d_model)
                self.max_seq_len = int(getattr(config, "max_position_embeddings", 2048))
                self.use_bf16 = (self.device == "cuda")
                self.census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
                print(f"TEDDY-G {MODEL_SIZE} on {self.device}, d_model={self.d_model}, "
                      f"vocab={len(gene_names):,}", flush=True)

            def _build_batch(self, X_dense):
                X_t = torch.tensor(X_dense, dtype=torch.float32, device=self.device)
                seq_tokens = self.max_seq_len - 1 if self.add_cls else self.max_seq_len
                k = min(seq_tokens, X_t.shape[1])
                _vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)
                gene_ids_b = self.token_array[top_idx]
                rank_vec = torch.linspace(1.0, -1.0, steps=k, device=self.device)
                gene_values = rank_vec.unsqueeze(0).expand(X_t.shape[0], -1).clone()
                if self.add_cls:
                    b = X_t.shape[0]
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
                soma_ids = [int(x) for x in batch["soma_joinid"]]
                # metadata keyed by soma_joinid to realign with Census's return order
                cols = ("cell_type", "disease", "tissue", "tissue_general", "dataset_id")
                meta = {int(batch["soma_joinid"][i]): {c: str(batch[c][i]) for c in cols}
                        for i in range(len(soma_ids))}

                adata = cellxgene_census.get_anndata(
                    self.census, organism="Homo sapiens", obs_coords=soma_ids,
                )
                n_cells = adata.shape[0]
                if n_cells == 0:
                    return {"cell_id": np.array([], dtype=object), "embedding": [],
                            **{c: np.array([], dtype=object) for c in cols}}

                X = adata.X
                X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                embeddings = np.zeros((n_cells, self.d_model), dtype=np.float32)
                for s in range(0, n_cells, BATCH_EMB):
                    e = min(s + BATCH_EMB, n_cells)
                    gids, gvals, attn = self._build_batch(X_dense[s:e])
                    hidden, is_pooled = self._extract_hidden(self._forward(gids, gvals, attn))
                    if hidden is None:
                        raise RuntimeError("Could not extract hidden state from TEDDY forward")
                    emb = hidden if is_pooled else hidden.mean(dim=1)
                    embeddings[s:e] = emb.detach().cpu().float().numpy()

                returned = adata.obs["soma_joinid"].astype("int64").to_numpy()
                def _n(v): return v if v else None
                return {
                    "cell_id":        np.array([str(s) for s in returned], dtype=object),
                    "embedding":      [row.tolist() for row in embeddings],
                    "cell_type":      np.array([_n(meta[int(s)]["cell_type"]) for s in returned], dtype=object),
                    "disease":        np.array([_n(meta[int(s)]["disease"]) for s in returned], dtype=object),
                    "tissue":         np.array([_n(meta[int(s)]["tissue"]) for s in returned], dtype=object),
                    "tissue_general": np.array([_n(meta[int(s)]["tissue_general"]) for s in returned], dtype=object),
                    "dataset_id":     np.array([_n(meta[int(s)]["dataset_id"]) for s in returned], dtype=object),
                }

        ds = ray.data.read_parquet(INPUT_STAGE_DIR)
        print(f"Dataset rows: {ds.count()}", flush=True)
        result = ds.map_batches(
            TeddyEmbedActor,
            batch_size=CENSUS_FETCH_BATCH,
            num_gpus=1,                     # one GPU per actor
            concurrency=NUM_GPU_WORKERS,    # one actor per A10 node
            batch_format="numpy",
        )
        # Each worker writes its own blocks to the Volume in parallel — no
        # driver-side accumulation, so memory stays bounded.
        result.write_parquet(OUTPUT_STAGE_PARQUET)
        print(f"Ray write_parquet complete → {OUTPUT_STAGE_PARQUET}", flush=True)

# COMMAND ----------

if not already_done:
    dbutils.fs.rm(OUTPUT_STAGE_PARQUET, recurse=True)
    import time as _time
    _t0 = _time.time()
    embed_with_ray.distributed()
    print(f"Ray embedding job complete in {(_time.time()-_t0)/60:.1f} min")

# COMMAND ----------

# DBTITLE 1,Promote parquet output → managed UC Delta table (Spark CTAS)
if not already_done:
    spark.sql(f"DROP TABLE IF EXISTS {CELLS_TABLE}")
    spark.sql(f"""
        CREATE TABLE {CELLS_TABLE}
        USING DELTA
        AS SELECT * FROM parquet.`{OUTPUT_STAGE_PARQUET}`
    """)
    _n_rows = spark.table(CELLS_TABLE).count()
    print(f"Managed UC Delta table {CELLS_TABLE} created with {_n_rows:,} rows")

# COMMAND ----------

# DBTITLE 1,Clean up the Volume stages
if not already_done:
    dbutils.fs.rm(INPUT_STAGE_DIR, recurse=True)
    dbutils.fs.rm(OUTPUT_STAGE_PARQUET, recurse=True)
    dbutils.fs.rm(GENE_VOCAB_PATH)
    print("Removed input/output stages + gene vocab JSON")

# COMMAND ----------

# DBTITLE 1,Enable CDF and verify
spark.sql(f"ALTER TABLE {CELLS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"CDF enabled on {CELLS_TABLE}")

from pyspark.sql import functions as F
stats = (
    spark.table(CELLS_TABLE)
    .select(
        F.count("*").alias("n_rows"),
        F.countDistinct("cell_type").alias("n_cell_types"),
        F.countDistinct("disease").alias("n_diseases"),
        F.countDistinct("tissue_general").alias("n_tissues"),
        F.min(F.size("embedding")).alias("min_dim"),
        F.max(F.size("embedding")).alias("max_dim"),
    )
    .collect()[0]
    .asDict()
)
print(stats)
display(spark.table(CELLS_TABLE).limit(5))
