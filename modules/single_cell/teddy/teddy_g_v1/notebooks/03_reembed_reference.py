# Databricks notebook source
# MAGIC %md
# MAGIC ### Re-embed the CELLxGENE Census reference with TEDDY
# MAGIC
# MAGIC Pulls a curated subset from CELLxGENE Census (human, has disease label,
# MAGIC stratified across tissues + diseases), embeds each cell with TEDDY-G locally on
# MAGIC the GPU cluster, and writes a Delta table `{catalog}.{schema}.teddy_cells` with:
# MAGIC
# MAGIC - `cell_id     STRING` (primary key — Census soma_joinid as string)
# MAGIC - `embedding   ARRAY<FLOAT>` (d_model — 512 for 70M, 768 for 160M, 1024 for 400M)
# MAGIC - `cell_type   STRING` — from `cell_type` column of Census obs
# MAGIC - `disease     STRING` — from `disease` column
# MAGIC - `tissue      STRING`
# MAGIC - `dataset_id  STRING`
# MAGIC
# MAGIC Change Data Feed is enabled so the Vector Search Delta Sync index (notebook 04)
# MAGIC can track it. The notebook is idempotent — skips the rebuild if the table is
# MAGIC already populated to the target size.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Why a serial Python loop and not `pandas_udf` over Spark?
# MAGIC
# MAGIC This notebook embeds cells in a serial driver-side loop rather than fanning
# MAGIC out via `pandas_udf` + Spark. That's a deliberate trade-off for THIS setup:
# MAGIC
# MAGIC 1. **Single-node cluster, single GPU.** The orchestrator job runs on
# MAGIC    `num_workers=0` — there's one T4 on the driver. `pandas_udf` parallelises
# MAGIC    across Spark Python workers, but on a single-node cluster those workers
# MAGIC    spawn on the same machine and contend for the same GPU. No speedup, just
# MAGIC    CUDA-context thrashing.
# MAGIC 2. **Heavy, serialization-fragile model.** TEDDY-G is ~280 MB on disk plus
# MAGIC    the bundled `teddy/` source and `GeneTokenizer`. `pandas_udf` serialises
# MAGIC    the function (and its closure) to each worker via cloudpickle — the same
# MAGIC    cloudpickle path that bit us for ~7 rev iterations on the serving deploy.
# MAGIC    Lazy-load-inside-the-UDF workarounds exist but add complexity with no
# MAGIC    payoff on a single-node cluster.
# MAGIC 3. **Data origin is Python-native, not Spark.** Cells come from CELLxGENE
# MAGIC    Census (TileDB-SOMA) via the `cellxgene_census` Python API. Flow today is:
# MAGIC    SOMA → pandas → numpy → torch → numpy → pandas → Spark (only for the Delta
# MAGIC    write). To use `pandas_udf` properly, the Census data would have to be
# MAGIC    materialised into a Spark/Delta table first — significant up-front cost
# MAGIC    just to make the data partitionable.
# MAGIC 4. **CUDA contexts don't share across Python workers cleanly.** PyTorch's
# MAGIC    CUDA context is per-process; multiple Spark workers on one GPU need MPS
# MAGIC    or queueing to coexist.
# MAGIC
# MAGIC `pandas_udf` would be the right call when:
# MAGIC - The cluster is multi-node GPU (e.g., 4 workers × T4) so each partition lands
# MAGIC   on its own GPU.
# MAGIC - The source data is already in a partitioned Delta table (no Census API
# MAGIC   conversion at the head of the pipeline).
# MAGIC - Use `applyInPandas` with `pandas_udf` over the partitioned DataFrame; each
# MAGIC   partition lazy-loads the model once per executor.
# MAGIC
# MAGIC Faster alternatives that fit the current single-node setup:
# MAGIC - **Bigger driver GPU** (A10 g5.4xlarge with 24 GB → `BATCH_EMB=32+` vs current
# MAGIC   8 → ~4× faster throughput).
# MAGIC - **fp16 inference** (`torch.bfloat16` autocast → ~2× faster, half the VRAM).
# MAGIC - **Bigger batch with gradient checkpointing** (already grad-free here since
# MAGIC   inference; checkpointing not applicable).
# MAGIC
# MAGIC For a one-time-per-deploy 500k–2M-cell reference build, serial-loop-on-T4
# MAGIC ships in 3–6h and keeps the code simple. If the reference grows to 5M+ cells
# MAGIC and rebuilds become a frequent operation, switch to multi-node + `pandas_udf`.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q -r ../requirements.txt cellxgene-census==1.17.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_yyang_genesis_workbench", "Schema")
dbutils.widgets.text("cache_dir", "teddy", "Cache dir")
dbutils.widgets.text("teddy_model_size", "70M", "TEDDY-G variant")
dbutils.widgets.text("target_n_cells", "2000000", "Target reference cell count")
dbutils.widgets.text("per_stratum_cap", "30000", "Max cells per (tissue, disease) stratum")
dbutils.widgets.text("census_version", "2024-07-01", "CELLxGENE Census version (LTS tag or 'latest')")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
cache_dir = dbutils.widgets.get("cache_dir")
model_size = dbutils.widgets.get("teddy_model_size")
target_n_cells = int(dbutils.widgets.get("target_n_cells"))
per_stratum_cap = int(dbutils.widgets.get("per_stratum_cap"))
census_version = dbutils.widgets.get("census_version")

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
snapshot_dir = f"{cache_full_path}/snapshots/main"
teddy_pkg_dir = f"{snapshot_dir}/teddy"
model_dir = f"{teddy_pkg_dir}/models/teddy_g/{model_size}"

CELLS_TABLE = f"{catalog}.{schema}.teddy_cells"
# Attention is O(B * H * S^2) — at S=2048 even modest batches blow T4's 16GB.
# 8 cells/batch * 8 heads * 2048^2 * fp32 ≈ 1 GB attention buffer; leaves room
# for params (280 MB), activations, and gradient-free overhead.
BATCH_EMB = 8        # cells per GPU forward pass
BATCH_WRITE = 50_000 # cells per Spark write batch

print(f"Cache: {cache_full_path}")
print(f"Model: TEDDY-G {model_size} at {model_dir}")
print(f"Target table: {CELLS_TABLE}")
print(f"Target cells: {target_n_cells:,} (cap {per_stratum_cap:,}/stratum)")

# COMMAND ----------

# DBTITLE 1,Idempotency check
already_done = False
if spark.catalog.tableExists(CELLS_TABLE):
    existing = spark.table(CELLS_TABLE).count()
    print(f"{CELLS_TABLE} already has {existing:,} rows")
    if existing >= target_n_cells * 0.95:  # tolerate small undercount from stratum balancing
        print("Looks complete — skipping rebuild. Drop the table if you want to re-run.")
        already_done = True

# COMMAND ----------

# DBTITLE 1,Connect to CELLxGENE Census and build stratified obs sample
if not already_done:
    import cellxgene_census
    import numpy as np
    import pandas as pd

    print(f"Opening CELLxGENE Census {census_version}…")
    census = cellxgene_census.open_soma(census_version=census_version)
    obs = census["census_data"]["homo_sapiens"].obs

    # Pull only the columns we need + soma_joinid for X lookup.
    # Filter to cells with non-null disease at the SOMA query level so we
    # don't materialize hundreds of millions of irrelevant rows.
    obs_df = (
        obs.read(
            column_names=[
                "soma_joinid", "cell_type", "disease", "tissue_general",
                "tissue", "dataset_id", "assay", "is_primary_data",
            ],
            value_filter="is_primary_data == True and disease != 'normal'",
        )
        .concat()
        .to_pandas()
    )
    print(f"Census cells (primary, disease != normal): {len(obs_df):,}")

    # Use tissue_general (~50 categories) for the strata to keep the cardinality manageable.
    # Cast to plain string first — Census returns these as pandas Categorical, and
    # .fillna("unknown") on a Categorical with no "unknown" category raises TypeError.
    _tissue = obs_df["tissue_general"].astype("object").where(obs_df["tissue_general"].notna(), "unknown")
    _disease = obs_df["disease"].astype("object").where(obs_df["disease"].notna(), "unknown")
    obs_df["__stratum"] = _tissue.astype(str) + " | " + _disease.astype(str)
    strata_counts = obs_df["__stratum"].value_counts()
    print(f"Distinct (tissue_general, disease) strata: {len(strata_counts)}")
    print(f"Top 10 strata:\n{strata_counts.head(10)}")

    # Stratified sample: cap each stratum, then if we still need to hit target_n_cells
    # top up uniformly across the strata that have headroom.
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
    print(f"Selected {len(obs_sample):,} cells across {obs_sample['__stratum'].nunique()} strata")

# COMMAND ----------

# DBTITLE 1,Load TEDDY locally on the GPU cluster (skipping the serving endpoint for throughput)
if not already_done:
    import sys, os, json, torch
    if snapshot_dir not in sys.path:
        sys.path.insert(0, snapshot_dir)

    from teddy.models.model_directory import get_architecture, model_dict
    from teddy.tokenizer.gene_tokenizer import GeneTokenizer

    arch = get_architecture(model_dir)
    config_cls = model_dict[arch]["config_cls"]
    model_cls = model_dict[arch]["model_cls"]
    config = config_cls.from_pretrained(model_dir)
    model = model_cls.from_pretrained(model_dir, config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tokenizer = GeneTokenizer.from_pretrained(model_dir)
    d_model = int(config.d_model)
    add_cls = bool(getattr(config, "add_cls", False))
    cls_token_id = int(getattr(config, "cls_token_id", 0))
    max_seq_len = int(getattr(config, "max_position_embeddings", 2048))
    print(f"TEDDY-G {model_size} loaded on {device}, d_model={d_model}")

# COMMAND ----------

# DBTITLE 1,Embed cells in GPU batches and write Delta in larger Spark batches
if not already_done:
    import inspect
    import scipy.sparse
    from pyspark.sql.types import (
        StructType, StructField, StringType, ArrayType, FloatType,
    )

    spark.sql(f"DROP TABLE IF EXISTS {CELLS_TABLE}")

    schema_spark = StructType([
        StructField("cell_id", StringType(), nullable=False),
        StructField("embedding", ArrayType(FloatType(), containsNull=False), nullable=False),
        StructField("cell_type", StringType(), nullable=True),
        StructField("disease", StringType(), nullable=True),
        StructField("tissue", StringType(), nullable=True),
        StructField("tissue_general", StringType(), nullable=True),
        StructField("dataset_id", StringType(), nullable=True),
    ])

    fwd_params = set(inspect.signature(model.forward).parameters.keys())

    _unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    def encode_gene_names(names):
        # TEDDY's GeneTokenizer returns None (not unk_token_id) for OOV tokens
        # from both encode() and convert_tokens_to_ids(). Substitute manually.
        ids = tokenizer.convert_tokens_to_ids(list(names))
        ids = [_unk_id if i is None else i for i in ids]
        return torch.tensor(ids, dtype=torch.long)

    def build_batch(X_dense, token_array):
        X_t = torch.tensor(X_dense, dtype=torch.float32)
        seq_tokens = max_seq_len - 1 if add_cls else max_seq_len
        k = min(seq_tokens, X_t.shape[1])
        _vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)
        gene_ids = token_array[top_idx]
        rank_vec = torch.linspace(1.0, -1.0, steps=k)
        gene_values = rank_vec.unsqueeze(0).expand(X_t.shape[0], -1).clone()
        if add_cls:
            batch = X_t.shape[0]
            cls_col = torch.full((batch, 1), cls_token_id, dtype=gene_ids.dtype)
            gene_ids = torch.cat([cls_col, gene_ids], dim=1)
            ones_col = torch.ones(batch, 1, dtype=gene_values.dtype)
            gene_values = torch.cat([ones_col, gene_values], dim=1)
        attn = torch.ones_like(gene_ids, dtype=torch.long)
        return gene_ids.to(device), gene_values.to(device), attn.to(device)

    def call_forward(gene_ids, gene_values, attn):
        kw = {}
        for n in ("gene_ids", "input_ids"):
            if n in fwd_params:
                kw[n] = gene_ids; break
        for n in ("gene_values", "values", "expression_values", "gene_value"):
            if n in fwd_params:
                kw[n] = gene_values; break
        for n in ("attention_mask", "mask"):
            if n in fwd_params:
                kw[n] = attn; break
        with torch.no_grad():
            return model(**kw)

    def extract_hidden(out):
        """Return (tensor, is_pooled). is_pooled=True means already (batch, d_model)."""
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

    # Use cellxgene_census.get_anndata() to pull (X, obs, var) in CONSISTENT row
    # order — the raw SOMA `.X.read(coords=...).tocsr()` returns a sparse matrix
    # indexed by soma_joinid (not positional row), which silently misaligns rows
    # when joinids are sparse. get_anndata() handles that for us.
    soma_ids = obs_sample["soma_joinid"].astype(int).to_numpy()
    total = len(soma_ids)
    written = 0
    token_array = None  # built lazily on first chunk so we use the right gene order

    for write_start in range(0, total, BATCH_WRITE):
        write_end = min(write_start + BATCH_WRITE, total)
        chunk_ids = soma_ids[write_start:write_end].tolist()

        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=chunk_ids,
        )
        # adata.shape: (n_cells_returned, n_genes). Rows aligned with adata.obs.
        n_cells = adata.shape[0]
        if n_cells == 0:
            print(f"chunk {write_start}-{write_end}: 0 cells returned, skipping")
            continue
        if n_cells != (write_end - write_start):
            print(f"chunk {write_start}-{write_end}: requested {write_end - write_start}, got {n_cells} (some joinids missing)")

        # First chunk: build the gene-id token array using the CONSISTENT var
        # order that get_anndata returns. var_df.shape may vary chunk-to-chunk
        # in theory but in practice Census uses the global gene vocab.
        if token_array is None:
            gene_ids_list = adata.var["feature_id"].astype(str).tolist()
            token_array = encode_gene_names(gene_ids_list)
            n_unk = int((token_array == _unk_id).sum().item())
            print(f"Census gene vocab: {len(gene_ids_list):,}  (unk after lookup: {n_unk:,})")

        # Densify X. AnnData's X is scipy.sparse.csr or numpy depending on size.
        X = adata.X
        if hasattr(X, "toarray"):
            X_dense_full = X.toarray()
        else:
            X_dense_full = np.asarray(X)

        embeddings = np.zeros((n_cells, d_model), dtype=np.float32)
        for s in range(0, n_cells, BATCH_EMB):
            e = min(s + BATCH_EMB, n_cells)
            X_batch = X_dense_full[s:e]
            gene_ids, gene_values, attn = build_batch(X_batch, token_array)
            fwd = call_forward(gene_ids, gene_values, attn)
            hidden, is_pooled = extract_hidden(fwd)
            if hidden is None:
                raise RuntimeError(f"Could not extract hidden state from TEDDY forward: {type(fwd).__name__}")
            emb = hidden if is_pooled else hidden.mean(dim=1)
            embeddings[s:e] = emb.detach().cpu().numpy().astype(np.float32)

        # adata.obs is guaranteed row-aligned with adata.X (this is the whole
        # point of get_anndata over raw SOMA reads).
        obs = adata.obs.reset_index(drop=True)
        def _strcol(col):
            return obs[col].astype("object").where(obs[col].notna(), None).astype(object).map(
                lambda v: None if v is None else str(v)
            )
        pdf = pd.DataFrame({
            "cell_id": obs["soma_joinid"].astype(str).to_numpy(),
            "embedding": [row.tolist() for row in embeddings],
            "cell_type": _strcol("cell_type"),
            "disease": _strcol("disease"),
            "tissue": _strcol("tissue"),
            "tissue_general": _strcol("tissue_general"),
            "dataset_id": _strcol("dataset_id"),
        })

        sdf = spark.createDataFrame(pdf, schema=schema_spark)
        sdf.write.format("delta").mode("append").saveAsTable(CELLS_TABLE)
        written += n_cells
        print(f"Wrote {written:,}/{total:,} cells")

    print(f"Done. Total rows in {CELLS_TABLE}: {spark.table(CELLS_TABLE).count():,}")

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
