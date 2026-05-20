# Databricks notebook source
# MAGIC %md
# MAGIC ### Re-embed the CELLxGENE Census reference with TEDDY — multi-node GPU
# MAGIC
# MAGIC Pulls a curated subset from CELLxGENE Census (human, stratified across
# MAGIC tissues + diseases), embeds each cell with TEDDY-G on a multi-node A10 GPU
# MAGIC cluster via `mapInPandas`, and writes a Delta table
# MAGIC `{catalog}.{schema}.teddy_cells` with:
# MAGIC
# MAGIC - `cell_id     STRING` (primary key — Census soma_joinid as string)
# MAGIC - `embedding   ARRAY<FLOAT>` (d_model — 512 for 70M, 768 for 160M, 1024 for 400M)
# MAGIC - `cell_type   STRING` — from `cell_type` column of Census obs
# MAGIC - `disease     STRING` — from `disease` column
# MAGIC - `tissue      STRING`
# MAGIC - `tissue_general STRING`
# MAGIC - `dataset_id  STRING`
# MAGIC
# MAGIC Change Data Feed is enabled so the Vector Search Delta Sync index (notebook 04)
# MAGIC can track it. The notebook is idempotent — skips the rebuild if the table is
# MAGIC already populated to the target size.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Why `mapInPandas` on a multi-node GPU cluster
# MAGIC
# MAGIC Earlier revisions of this notebook ran a serial driver-side loop on a single
# MAGIC A10 GPU — 5+ hours for 2M cells. The bottleneck was always GPU throughput
# MAGIC since attention is O(B * H * S^2). To get linear speedup we need more GPUs
# MAGIC running concurrently.
# MAGIC
# MAGIC The new design (matching the cluster spec `job_cluster_teddy_a10_multinode`
# MAGIC = 4 workers × 1 A10 each):
# MAGIC
# MAGIC 1. **Driver**: opens Census, builds a stratified obs sample, creates a Spark
# MAGIC    DataFrame containing only `soma_joinid`s + metadata, repartitions for
# MAGIC    even fanout, hands it to Spark.
# MAGIC 2. **Each worker** (via `mapInPandas`): opens Census once per Python process
# MAGIC    (S3-backed TileDB-SOMA — concurrent reads are safe), loads TEDDY-G once
# MAGIC    per process from the UC Volume, then for each Arrow batch fetches X via
# MAGIC    `cellxgene_census.get_anndata`, embeds with bf16 autocast at batch=32,
# MAGIC    yields a pandas DF that Spark writes to Delta.
# MAGIC 3. **`spark.task.resource.gpu.amount=1`** on the cluster ensures one Spark
# MAGIC    task per GPU — no CUDA context contention.
# MAGIC
# MAGIC Why this is safe / fast despite earlier reservations against `pandas_udf`:
# MAGIC - **Model serialization isn't an issue**: the model is loaded inside the
# MAGIC   UDF from the UC Volume, never sent through cloudpickle.
# MAGIC - **One CUDA context per worker process**: enforced by the Spark GPU
# MAGIC   scheduler config.
# MAGIC - **Census API is worker-safe**: `cellxgene_census.open_soma()` works from
# MAGIC   any Python process — it's just TileDB-SOMA reads from S3.
# MAGIC - **No Spark / Census conversion penalty up front**: we only ship
# MAGIC   `soma_joinid`s through Spark; workers fetch their own X slices in
# MAGIC   parallel from Census.
# MAGIC
# MAGIC #### Observing progress while the rebuild runs
# MAGIC
# MAGIC The final `mapInPandas(...).write.mode("overwrite").saveAsTable()` is an
# MAGIC atomic Spark write — the Delta table is only visible AT THE END. To watch
# MAGIC progress LIVE:
# MAGIC
# MAGIC 1. **Spark UI → Stages tab** (linked from the run page): the
# MAGIC    `mapInPandas` stage shows `tasks completed / N`. With N partitions
# MAGIC    across 4 workers, expect ~5-6 tasks per worker.
# MAGIC 2. **Cluster → Driver Logs → stderr**: each worker prints a tagged
# MAGIC    progress line per GPU batch and per Arrow batch (`[w=<host>]`).
# MAGIC 3. **GPU utilization** via `databricks clusters get <cluster_id>` or
# MAGIC    the cluster's Metrics tab — should be near 100 % on all 4 GPUs.
# MAGIC
# MAGIC If a worker is slow / stuck, the Spark UI's task duration histogram
# MAGIC will show it as a straggler (one outlier task duration).

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q -r ../requirements.txt cellxgene-census==1.17.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_yyang_genesis_workbench", "Schema")
dbutils.widgets.text("cache_dir", "teddy", "Cache dir")
dbutils.widgets.text("teddy_model_size", "400M", "TEDDY-G variant")
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
# Per-worker GPU forward batch. Empirically measured on A10 + bf16 + 400M:
# batch=32 → 17 GB VRAM, 60 % GPU compute util. Bumped to 48 to use the
# remaining ~7 GB of headroom and push compute toward saturation.
# Memory scales ~linearly with batch (attention activations + activations),
# so batch=48 should land at ~22-24 GB — at the edge but within A10's 24 GB.
# If you see CUDA OOM in worker stderr, drop to 40.
BATCH_EMB = 48
# Cells per pandas_udf Arrow batch (one input DF per call to the UDF). Each
# worker processes its partition as a stream of these.
ARROW_BATCH = 10_000

print(f"Cache: {cache_full_path}")
print(f"Model: TEDDY-G {model_size} at {model_dir}")
print(f"Target table: {CELLS_TABLE}")
print(f"Target cells: {target_n_cells:,} (cap {per_stratum_cap:,}/stratum)")

# COMMAND ----------

# DBTITLE 1,Idempotency check
already_done = False
if spark.catalog.tableExists(CELLS_TABLE):
    existing_rows = spark.table(CELLS_TABLE).count()
    existing_dim = (
        spark.table(CELLS_TABLE).selectExpr("size(embedding) as d").limit(1).collect()[0]["d"]
        if existing_rows > 0 else None
    )
    expected_dim = {"70M": 512, "160M": 768, "400M": 1024}.get(model_size)
    print(f"{CELLS_TABLE} already has {existing_rows:,} rows, embedding dim={existing_dim}")
    # Only consider it done if BOTH the row count AND the embedding dim match
    # the current variant — switching 70M → 400M means dim 512 → 1024, in
    # which case we MUST rebuild even if the row count looks fine.
    if existing_rows >= target_n_cells * 0.95 and existing_dim == expected_dim:
        print("Looks complete and dim matches — skipping rebuild. Drop the table to force re-run.")
        already_done = True
    elif existing_dim is not None and existing_dim != expected_dim:
        print(f"Embedding dim mismatch (have {existing_dim}, want {expected_dim} for {model_size}) — rebuilding.")

# COMMAND ----------

# DBTITLE 1,Driver: open Census and build stratified obs sample
if not already_done:
    import cellxgene_census
    import numpy as np
    import pandas as pd

    print(f"Opening CELLxGENE Census {census_version}…")
    census = cellxgene_census.open_soma(census_version=census_version)
    obs = census["census_data"]["homo_sapiens"].obs

    # NB: previously this filtered `disease != 'normal'`, which under-represented
    # healthy cell types (NK, naive T, etc.) in the reference and biased KNN
    # annotation toward disease cells. We now include healthy cells.
    obs_df = (
        obs.read(
            column_names=[
                "soma_joinid", "cell_type", "disease", "tissue_general",
                "tissue", "dataset_id", "assay", "is_primary_data",
            ],
            value_filter="is_primary_data == True",
        )
        .concat()
        .to_pandas()
    )
    print(f"Census cells (primary, healthy + disease): {len(obs_df):,}")

    # Stratify on (tissue_general, disease). Cast through object to dodge
    # pandas Categorical fillna issues.
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

    # Driver closes its Census handle — workers open their own.
    del census

# COMMAND ----------

# DBTITLE 1,Discover Census gene vocab (driver-side, one-time) for worker token cache
if not already_done:
    # Workers need the exact gene order Census returns inside get_anndata().
    # Census uses a global gene vocab per organism — same order for every cell.
    # We discover it once on the driver by reading the `var` axis, then pass
    # the gene list as a broadcast variable so all workers have identical
    # token_array without re-querying Census just to learn gene order.
    import cellxgene_census as _cc_for_var
    _c2 = _cc_for_var.open_soma(census_version=census_version)
    _var = (
        _c2["census_data"]["homo_sapiens"]
        .ms["RNA"]
        .var.read(column_names=["soma_joinid", "feature_id"])
        .concat()
        .to_pandas()
    )
    # Census ordering: rows in var match the gene axis index used by get_anndata.
    census_gene_ids = _var.sort_values("soma_joinid")["feature_id"].astype(str).tolist()
    print(f"Census gene vocab size: {len(census_gene_ids):,}")
    del _c2
    gene_ids_b = spark.sparkContext.broadcast(census_gene_ids)

# COMMAND ----------

# DBTITLE 1,Build Spark DataFrame of soma_joinids + metadata; repartition for fanout
if not already_done:
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, ArrayType, FloatType, LongType,
    )

    # CRITICAL: sort by soma_joinid before partitioning. CELLxGENE Census
    # is TileDB-SOMA backed; random-access reads (scattered soma_joinids in
    # one get_anndata call) trigger many S3 point reads. Sorting gives each
    # partition a contiguous range of joinids → orders of magnitude faster
    # Census fetch per batch. This was the dominant bottleneck on the prior
    # run (10+ min per 7,812-cell partition just for the Census read).
    obs_sample_sorted = obs_sample.sort_values("soma_joinid").reset_index(drop=True)

    input_pdf = pd.DataFrame({
        "soma_joinid": obs_sample_sorted["soma_joinid"].astype("int64").to_numpy(),
        "cell_type":     obs_sample_sorted["cell_type"].astype(str).fillna("").to_numpy(),
        "disease":       obs_sample_sorted["disease"].astype(str).fillna("").to_numpy(),
        "tissue":        obs_sample_sorted["tissue"].astype(str).fillna("").to_numpy(),
        "tissue_general":obs_sample_sorted["tissue_general"].astype(str).fillna("").to_numpy(),
        "dataset_id":    obs_sample_sorted["dataset_id"].astype(str).fillna("").to_numpy(),
    })

    input_schema = StructType([
        StructField("soma_joinid",    LongType(),   nullable=False),
        StructField("cell_type",      StringType(), nullable=True),
        StructField("disease",        StringType(), nullable=True),
        StructField("tissue",         StringType(), nullable=True),
        StructField("tissue_general", StringType(), nullable=True),
        StructField("dataset_id",     StringType(), nullable=True),
    ])

    # Target ~50k cells per partition. With 8 GPUs (8 concurrent tasks)
    # that's ~5 partitions per GPU — enough to absorb stragglers without
    # paying model-load + Census-setup overhead too many times.
    #
    # IMPORTANT: do NOT use `spark.sparkContext.defaultParallelism` as a
    # floor. On a g5.16xlarge multinode cluster (64 vCPU × N workers), that
    # number is in the hundreds, which produces tiny partitions and
    # serializes them through the GPU bottleneck. The prior 4-GPU run set
    # 256 partitions of 7,812 cells each, took ~9 min/partition due to S3
    # random-access in Census, and would not have completed in 12h.
    TARGET_PARTITION_SIZE = 50_000
    n_partitions = max(8, (n_to_embed + TARGET_PARTITION_SIZE - 1) // TARGET_PARTITION_SIZE)
    print(
        f"Input partitions: {n_partitions} (~{n_to_embed // max(n_partitions, 1):,} cells each); "
        f"sorted by soma_joinid for contiguous Census reads"
    )

    # repartitionByRange("soma_joinid") gives exactly n_partitions, each
    # containing a contiguous range of soma_joinids. This guarantees that
    # within each partition, the soma_ids passed to `get_anndata` form a
    # tight range — TileDB-SOMA fetches them with sequential S3 reads
    # instead of random point reads.
    input_sdf = (
        spark.createDataFrame(input_pdf, schema=input_schema)
        .repartitionByRange(n_partitions, "soma_joinid")
    )

# COMMAND ----------

# DBTITLE 1,Define the mapInPandas UDF (loads TEDDY locally on each worker)
if not already_done:
    # Capture vars by value for closure into the UDF — keep these pure Python
    # primitives so cloudpickle has no trouble.
    _snapshot_dir = snapshot_dir
    _model_dir = model_dir
    _model_size = model_size
    _census_version = census_version
    _batch_emb = BATCH_EMB
    _expected_dim = {"70M": 512, "160M": 768, "400M": 1024}.get(model_size, 1024)
    _gene_ids_bcast = gene_ids_b  # broadcast — cheap to send

    out_schema = StructType([
        StructField("cell_id",        StringType(), nullable=False),
        StructField("embedding",      ArrayType(FloatType(), containsNull=False), nullable=False),
        StructField("cell_type",      StringType(), nullable=True),
        StructField("disease",        StringType(), nullable=True),
        StructField("tissue",         StringType(), nullable=True),
        StructField("tissue_general", StringType(), nullable=True),
        StructField("dataset_id",     StringType(), nullable=True),
    ])

    def embed_partition(iterator):
        """Runs on each Spark executor. mapInPandas hands us an iterator of
        pandas DFs (one per Arrow batch ~10k cells) and we yield embedded DFs.

        Per-worker-process caching of model + tokenizer + Census handle +
        token_array is via a module-level global dict on the executor.

        Progress observability: each batch logs a tagged line to stderr with
        worker host, batch num, n_cells, GPU forward time, total elapsed, and
        throughput cells/s. Visible in Spark UI → Executors → stderr."""
        import sys
        import socket
        import time
        import inspect
        import numpy as np
        import pandas as pd
        import torch
        import cellxgene_census

        # Worker tag for log lines — host helps differentiate the 4 workers.
        _worker_tag = f"[w={socket.gethostname()}]"
        def _log(msg):
            print(f"{_worker_tag} {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)

        _partition_start = time.time()
        _partition_n_cells = 0
        _batch_num = 0

        # ---- one-shot per-process init ----
        global _TEDDY_WORKER_CACHE
        try:
            _TEDDY_WORKER_CACHE
        except NameError:
            _TEDDY_WORKER_CACHE = {}

        if "model" not in _TEDDY_WORKER_CACHE:
            _t_load = time.time()
            _log(f"loading TEDDY-G {_model_size} from {_model_dir}…")
            if _snapshot_dir not in sys.path:
                sys.path.insert(0, _snapshot_dir)
            from teddy.models.model_directory import get_architecture, model_dict
            from teddy.tokenizer.gene_tokenizer import GeneTokenizer

            arch = get_architecture(_model_dir)
            config_cls = model_dict[arch]["config_cls"]
            model_cls = model_dict[arch]["model_cls"]
            config = config_cls.from_pretrained(_model_dir)
            model = model_cls.from_pretrained(_model_dir, config=config)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()
            tokenizer = GeneTokenizer.from_pretrained(_model_dir)
            unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

            # Pre-tokenize the global Census gene vocab — same array reused for
            # every cell since get_anndata returns rows aligned to this vocab.
            gene_names = _gene_ids_bcast.value
            ids = tokenizer.convert_tokens_to_ids(list(gene_names))
            ids = [unk_id if i is None else i for i in ids]
            token_array = torch.tensor(ids, dtype=torch.long)

            fwd_params = set(inspect.signature(model.forward).parameters.keys())
            add_cls = bool(getattr(config, "add_cls", False))
            cls_token_id = int(getattr(config, "cls_token_id", 0))
            d_model = int(config.d_model)
            max_seq_len = int(getattr(config, "max_position_embeddings", 2048))

            _TEDDY_WORKER_CACHE.update({
                "model": model, "device": device, "fwd_params": fwd_params,
                "add_cls": add_cls, "cls_token_id": cls_token_id,
                "d_model": d_model, "max_seq_len": max_seq_len,
                "token_array": token_array, "unk_id": unk_id,
            })
            _log(f"TEDDY loaded on {device} in {time.time()-_t_load:.1f}s, d_model={d_model}, vocab={len(gene_names):,}")

        if "census" not in _TEDDY_WORKER_CACHE:
            _TEDDY_WORKER_CACHE["census"] = cellxgene_census.open_soma(
                census_version=_census_version
            )

        c = _TEDDY_WORKER_CACHE
        model = c["model"]; device = c["device"]
        fwd_params = c["fwd_params"]; add_cls = c["add_cls"]
        cls_token_id = c["cls_token_id"]; d_model = c["d_model"]
        max_seq_len = c["max_seq_len"]; token_array = c["token_array"]
        census = c["census"]

        # Move token_array to device once (CPU→GPU lookup once per worker).
        token_array_dev = token_array.to(device)

        # ---- helpers (closed over the per-worker model state) ----
        def build_batch(X_dense):
            # Build the whole batch on GPU — topk over 60k genes is the per-batch
            # hot path. CPU topk for batch=32 × 60k ≈ 2M comparisons; moving to
            # GPU made the original 70M reembed ~3× faster end-to-end.
            X_t = torch.tensor(X_dense, dtype=torch.float32, device=device)
            seq_tokens = max_seq_len - 1 if add_cls else max_seq_len
            k = min(seq_tokens, X_t.shape[1])
            _vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)
            gene_ids_b = token_array_dev[top_idx]
            rank_vec = torch.linspace(1.0, -1.0, steps=k, device=device)
            gene_values = rank_vec.unsqueeze(0).expand(X_t.shape[0], -1).clone()
            if add_cls:
                batch = X_t.shape[0]
                cls_col = torch.full((batch, 1), cls_token_id, dtype=gene_ids_b.dtype, device=device)
                gene_ids_b = torch.cat([cls_col, gene_ids_b], dim=1)
                ones_col = torch.ones(batch, 1, dtype=gene_values.dtype, device=device)
                gene_values = torch.cat([ones_col, gene_values], dim=1)
            attn = torch.ones_like(gene_ids_b, dtype=torch.long)
            return gene_ids_b, gene_values, attn

        _use_bf16 = (device == "cuda")

        def call_forward(gene_ids_b, gene_values, attn):
            kw = {}
            for n in ("gene_ids", "input_ids"):
                if n in fwd_params: kw[n] = gene_ids_b; break
            for n in ("gene_values", "values", "expression_values", "gene_value"):
                if n in fwd_params: kw[n] = gene_values; break
            for n in ("attention_mask", "mask"):
                if n in fwd_params: kw[n] = attn; break
            with torch.no_grad():
                if _use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        return model(**kw)
                return model(**kw)

        def extract_hidden(out):
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

        # ---- main loop: per Arrow batch ----
        for in_pdf in iterator:
            if in_pdf.empty:
                continue
            _batch_num += 1
            _batch_start = time.time()
            soma_ids = in_pdf["soma_joinid"].astype("int64").tolist()

            _t = time.time()
            adata = cellxgene_census.get_anndata(
                census, organism="Homo sapiens", obs_coords=soma_ids,
            )
            n_cells = adata.shape[0]
            _t_census = time.time() - _t
            if n_cells == 0:
                _log(f"batch {_batch_num}: 0 cells returned from Census, skipping")
                continue

            _t = time.time()
            X = adata.X
            X_dense_full = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            _t_densify = time.time() - _t

            _t = time.time()
            embeddings = np.zeros((n_cells, d_model), dtype=np.float32)
            for s in range(0, n_cells, _batch_emb):
                e = min(s + _batch_emb, n_cells)
                gids, gvals, attn = build_batch(X_dense_full[s:e])
                fwd = call_forward(gids, gvals, attn)
                hidden, is_pooled = extract_hidden(fwd)
                if hidden is None:
                    raise RuntimeError(
                        f"Could not extract hidden state from TEDDY forward: {type(fwd).__name__}"
                    )
                emb = hidden if is_pooled else hidden.mean(dim=1)
                embeddings[s:e] = emb.detach().cpu().numpy().astype(np.float32)
            _t_embed = time.time() - _t

            _partition_n_cells += n_cells
            _partition_elapsed = time.time() - _partition_start
            _batch_elapsed = time.time() - _batch_start
            _log(
                f"batch {_batch_num}: {n_cells} cells in {_batch_elapsed:.1f}s "
                f"(census {_t_census:.1f}s, densify {_t_densify:.1f}s, embed {_t_embed:.1f}s) | "
                f"partition: {_partition_n_cells:,} cells / {_partition_elapsed/60:.1f} min "
                f"({_partition_n_cells/max(_partition_elapsed,0.1):.1f} cells/s)"
            )

            # Align metadata with the row order Census returned (adata.obs.soma_joinid).
            obs_pd = adata.obs.reset_index(drop=True)
            returned_ids = obs_pd["soma_joinid"].astype("int64").to_numpy()
            in_pdf_indexed = in_pdf.set_index("soma_joinid")
            meta_pd = in_pdf_indexed.loc[returned_ids].reset_index()

            out_pdf = pd.DataFrame({
                "cell_id":        meta_pd["soma_joinid"].astype(str).to_numpy(),
                "embedding":      [row.tolist() for row in embeddings],
                "cell_type":      meta_pd["cell_type"].astype(str).where(meta_pd["cell_type"] != "", None),
                "disease":        meta_pd["disease"].astype(str).where(meta_pd["disease"] != "", None),
                "tissue":         meta_pd["tissue"].astype(str).where(meta_pd["tissue"] != "", None),
                "tissue_general": meta_pd["tissue_general"].astype(str).where(meta_pd["tissue_general"] != "", None),
                "dataset_id":     meta_pd["dataset_id"].astype(str).where(meta_pd["dataset_id"] != "", None),
            })
            yield out_pdf

        _log(
            f"partition done: {_partition_n_cells:,} cells in "
            f"{(time.time()-_partition_start)/60:.1f} min "
            f"({_partition_n_cells/max(time.time()-_partition_start,0.1):.1f} cells/s)"
        )

# COMMAND ----------

# DBTITLE 1,Run mapInPandas across workers and write Delta in one shot
if not already_done:
    import time as _time
    spark.sql(f"DROP TABLE IF EXISTS {CELLS_TABLE}")

    # Surface the Spark UI URL so progress is observable mid-run. mapInPandas
    # below is one big stage with N tasks; the stage's task progress is the
    # only live signal until the final write commits.
    _app_id = spark.sparkContext.applicationId
    try:
        _ui_url = spark.sparkContext.uiWebUrl
        print(f"Spark UI: {_ui_url}  (watch the mapInPandas stage's tasks completed/N)")
    except Exception:
        print(f"Spark UI: open the cluster's Spark UI (app id {_app_id})")
    print(f"Workers will log per-batch progress to executor stderr — see Spark UI → Executors.")

    _t0 = _time.time()
    result_sdf = input_sdf.mapInPandas(embed_partition, schema=out_schema)
    (
        result_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(CELLS_TABLE)
    )
    _elapsed_min = (_time.time() - _t0) / 60
    _n_rows = spark.table(CELLS_TABLE).count()
    print(
        f"Done in {_elapsed_min:.1f} min. Total rows in {CELLS_TABLE}: {_n_rows:,} "
        f"({_n_rows/max(_elapsed_min*60,0.1):.1f} cells/s overall — includes setup, "
        f"Census fetch, embedding, and Delta write across all workers)"
    )

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
