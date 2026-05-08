# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Protein Sequences with ESM-2 — Serverless GPU (Multi-Node A10)
# MAGIC
# MAGIC SGC variant of `03_batch_embed_sequences_gpu.py`. Generates 1280-dimensional mean-pooled
# MAGIC embeddings for all sequences in the `sequence_db` table using ESM-2
# MAGIC (`facebook/esm2_t33_650M_UR50D`) and writes them to `sequence_embeddings`.
# MAGIC
# MAGIC Instead of a Spark `pandas_udf` on a classic GPU cluster, this notebook uses the
# MAGIC Serverless GPU Compute `@distributed(gpus=8, gpu_type="a10", remote=True)` decorator
# MAGIC to fan the work out across 8 single-A10 SGC nodes.
# MAGIC
# MAGIC The notebook itself acts as the orchestrator (runs on a single A10) and:
# MAGIC   1. Stages the source table into 8 parquet shards in a UC Volume (one per rank).
# MAGIC   2. Launches the remote `@distributed` function — each rank reads its shard,
# MAGIC      loads ESM-2 once on its local A10, embeds in batches, writes its output parquet.
# MAGIC   3. Consolidates the per-rank parquets into the `sequence_embeddings` Delta table.
# MAGIC
# MAGIC The embedding logic (tokenize → forward → mean pool last_hidden_state) matches the
# MAGIC UC-registered serving model and the existing `_gpu.py` notebook exactly.
# MAGIC
# MAGIC **Connect this notebook to Serverless GPU compute with A10 accelerator before running.**

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q torch==2.3.1 transformers==4.41.2 pyarrow==15.0.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")

# COMMAND ----------

# DBTITLE 1,Configuration
NUM_WORKERS = 8                            # 8 A10 single-GPU SGC nodes
MAX_SEQUENCES = 1_000_000                  # Embed 1M representative sequences for vector search
BATCH_SIZE = 32                            # Per-GPU batch size (matches _gpu.py Arrow batch size)
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"

SOURCE_TABLE = f"{catalog}.{schema}.sequence_db"
TARGET_TABLE = f"{catalog}.{schema}.sequence_embeddings"
SHARD_DIR = f"/Volumes/{catalog}/{schema}/{volume_name}/sgc_embedding_shards"
INPUT_DIR = f"{SHARD_DIR}/input"
OUTPUT_DIR = f"{SHARD_DIR}/output"

print(f"workers (gpus): {NUM_WORKERS}, batch size: {BATCH_SIZE}, "
      f"max sequences: {MAX_SEQUENCES:,}, model: {ESM2_MODEL}")
print(f"source: {SOURCE_TABLE}")
print(f"target: {TARGET_TABLE}")
print(f"shards: {SHARD_DIR}")

# COMMAND ----------

# DBTITLE 1,Skip-if-populated guard
skip_embedding = False
if spark.catalog.tableExists(TARGET_TABLE):
    existing_count = spark.table(TARGET_TABLE).count()
    if existing_count > 100:
        print(f"Embeddings table {TARGET_TABLE} already has {existing_count} rows, skipping.")
        skip_embedding = True

# COMMAND ----------

# DBTITLE 1,Stage source data into per-rank parquet shards
# Each remote @distributed worker has no Spark session — it reads its shard with pandas/pyarrow.
# We hash on seq_id so the assignment is deterministic across re-runs.
from pyspark.sql.functions import pmod, hash as spark_hash

if not skip_embedding:
    src = spark.table(SOURCE_TABLE).select("seq_id", "sequence").limit(MAX_SEQUENCES)
    total_rows = src.count()
    print(f"Sharding {total_rows:,} rows into {NUM_WORKERS} parquet partitions at {INPUT_DIR}")
    print(f"Estimated time: ~{total_rows / BATCH_SIZE * 0.2 / NUM_WORKERS / 60:.0f} minutes "
          f"with {NUM_WORKERS} A10 workers")

    sharded = src.withColumn("__shard__", pmod(spark_hash("seq_id"), NUM_WORKERS))
    (
        sharded
        .repartition("__shard__")
        .write
        .mode("overwrite")
        .partitionBy("__shard__")
        .parquet(INPUT_DIR)
    )
    print(f"Staged shards under {INPUT_DIR}/__shard__=<0..{NUM_WORKERS - 1}>")

# COMMAND ----------

# DBTITLE 1,Define the remote @distributed embedding function
# `@distributed(gpus=N, gpu_type=..., remote=True)` provisions N single-A10 SGC workers.
# `gpus` is the total across all nodes; SGC packs 1 GPU per A10 instance.
# Rank/world-size helpers come from `serverless_gpu.runtime`.
from serverless_gpu.launcher import distributed
from serverless_gpu import runtime as rt


@distributed(gpus=NUM_WORKERS, gpu_type="a10", remote=True)
def embed_shards():
    import os
    import glob
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModel

    rank = rt.get_global_rank()
    local_rank = rt.get_local_rank()

    # Each rank reads exactly one Spark partition directory: __shard__=<rank>
    shard_glob = f"{INPUT_DIR}/__shard__={rank}/*.parquet"
    files = sorted(glob.glob(shard_glob))
    if not files:
        print(f"[rank {rank}] no shard files matched {shard_glob}; nothing to do")
        return
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    print(f"[rank {rank} local {local_rank}] loaded {len(df):,} sequences from {len(files)} file(s)")

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModel.from_pretrained(ESM2_MODEL, torch_dtype=torch.float16).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[rank {rank}] ESM-2 (FP16) loaded on {torch.cuda.get_device_name(0)}")

    seq_ids: list = []
    embeddings: list = []
    n = len(df)
    for start in range(0, n, BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        tokens = tokenizer(
            batch["sequence"].tolist(),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            output = model(**tokens)
        # Mean pool per sequence, excluding padding (matches _gpu.py exactly)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embs = (summed / counts).cpu().float().tolist()
        seq_ids.extend(batch["seq_id"].tolist())
        embeddings.extend(embs)
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"[rank {rank}] processed {start + len(batch):,}/{n:,}")

    out_path = f"{OUTPUT_DIR}/embeddings_rank{rank:03d}.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame({"seq_id": seq_ids, "embedding": embeddings}).to_parquet(out_path, index=False)
    print(f"[rank {rank}] wrote {len(seq_ids):,} embeddings to {out_path}")


# COMMAND ----------

# DBTITLE 1,Launch distributed embedding across 8 A10 nodes
if not skip_embedding:
    embed_shards.distributed()
    print(f"All {NUM_WORKERS} ranks complete")

# COMMAND ----------

# DBTITLE 1,Consolidate per-rank output parquets into the Delta table
if not skip_embedding:
    embeddings_df = spark.read.parquet(f"{OUTPUT_DIR}/embeddings_rank*.parquet")
    embeddings_df.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
    print(f"Embeddings written to {TARGET_TABLE}")

# COMMAND ----------

# DBTITLE 1,Optional — clean up staged shards
# Uncomment to remove input + output parquet shards after a successful run.
# dbutils.fs.rm(SHARD_DIR, recurse=True)

# COMMAND ----------

# DBTITLE 1,Verify embeddings
result_df = spark.table(TARGET_TABLE)
print(f"Total embeddings: {result_df.count()}")

from pyspark.sql.functions import size
dim_check = result_df.select(size("embedding").alias("dim")).limit(1).collect()[0]["dim"]
print(f"Embedding dimension: {dim_check}")
assert dim_check == 1280, f"Expected 1280d embeddings, got {dim_check}d"

display(result_df.limit(5))
