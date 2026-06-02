# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Protein Sequences with ESM-2 — Serverless GPU + Ray Data
# MAGIC
# MAGIC Ray Data variant of `03_batch_embed_sequences_sgc.py`. Uses
# MAGIC `serverless_gpu.ray.ray_launch` to bring up a Ray cluster across 8 A10 SGC nodes,
# MAGIC then `ray.data.read_parquet → map_batches(EmbedActor) → write_parquet` to
# MAGIC distribute work without manual hash sharding.
# MAGIC
# MAGIC Compared to `_sgc.py`:
# MAGIC   - No `__shard__` column or partition-by-rank write — Ray Data balances batches dynamically.
# MAGIC   - **End state is a managed UC Delta table** (`{catalog}.{schema}.sequence_embeddings`).
# MAGIC
# MAGIC Data path (UC managed tables can't be read or written by non-Spark clients on
# MAGIC the SGC remote workers — UC vends storage credentials only to Spark. UC Volumes
# MAGIC are FUSE-accessible from any client, so we use them as the bridge for both
# MAGIC ends of the Ray run):
# MAGIC   1. Orchestrator: Spark reads `sequence_db` (uses UC creds) and writes the
# MAGIC      `seq_id` + `sequence` columns as parquet to a UC Volume input stage.
# MAGIC   2. `@ray_launch`: Ray reads the input stage parquet, embeds across 8 A10s,
# MAGIC      and **`write_parquet`** to a UC Volume output stage. Each worker writes its
# MAGIC      own block in parallel — no driver-side accumulation, so memory stays bounded.
# MAGIC   3. Orchestrator: Spark CTAS promotes the parquet output into a **managed**
# MAGIC      UC Delta table; both stage paths get cleaned up.
# MAGIC
# MAGIC Embedding math (tokenize → forward → attention-mask-weighted mean pool) is identical
# MAGIC to `_sgc.py` and `_gpu.py`. Same model, same FP16, same max_length=1024, same batch=32.
# MAGIC
# MAGIC **Connect this notebook to Serverless GPU compute with A10 accelerator before running.**

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q torch==2.3.1 transformers==4.41.2 pyarrow==15.0.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")

# COMMAND ----------

# DBTITLE 1,Configuration
NUM_WORKERS = 8                            # 8 A10 single-GPU SGC nodes
MAX_SEQUENCES = 1_000_000
BATCH_SIZE = 32
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"

SOURCE_TABLE = f"{catalog}.{schema}.sequence_db"
TARGET_TABLE = f"{catalog}.{schema}.sequence_embeddings"
# Stage paths on a UC Volume — both ends of the Ray run use these because UC
# Volumes are FUSE-accessible from non-Spark clients (Ray, pyarrow).
# Spark fills the input stage; Ray writes parquet to the output stage in
# parallel from each worker; the orchestrator promotes output to a managed
# UC Delta table via CTAS and removes both stages.
INPUT_STAGE_DIR = f"/Volumes/{catalog}/{schema}/{volume_name}/ray_input_stage"
OUTPUT_STAGE_PARQUET = f"/Volumes/{catalog}/{schema}/{volume_name}/ray_output_stage_parquet"

print(f"workers (gpus): {NUM_WORKERS}, batch size: {BATCH_SIZE}, "
      f"max sequences: {MAX_SEQUENCES if MAX_SEQUENCES else 'no limit'}, model: {ESM2_MODEL}")
print(f"source:       {SOURCE_TABLE}")
print(f"target table: {TARGET_TABLE} (managed UC Delta)")
print(f"input stage:  {INPUT_STAGE_DIR}")
print(f"output stage: {OUTPUT_STAGE_PARQUET}")

# COMMAND ----------

# DBTITLE 1,Skip-if-populated guard
skip_embedding = False
if spark.catalog.tableExists(TARGET_TABLE):
    existing_count = spark.table(TARGET_TABLE).count()
    if existing_count > 100:
        print(f"Embeddings table {TARGET_TABLE} already has {existing_count} rows, skipping.")
        skip_embedding = True

# COMMAND ----------

# DBTITLE 1,Stage source columns to a UC Volume as parquet
# Ray on SGC remote workers can't read UC managed table storage directly (UC
# vends storage creds only to Spark on Databricks). Spark CAN read it, and a UC
# Volume is FUSE-accessible from Ray, so we use Spark to materialize just the
# columns Ray needs into the Volume.
if not skip_embedding:
    src = spark.table(SOURCE_TABLE).select("seq_id", "sequence").limit(MAX_SEQUENCES)
    total_rows = src.count()
    print(f"Staging {total_rows:,} rows ({SOURCE_TABLE} → {INPUT_STAGE_DIR})")
    print(f"Estimated time: ~{total_rows / BATCH_SIZE * 0.2 / NUM_WORKERS / 60:.0f} minutes "
          f"with {NUM_WORKERS} A10 workers (embedding step only)")
    src.write.mode("overwrite").parquet(INPUT_STAGE_DIR)

# COMMAND ----------

# DBTITLE 1,Define the Ray-driven embedding job
# `ray_launch` provisions N single-A10 SGC workers AND auto-bootstraps a Ray cluster
# across them (rank-0 starts ray head, others join). The decorated function runs
# on rank-0 as the Ray driver; Ray Data distributes batches to all GPUs.
from serverless_gpu.ray import ray_launch


@ray_launch(gpus=NUM_WORKERS, gpu_type="a10", remote=True)
def embed_with_ray():
    import numpy as np
    import ray
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    class ESM2Embedder:
        """Stateful Ray Data actor — loads ESM-2 once per actor (i.e. once per GPU)."""

        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
            self.model = AutoModel.from_pretrained(
                ESM2_MODEL, torch_dtype=torch.float16
            ).cuda().eval()
            torch.backends.cuda.matmul.allow_tf32 = True
            print(f"ESM-2 (FP16) loaded on {torch.cuda.get_device_name(0)}")

        def __call__(self, batch):
            seqs = list(batch["sequence"])
            tokens = self.tokenizer(
                seqs,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            ).to("cuda")
            with torch.no_grad():
                out = self.model(**tokens)
            # Attention-mask-weighted mean pool (matches _gpu.py / _sgc.py exactly)
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            embs = (summed / counts).cpu().float().numpy()
            return {
                "seq_id": np.asarray(batch["seq_id"]),
                "embedding": embs.tolist(),
            }

    # Read the staged parquet directly from the UC Volume (FUSE path-accessible
    # from Ray workers — no S3 credentials needed).
    ds = ray.data.read_parquet(INPUT_STAGE_DIR)
    print(f"Dataset rows: {ds.count()}")

    result = ds.map_batches(
        ESM2Embedder,
        batch_size=BATCH_SIZE,
        num_gpus=1,                # one GPU per actor
        concurrency=NUM_WORKERS,   # one actor per GPU → one per node
        batch_format="numpy",
    )

    # Distributed write to parquet: each Ray worker writes its own block to the
    # UC Volume directly. No driver-side accumulation = no driver memory bottleneck.
    # The orchestrator does a single CTAS afterwards to promote the parquet
    # directory into a managed UC Delta table.
    print(f"Distributed write_parquet to {OUTPUT_STAGE_PARQUET}")
    result.write_parquet(OUTPUT_STAGE_PARQUET)
    print(f"Ray write_parquet complete")


# COMMAND ----------

# DBTITLE 1,Launch Ray-driven embedding across 8 A10 nodes
if not skip_embedding:
    embed_with_ray.distributed()
    print("Ray job complete")

# COMMAND ----------

# DBTITLE 1,Promote the parquet output into a managed UC Delta table
# Ray wrote parquet files at OUTPUT_STAGE_PARQUET. CTAS reads them via Spark
# and materializes a fresh **managed** UC Delta table — UC owns the storage
# and lifecycle; the staging Volume paths are no longer needed.
if not skip_embedding:
    spark.sql(f"""
        CREATE OR REPLACE TABLE {TARGET_TABLE}
        USING DELTA
        AS SELECT * FROM parquet.`{OUTPUT_STAGE_PARQUET}`
    """)
    print(f"Managed UC Delta table {TARGET_TABLE} created from parquet output")

# COMMAND ----------

# DBTITLE 1,Clean up the input + output stage paths
# The managed UC table now owns its own copy of the data; both stages are disposable.
if not skip_embedding:
    dbutils.fs.rm(INPUT_STAGE_DIR, recurse=True)
    dbutils.fs.rm(OUTPUT_STAGE_PARQUET, recurse=True)
    print(f"Removed input stage  {INPUT_STAGE_DIR}")
    print(f"Removed output stage {OUTPUT_STAGE_PARQUET}")

# COMMAND ----------

# DBTITLE 1,Verify embeddings
result_df = spark.table(TARGET_TABLE)
print(f"Total embeddings: {result_df.count()}")

from pyspark.sql.functions import size
dim_check = result_df.select(size("embedding").alias("dim")).limit(1).collect()[0]["dim"]
print(f"Embedding dimension: {dim_check}")
assert dim_check == 1280, f"Expected 1280d embeddings, got {dim_check}d"

display(result_df.limit(5))
