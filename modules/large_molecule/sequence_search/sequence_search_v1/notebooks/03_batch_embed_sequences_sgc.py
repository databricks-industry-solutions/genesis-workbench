# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Protein Sequences with ESM-2 — Serverless GPU (single A10)
# MAGIC
# MAGIC Generates 1280-dimensional mean-pooled embeddings for sequences in the
# MAGIC `sequence_db` table using ESM-2 (`facebook/esm2_t33_650M_UR50D`) and writes
# MAGIC them to `sequence_embeddings`.
# MAGIC
# MAGIC Runs on the **single A10** the task is scheduled on (serverless GPU compute),
# MAGIC so it draws from the managed serverless-GPU pool rather than the account's EC2
# MAGIC GPU vCPU quota — which is what blocked the old classic 4×A10 `multi_gpu_cluster`
# MAGIC (`VcpuLimitExceeded`, limit 0).
# MAGIC
# MAGIC > A single A10 is the simplest and most reliably-provisioned serverless-GPU
# MAGIC > option (it is exactly what the register jobs use). A multi-node fan-out via
# MAGIC > `serverless_gpu`'s `@distributed(remote=True)` is possible (see `_ray.py` and
# MAGIC > this file's git history) and would be faster for the full corpus, but adds
# MAGIC > provisioning complexity and is best reserved for interactive SGC sessions.
# MAGIC > Here `max_sequences` bounds the corpus and embeddings are flushed to parquet
# MAGIC > in chunks, so driver memory stays bounded regardless of corpus size.
# MAGIC
# MAGIC The embedding logic (tokenize → forward → mask-weighted mean pool) matches the
# MAGIC UC-registered serving model and the `_gpu.py` / `05_..._sgc.py` notebooks.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# torch/CUDA are preinstalled on the serverless GPU AI runtime; add transformers.
# hf_transfer is required because the runtime sets HF_HUB_ENABLE_HF_TRANSFER=1.
# MAGIC %pip install -q transformers==4.41.2 pyarrow==15.0.2 hf_transfer==0.1.9
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
# max_sequences is a job parameter so the run can be tuned (or capped to a small
# subset for verification) without editing the notebook. Default is the production
# corpus size (1M representative sequences).
dbutils.widgets.text("max_sequences", "1000000", "Max sequences to embed")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")

# COMMAND ----------

# DBTITLE 1,Configuration
MAX_SEQUENCES = int(dbutils.widgets.get("max_sequences"))  # representative sequences to embed
BATCH_SIZE = 32                            # per-forward batch size (matches _gpu.py)
FLUSH_ROWS = 50_000                        # write a parquet part every N embeddings (bounds RAM)
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"

SOURCE_TABLE = f"{catalog}.{schema}.sequence_db"
TARGET_TABLE = f"{catalog}.{schema}.sequence_embeddings"
OUTPUT_DIR = f"/Volumes/{catalog}/{schema}/{volume_name}/sgc_embedding_output"

print(f"batch size: {BATCH_SIZE}, max sequences: {MAX_SEQUENCES:,}, model: {ESM2_MODEL}")
print(f"source: {SOURCE_TABLE}")
print(f"target: {TARGET_TABLE}")
print(f"output staging: {OUTPUT_DIR}")

# COMMAND ----------

# DBTITLE 1,Skip-if-populated guard
skip_embedding = False
if spark.catalog.tableExists(TARGET_TABLE):
    existing_count = spark.table(TARGET_TABLE).count()
    if existing_count > 100:
        print(f"Embeddings table {TARGET_TABLE} already has {existing_count} rows, skipping.")
        skip_embedding = True

# COMMAND ----------

# DBTITLE 1,Load the (capped) source sequences to the driver
if not skip_embedding:
    src_pdf = (
        spark.table(SOURCE_TABLE)
        .select("seq_id", "sequence")
        .limit(MAX_SEQUENCES)
        .toPandas()
    )
    total_rows = len(src_pdf)
    est_min = total_rows / BATCH_SIZE * 0.25 / 60
    print(f"Loaded {total_rows:,} sequences; est ~{est_min:.0f} min on one A10")

# COMMAND ----------

# DBTITLE 1,Embed on the local A10 (FP16), flushing parquet parts to bound memory
if not skip_embedding:
    import os
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Start from a clean staging dir so re-runs don't mix old parts in.
    dbutils.fs.rm(OUTPUT_DIR, recurse=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModel.from_pretrained(ESM2_MODEL, torch_dtype=torch.float16).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"ESM-2 (FP16) loaded on {torch.cuda.get_device_name(0)}")

    buf_ids: list = []
    buf_emb: list = []
    part = 0

    def _flush():
        global part, buf_ids, buf_emb
        if not buf_ids:
            return
        path = os.path.join(OUTPUT_DIR, f"embeddings_part{part:05d}.parquet")
        pd.DataFrame({"seq_id": buf_ids, "embedding": buf_emb}).to_parquet(path, index=False)
        print(f"  flushed {len(buf_ids):,} rows -> {os.path.basename(path)}")
        part += 1
        buf_ids, buf_emb = [], []

    n = total_rows
    for start in range(0, n, BATCH_SIZE):
        batch = src_pdf.iloc[start:start + BATCH_SIZE]
        tokens = tokenizer(
            batch["sequence"].tolist(),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            output = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embs = (summed / counts).cpu().float().tolist()
        buf_ids.extend(batch["seq_id"].tolist())
        buf_emb.extend(embs)
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"processed {start + len(batch):,}/{n:,}")
        if len(buf_ids) >= FLUSH_ROWS:
            _flush()
    _flush()
    print(f"Wrote {part} parquet part(s) under {OUTPUT_DIR}")

# COMMAND ----------

# DBTITLE 1,Consolidate parquet parts into the Delta table
if not skip_embedding:
    (
        spark.read.parquet(OUTPUT_DIR)
        .write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
    )
    print(f"Embeddings written to {TARGET_TABLE}")

# COMMAND ----------

# DBTITLE 1,Optional — clean up staged parquet parts
# Uncomment to remove the parquet staging dir after a successful run.
# dbutils.fs.rm(OUTPUT_DIR, recurse=True)

# COMMAND ----------

# DBTITLE 1,Verify embeddings
result_df = spark.table(TARGET_TABLE)
print(f"Total embeddings: {result_df.count()}")

from pyspark.sql.functions import size
dim_check = result_df.select(size("embedding").alias("dim")).limit(1).collect()[0]["dim"]
print(f"Embedding dimension: {dim_check}")
assert dim_check == 1280, f"Expected 1280d embeddings, got {dim_check}d"

display(result_df.limit(5))
