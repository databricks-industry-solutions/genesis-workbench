# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Protein Sequences with ESM-2
# MAGIC
# MAGIC Generates 1280-dimensional mean-pooled embeddings for all sequences in
# MAGIC the `sequence_db` table using ESM-2 (`facebook/esm2_t33_650M_UR50D`).
# MAGIC
# MAGIC Uses a `pandas_udf` (Iterator variant) for GPU-efficient batch inference —
# MAGIC the model loads once per worker and Arrow batch sizing controls GPU memory.
# MAGIC
# MAGIC The embedding logic (tokenize → forward → mean pool last_hidden_state)
# MAGIC matches the UC-registered serving model exactly.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q torch==2.3.1 transformers==4.41.2 databricks-sdk==0.50.0 databricks-sql-connector==4.0.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Configure Arrow batch size for GPU memory control
# g5.16xlarge: 1x A10G (24 GB VRAM), 256 GB RAM per node
# ESM2 FP16: ~1.3 GB VRAM. Batch of 32 × 1024 tokens padded: ~4-6 GB VRAM.
# Arrow maxRecordsPerBatch controls how many rows each UDF call receives.
# All 32 sequences are tokenized+forwarded together in one GPU call.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "32")

NUM_WORKERS = 4
MAX_SEQUENCES = 1_000_000  # Embed 1M representative sequences for vector search
NUM_PARTITIONS = NUM_WORKERS * 10  # 40 partitions → ~25K rows each, well distributed
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
print(f"Arrow batch size: 32, partitions: {NUM_PARTITIONS}, workers: {NUM_WORKERS}, "
      f"max sequences: {MAX_SEQUENCES:,}, model: {ESM2_MODEL}")

# COMMAND ----------

# DBTITLE 1,Define pandas_udf for ESM-2 embedding
import pandas as pd
from typing import Iterator
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType


@pandas_udf(ArrayType(FloatType()))
def embed_sequences(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """
    Iterator pandas_udf — loads ESM-2 once per worker in FP16, then
    processes each Arrow batch on GPU with batched tokenization/inference.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
    model = AutoModel.from_pretrained(ESM2_MODEL, torch_dtype=torch.float16).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"ESM-2 (FP16) loaded on {torch.cuda.get_device_name(0)}")

    for sequences in batches:
        seq_list = sequences.tolist()
        # Batched tokenization — all sequences in the Arrow batch at once
        tokens = tokenizer(
            seq_list, return_tensors="pt", truncation=True,
            max_length=1024, padding=True
        ).to("cuda")
        with torch.no_grad():
            output = model(**tokens)
        # Mean pool per sequence, excluding padding and BOS/EOS
        attention_mask = tokens["attention_mask"]
        hidden = output.last_hidden_state
        # Zero out padding positions, then mean over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = (summed / counts).cpu().float().tolist()
        yield pd.Series(embeddings)

# COMMAND ----------

# DBTITLE 1,Run batch embedding
source_table = f"{catalog}.{schema}.sequence_db"
target_table = f"{catalog}.{schema}.sequence_embeddings"

# Skip if embeddings table already has data
skip_embedding = False
if spark.catalog.tableExists(target_table):
    existing_count = spark.table(target_table).count()
    if existing_count > 100:
        print(f"Embeddings table {target_table} already has {existing_count} rows, skipping.")
        skip_embedding = True

if not skip_embedding:
    df = spark.table(source_table).limit(MAX_SEQUENCES)
    total_rows = df.count()
    print(f"Source table: {source_table}, embedding {total_rows:,} rows (limit: {MAX_SEQUENCES:,})")
    print(f"Generating embeddings with model: {ESM2_MODEL}")
    print(f"Estimated time: ~{total_rows / 32 * 0.2 / NUM_WORKERS / 60:.0f} minutes with {NUM_WORKERS} workers")

    # Repartition across workers — 40 partitions (~25K rows each)
    # Model loads once per worker (Iterator UDF), Arrow sends 32 rows per batched GPU call.
    embeddings_df = (
        df.repartition(NUM_PARTITIONS)
          .select("seq_id", embed_sequences("sequence").alias("embedding"))
    )

    embeddings_df.write.format("delta").mode("overwrite").saveAsTable(target_table)
    print(f"Embeddings written to {target_table}")

# COMMAND ----------

# DBTITLE 1,Verify embeddings
result_df = spark.table(target_table)
print(f"Total embeddings: {result_df.count()}")

# Check embedding dimension
from pyspark.sql.functions import size
dim_check = result_df.select(size("embedding").alias("dim")).limit(1).collect()[0]["dim"]
print(f"Embedding dimension: {dim_check}")
assert dim_check == 1280, f"Expected 1280d embeddings, got {dim_check}d"

display(result_df.limit(5))
