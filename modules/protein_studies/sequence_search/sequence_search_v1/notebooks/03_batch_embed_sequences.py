# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Protein Sequences with ESM-2
# MAGIC
# MAGIC Generates 1280-dimensional mean-pooled embeddings for all sequences in
# MAGIC the `sequence_db` table using `facebook/esm2_t33_650M_UR50D`.
# MAGIC
# MAGIC Uses `predict_batch_udf` for GPU-efficient batch inference — the model is
# MAGIC loaded once per worker and reused across all batches.
# MAGIC
# MAGIC **Important**: The embedding logic here (model tag + mean-pooling) must match
# MAGIC the ESM2 Embeddings serving endpoint exactly to ensure vector space consistency
# MAGIC between the indexed embeddings and query-time embeddings.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q torch==2.3.1 transformers==4.41.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Define batch embedding UDF
import numpy as np
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.functions import predict_batch_udf

# Must match the model used in the ESM2 Embeddings serving endpoint
MODEL_TAG = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_LEN = 1022  # ESM-2 max tokens (1024 minus BOS/EOS)


def make_embed_fn():
    """Factory that loads ESM-2 once per worker and returns an embed function."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TAG)
    model = AutoModel.from_pretrained(MODEL_TAG).to(device).eval()
    print(f"ESM-2 ({MODEL_TAG}) loaded on {device}")

    def embed(sequences: np.ndarray) -> list:
        """Embed protein sequences, returning 1280d mean-pooled vectors."""
        results = []
        for seq in sequences.tolist():
            # Truncate to max length
            seq = seq[:MAX_SEQ_LEN]
            tokens = tokenizer(
                seq, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)
            with torch.no_grad():
                output = model(**tokens)
            # Mean pool over sequence positions, excluding BOS/EOS tokens
            embedding = output.last_hidden_state[0, 1:-1].mean(dim=0).cpu().tolist()
            results.append(embedding)
            torch.cuda.empty_cache()
        return results

    return embed


embed_udf = predict_batch_udf(
    make_embed_fn,
    return_type=ArrayType(FloatType()),
    batch_size=8,
)

# COMMAND ----------

# DBTITLE 1,Run batch embedding
source_table = f"{catalog}.{schema}.sequence_db"
target_table = f"{catalog}.{schema}.sequence_embeddings"

df = spark.table(source_table)

print(f"Source table: {source_table} ({df.count()} rows)")
print(f"Generating embeddings with {MODEL_TAG}...")

embeddings_df = df.select("seq_id", embed_udf("sequence").alias("embedding"))

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
