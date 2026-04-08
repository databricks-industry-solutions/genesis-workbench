# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Protein Sequences with ESM-2
# MAGIC
# MAGIC Generates 1280-dimensional mean-pooled embeddings for all sequences in
# MAGIC the `sequence_db` table using the ESM-2 model registered in Unity Catalog.
# MAGIC
# MAGIC Uses `predict_batch_udf` for GPU-efficient batch inference — the model is
# MAGIC loaded once per worker and reused across all batches.
# MAGIC
# MAGIC The UC-registered model is the same artifact deployed to the serving endpoint,
# MAGIC ensuring exact vector space consistency between indexed and query-time embeddings.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q torch==2.3.1 transformers==4.41.2 mlflow>=2.15
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Load UC model URI
import mlflow
from genesis_workbench.models import get_latest_model_version

model_uc_name = f"{catalog}.{schema}.esm2_embeddings"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(f"Using UC model: {model_uri}")

# COMMAND ----------

# DBTITLE 1,Define batch embedding UDF
import numpy as np
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.functions import predict_batch_udf

# Capture for use inside the UDF closure
_model_uri = model_uri


def make_embed_fn():
    """Factory that loads the UC-registered ESM-2 model once per worker."""
    import mlflow

    model = mlflow.pyfunc.load_model(_model_uri)
    print(f"Loaded ESM-2 from UC: {_model_uri}")

    def embed(sequences: np.ndarray) -> list:
        """Embed protein sequences using the UC model's predict method."""
        return model.predict(sequences.tolist())

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
print(f"Generating embeddings with UC model: {model_uri}")

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
