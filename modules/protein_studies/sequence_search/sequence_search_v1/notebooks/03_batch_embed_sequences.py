# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Embed Protein Sequences with ESM-2
# MAGIC
# MAGIC Generates 1280-dimensional mean-pooled embeddings for sequences in
# MAGIC the `sequence_db` table using the deployed ESM-2 serving endpoint via `ai_query()`.
# MAGIC
# MAGIC This approach uses the already-deployed model serving endpoint — Databricks
# MAGIC handles batching, scaling, and GPU management automatically. No GPU cluster needed.
# MAGIC
# MAGIC For the GPU-based Spark approach, see `03_batch_embed_sequences_gpu.py`.

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Configuration
MAX_SEQUENCES = 1_000_000

source_table = f"{catalog}.{schema}.sequence_db"
target_table = f"{catalog}.{schema}.sequence_embeddings_aiq"

# Resolve endpoint name from model_deployments table
endpoint_rows = spark.sql(f"""
    SELECT model_endpoint_name FROM {catalog}.{schema}.model_deployments
    WHERE (deploy_model_uc_name LIKE '%esm2_embeddings%' OR model_endpoint_name LIKE '%esm2_embeddings%')
      AND is_active = true
    LIMIT 1
""").collect()

if endpoint_rows:
    ESM2_ENDPOINT = endpoint_rows[0]["model_endpoint_name"]
else:
    # Fallback: list all active endpoints and find the one with esm2
    all_endpoints = spark.sql(f"""
        SELECT model_endpoint_name FROM {catalog}.{schema}.model_deployments
        WHERE is_active = true
    """).collect()
    print(f"Available endpoints: {[r['model_endpoint_name'] for r in all_endpoints]}")
    esm2_matches = [r["model_endpoint_name"] for r in all_endpoints if "esm2" in r["model_endpoint_name"].lower()]
    if esm2_matches:
        ESM2_ENDPOINT = esm2_matches[0]
    else:
        raise RuntimeError(f"No ESM2 embeddings endpoint found in model_deployments table. Available: {[r['model_endpoint_name'] for r in all_endpoints]}")

print(f"ESM2 endpoint: {ESM2_ENDPOINT}")
print(f"Source: {source_table}")
print(f"Target: {target_table}")
print(f"Max sequences: {MAX_SEQUENCES:,}")

# COMMAND ----------

# DBTITLE 1,Skip if already embedded
skip_embedding = False
if spark.catalog.tableExists(target_table):
    existing_count = spark.table(target_table).count()
    if existing_count > 100:
        print(f"Embeddings table {target_table} already has {existing_count} rows, skipping.")
        skip_embedding = True

# COMMAND ----------

# DBTITLE 1,Generate embeddings via ai_query
if not skip_embedding:
    total_rows = spark.table(source_table).count()
    embed_count = min(total_rows, MAX_SEQUENCES)
    print(f"Source table has {total_rows:,} rows, embedding {embed_count:,}")

    spark.sql(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT
            seq_id,
            ai_query(
                '{ESM2_ENDPOINT}',
                sequence,
                'ARRAY<FLOAT>'
            ) AS embedding
        FROM {source_table}
        LIMIT {MAX_SEQUENCES}
    """)

    result_count = spark.table(target_table).count()
    print(f"Embeddings written to {target_table}: {result_count:,} rows")

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
