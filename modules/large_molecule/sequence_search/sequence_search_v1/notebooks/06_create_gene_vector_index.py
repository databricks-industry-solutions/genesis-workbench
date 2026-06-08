# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Human Protein Vector Search Index
# MAGIC
# MAGIC Delta Sync index over `gene_sequence_embeddings` (human SwissProt) on the
# MAGIC same Vector Search endpoint as the UniRef index, with the same 1280-dim
# MAGIC ESM-2 embedding space. The app queries this index alongside
# MAGIC `sequence_embedding_index` so protein search returns human hits too.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q databricks-sdk==0.50.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Reuse the Vector Search endpoint (created by 04)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
import time

w = WorkspaceClient()

VS_ENDPOINT_NAME = "gwb_sequence_search_vs_endpoint"

try:
    endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Vector Search endpoint '{VS_ENDPOINT_NAME}' already exists (status: {endpoint.endpoint_status})")
except Exception:
    print(f"Creating Vector Search endpoint '{VS_ENDPOINT_NAME}'...")
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type=EndpointType.STANDARD)

for _ in range(60):
    endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    if endpoint.endpoint_status and endpoint.endpoint_status.state and endpoint.endpoint_status.state.value == "ONLINE":
        print(f"Endpoint '{VS_ENDPOINT_NAME}' is ONLINE")
        break
    print(f"  Status: {endpoint.endpoint_status}. Waiting...")
    time.sleep(30)
else:
    print("WARNING: Endpoint did not come online within 30 minutes. Index creation may still succeed.")

# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed and create the human index
spark.sql(
    f"ALTER TABLE {catalog}.{schema}.gene_sequence_embeddings "
    f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print("Change Data Feed enabled on gene_sequence_embeddings table")

from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    VectorIndexType,
    PipelineType,
)

source_table = f"{catalog}.{schema}.gene_sequence_embeddings"
index_name = f"{catalog}.{schema}.gene_sequence_embedding_index"

try:
    existing_index = w.vector_search_indexes.get_index(index_name)
    print(f"Index '{index_name}' already exists. Triggering sync...")
    w.vector_search_indexes.sync_index(index_name=index_name)
    print("Sync triggered.")
except Exception:
    print(f"Creating Delta Sync index '{index_name}' over {source_table}...")
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="seq_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=1280),
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=["seq_id"],
        ),
    )
    print(f"Index '{index_name}' created. Initial sync will begin automatically.")

# COMMAND ----------

# DBTITLE 1,Wait for index to be ready
for _ in range(120):
    index_status = w.vector_search_indexes.get_index(index_name)
    status = index_status.status
    if status and status.ready:
        print(f"Index '{index_name}' is READY")
        break
    print(f"  Index status: {status}. Waiting...")
    time.sleep(30)
else:
    print("WARNING: Index did not become ready within 60 minutes. It may still be syncing.")

# COMMAND ----------

# DBTITLE 1,Test query
print("Running test query to verify index...")
test_embedding = spark.table(source_table).limit(1).collect()[0]["embedding"]
results = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=["seq_id"],
    query_vector=test_embedding,
    num_results=5,
)
print("Test query results:")
for row in results.result.data_array:
    print(f"  {row}")
