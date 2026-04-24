# Databricks notebook source
# MAGIC %md
# MAGIC ## Create SCimilarity Cell Vector Search Index
# MAGIC
# MAGIC Creates a Databricks Vector Search endpoint and a Delta Sync index over the
# MAGIC `scimilarity_cells` table (built in notebook 06b) for fast ANN similarity
# MAGIC search against the ~23M-cell reference corpus.
# MAGIC
# MAGIC Mirrors the protein sequence search pattern in
# MAGIC `modules/protein_studies/sequence_search/sequence_search_v1/notebooks/04_create_vector_index.py`.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q databricks-sdk==0.50.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "<catalog_name>", "Catalog")
dbutils.widgets.text("schema", "<schema_name>", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
import time

w = WorkspaceClient()

VS_ENDPOINT_NAME = "gwb_scimilarity_vs_endpoint"

try:
    endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Vector Search endpoint '{VS_ENDPOINT_NAME}' already exists (status: {endpoint.endpoint_status})")
except Exception:
    print(f"Creating Vector Search endpoint '{VS_ENDPOINT_NAME}'...")
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type=EndpointType.STANDARD)
    print("Endpoint creation initiated. Waiting for it to become ready...")

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

# DBTITLE 1,Create Delta Sync index over scimilarity_cells
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    VectorIndexType,
    PipelineType,
)

source_table = f"{catalog}.{schema}.scimilarity_cells"
index_name = f"{catalog}.{schema}.scimilarity_cell_index"

# CDF is enabled in 06b, but re-assert to be safe in case the table was rebuilt.
spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

try:
    existing_index = w.vector_search_indexes.get_index(index_name)
    print(f"Index '{index_name}' already exists. Triggering sync...")
    w.vector_search_indexes.sync_index(index_name=index_name)
    print("Sync triggered.")
except Exception:
    print(f"Creating Delta Sync index '{index_name}'...")
    print(f"Source table: {source_table}")

    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="cell_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=128),
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=["cell_id"],
        ),
    )
    print(f"Index '{index_name}' created. Initial sync will begin automatically.")

# COMMAND ----------

# DBTITLE 1,Wait for index to be ready
# First sync over ~23M rows can take a while; give it a generous window.
for _ in range(240):
    index_status = w.vector_search_indexes.get_index(index_name)
    status = index_status.status
    if status and status.ready:
        print(f"Index '{index_name}' is READY")
        break
    print(f"  Index status: {status}. Waiting...")
    time.sleep(30)
else:
    print("WARNING: Index did not become ready within 2 hours. It may still be syncing.")

# COMMAND ----------

# DBTITLE 1,Test query
print("Running test query to verify index...")
test_embedding = spark.table(source_table).limit(1).collect()[0]["embedding"]

results = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=["cell_id"],
    query_vector=test_embedding,
    num_results=5,
)

print("Test query results:")
for row in results.result.data_array:
    print(f"  {row}")
