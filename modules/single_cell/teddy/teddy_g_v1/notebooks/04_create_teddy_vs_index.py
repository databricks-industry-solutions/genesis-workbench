# Databricks notebook source
# MAGIC %md
# MAGIC ## Create TEDDY Cell Vector Search Index
# MAGIC
# MAGIC Mirror of `modules/single_cell/scimilarity/.../06c_create_cell_vector_index.py` but
# MAGIC for the TEDDY-embedded reference. Creates `gwb_teddy_vs_endpoint` and a Delta Sync
# MAGIC index `teddy_cell_index` over the `teddy_cells` table populated by notebook 03.
# MAGIC
# MAGIC The destroy-preservation rule applies — these resources are preserved across
# MAGIC GWB destroys per the project policy (no auto-deletion of `gwb_*_vs_endpoint` /
# MAGIC `*_index`).

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q databricks-sdk==0.50.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_yyang_genesis_workbench", "Schema")
dbutils.widgets.text("teddy_model_size", "400M", "TEDDY-G variant (sets embedding dim)")
dbutils.widgets.text("target_n_cells", "2000000", "Target reference cell count (for idempotency)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_size = dbutils.widgets.get("teddy_model_size")
target_n_cells = int(dbutils.widgets.get("target_n_cells"))

# d_model per variant from the TEDDY-G configs (config.json)
EMB_DIM = {"70M": 512, "160M": 768, "400M": 1024}.get(model_size)
assert EMB_DIM, f"Unknown TEDDY-G variant: {model_size}"
print(f"Building VS index for TEDDY-G {model_size} (d_model={EMB_DIM})")

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
from databricks.sdk.errors import NotFound, BadRequest
import time

w = WorkspaceClient()
VS_ENDPOINT_NAME = "gwb_teddy_vs_endpoint"

try:
    endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists (status: {endpoint.endpoint_status})")
except Exception:
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'…")
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type=EndpointType.STANDARD)

for _ in range(60):
    endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    if endpoint.endpoint_status and endpoint.endpoint_status.state and endpoint.endpoint_status.state.value == "ONLINE":
        print(f"Endpoint '{VS_ENDPOINT_NAME}' is ONLINE")
        break
    print(f"  Status: {endpoint.endpoint_status}. Waiting…")
    time.sleep(30)
else:
    print("WARNING: endpoint not online within 30 minutes; index creation may still succeed.")

# COMMAND ----------

# DBTITLE 1,Create Delta Sync index over teddy_cells
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    VectorIndexType,
    PipelineType,
)

source_table = f"{catalog}.{schema}.teddy_cells"
index_name = f"{catalog}.{schema}.teddy_cell_index"

# ── Pre-flight idempotency check ──────────────────────────────────────────
# If teddy_cells is already populated to >=95% of target at the right dim AND
# teddy_cell_index already exists at the matching dim, this notebook is a
# no-op. Mirrors notebook 03's idempotency — together they guarantee that
# re-running `deploy.sh teddy` on a fully-built install does nothing.
def _existing_table_state():
    if not spark.catalog.tableExists(source_table):
        return (0, None)
    n = spark.table(source_table).count()
    if n == 0:
        return (0, None)
    d = spark.table(source_table).selectExpr("size(embedding) as d").limit(1).collect()[0]["d"]
    return (n, d)

def _existing_index_dim(idx):
    try:
        spec = getattr(idx, "delta_sync_index_spec", None)
        if spec is None: return None
        cols = getattr(spec, "embedding_vector_columns", None) or []
        if cols:
            return int(getattr(cols[0], "embedding_dimension", 0)) or None
    except Exception:
        return None
    return None

tbl_rows, tbl_dim = _existing_table_state()
print(f"teddy_cells: {tbl_rows:,} rows, dim={tbl_dim} (want >= {int(target_n_cells*0.95):,} rows, dim={EMB_DIM})")

try:
    _existing_for_skip = w.vector_search_indexes.get_index(index_name)
except NotFound:
    _existing_for_skip = None
except Exception as e:
    print(f"  get_index transient: {e}")
    _existing_for_skip = None

if (
    tbl_rows >= int(target_n_cells * 0.95)
    and tbl_dim == EMB_DIM
    and _existing_for_skip is not None
    and _existing_index_dim(_existing_for_skip) == EMB_DIM
):
    print(
        f"✅ Data + index already in place for {model_size} "
        f"(table dim={tbl_dim}, index dim={_existing_index_dim(_existing_for_skip)}). "
        f"Skipping create/sync — true no-op on re-deploy."
    )
    dbutils.notebook.exit("skipped: data and index already present")

# ──────────────────────────────────────────────────────────────────────────

spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

existing = None
for attempt in range(5):
    try:
        existing = w.vector_search_indexes.get_index(index_name)
        break
    except NotFound:
        existing = None
        break
    except Exception as e:
        print(f"  get_index transient error (attempt {attempt+1}/5): {e}")
        time.sleep(15)
else:
    raise RuntimeError(f"Could not determine whether '{index_name}' exists after 5 attempts.")

if existing is not None:
    existing_dim = _existing_index_dim(existing)
    if existing_dim is not None and existing_dim != EMB_DIM:
        # Variant switch (e.g. 70M dim=512 → 400M dim=1024). Sync can't change
        # the index's declared dim — we MUST drop and recreate. Destroy-
        # preservation policy applies only to `databricks bundle destroy`,
        # not to deliberate variant-switch deploys.
        print(
            f"Index '{index_name}' exists at dim={existing_dim}, "
            f"but {model_size} requires dim={EMB_DIM} — dropping and recreating."
        )
        w.vector_search_indexes.delete_index(index_name=index_name)
        # Brief wait for the delete to propagate before create_index.
        for _ in range(12):
            try:
                w.vector_search_indexes.get_index(index_name)
            except NotFound:
                break
            time.sleep(5)
        existing = None
    else:
        print(f"Index '{index_name}' exists at dim={existing_dim or EMB_DIM} — triggering sync.")
        try:
            w.vector_search_indexes.sync_index(index_name=index_name)
        except BadRequest as e:
            print(f"Sync not triggered (pipeline busy): {e}")

if existing is None:
    print(f"Creating Delta Sync index '{index_name}'…")
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="cell_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=EMB_DIM),
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=["cell_id"],
        ),
    )

# COMMAND ----------

# DBTITLE 1,Wait for index to be READY
for _ in range(240):
    try:
        status = w.vector_search_indexes.get_index(index_name).status
    except Exception as e:
        print(f"  get_index transient: {e}")
        time.sleep(30); continue
    if status and status.ready:
        print(f"Index '{index_name}' is READY")
        break
    print(f"  Status: {status}. Waiting…")
    time.sleep(30)
else:
    print("WARNING: index not ready in 2 hours; may still be syncing.")

# COMMAND ----------

# DBTITLE 1,Smoke test query
print("Running test query…")
sample_emb = spark.table(source_table).limit(1).collect()[0]["embedding"]
res = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=["cell_id"],
    query_vector=sample_emb,
    num_results=5,
)
for row in (res.result.data_array or []):
    print(f"  {row}")
