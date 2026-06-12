# Databricks notebook source
# MAGIC %md
# MAGIC ### Publish the node catalog
# MAGIC Writes the built-in Vortex node catalog (`genesis_workbench.builtin_nodes.CURATED_NODES`)
# MAGIC to the `node_catalog` Delta table — the single runtime source of truth read by
# MAGIC the app (Vortex), the executor, and the MCP server. Run at deploy time; it
# MAGIC full-overwrites the `builtin` rows (idempotent) and leaves other sources
# MAGIC (future `mcp:<server>` external tools) untouched.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(gwb_library_path)
if not gwb_library_path:
    raise RuntimeError(
        f"genesis_workbench wheel not found in /Volumes/{catalog}/{schema}/libraries "
        "— a deploy may have been mid-flight. Re-run this job."
    )

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

from genesis_workbench.workbench import initialize
from genesis_workbench.capabilities import publish_node_catalog

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema,
           sql_warehouse_id=sql_warehouse_id, token=databricks_token)

n = publish_node_catalog(catalog=catalog, schema=schema)
print(f"✅ Published {n} built-in nodes to {catalog}.{schema}.node_catalog")
