# Databricks notebook source
# MAGIC %md
# MAGIC # Register the OpenFold3 fold job with Genesis Workbench
# MAGIC Writes the `run_openfold3_job_id` settings row, grants the app service principal run permission
# MAGIC on the job + READ on the model Volume (so the UI can stream `ranked_0.pdb` into the viewer), and
# MAGIC registers the `openfold3` batch model. Mirrors the AlphaFold2 register job it replaces, minus the
# MAGIC ~2 TB database downloads (OpenFold3 needs only the ~2 GB checkpoint, cached by the fold job).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_core", "Schema")
dbutils.widgets.text("model_volume", "openfold3", "Model Volume")
dbutils.widgets.text("run_openfold3_job_id", "0", "OpenFold3 Fold Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "0", "SQL Warehouse Id")
dbutils.widgets.text("databricks_app_names", "genesis-workbench:mcp-genesis-workbench", "Databricks App Names (colon/comma-separated, UI + MCP)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_volume = dbutils.widgets.get("model_volume")
run_openfold3_job_id = dbutils.widgets.get("run_openfold3_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

spark.sql(f"""
MERGE INTO settings AS target
USING (SELECT 'run_openfold3_job_id' AS key, '{run_openfold3_job_id}' AS value, 'large_molecule' AS module) AS source
ON target.key = source.key AND target.module = source.module
WHEN MATCHED THEN UPDATE SET target.value = source.value
WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
""")

# COMMAND ----------

# Grant the app service principals run permission on the fold job + READ on the model Volume. The app
# SP reads /Volumes/{catalog}/{schema}/{model_volume}/results/{run_id}/ranked_0.pdb to stream the
# predicted structure into the React viewer.
import os
from genesis_workbench.workbench import (
    set_app_permissions_for_job,
    set_app_permissions_for_volume,
)

_app_names_raw = dbutils.widgets.get("databricks_app_names")
os.environ["DATABRICKS_APP_NAMES"] = ",".join(
    [n.strip() for n in _app_names_raw.replace(":", ",").split(",") if n.strip()]
)

set_app_permissions_for_job(job_id=run_openfold3_job_id, user_email=user_email)
set_app_permissions_for_volume(
    volume_full_name=f"{catalog}.{schema}.{model_volume}",
    write=False,
)

# COMMAND ----------

from genesis_workbench.models import register_batch_model

register_batch_model(
    model_name="openfold3",
    model_display_name="OpenFold3",
    model_description="AF3-style protein (and complex) structure prediction via OpenFold3 (OpenBind-0). MSA from the ColabFold server; reports pTM/ipTM/pLDDT and interface PAE.",
    model_category="large_molecule",
    module="large_molecule",
    job_id=run_openfold3_job_id,
    job_name="run_openfold3",
    cluster_type="GPU",
    added_by=user_email,
)
