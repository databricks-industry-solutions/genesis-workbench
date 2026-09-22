# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "genesis_schema", "Schema")
dbutils.widgets.text("run_parabricks_job_id", "1234", "Parabricks Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("databricks_app_names", "genesis-workbench:mcp-genesis-workbench", "Databricks App Names (colon/comma-separated, UI + MCP)")

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")

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

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")
run_parabricks_job_id = dbutils.widgets.get("run_parabricks_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Parabricks Job ID: {run_parabricks_job_id}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Persist the orchestrator job id so core/grant_app_permissions.py (which queries
# key LIKE '%_job_id') keeps the app SP's CAN_MANAGE_RUN grant in sync on redeploys.
query = f"""
    MERGE INTO settings AS target
    USING (SELECT 'run_parabricks_job_id' AS key, '{run_parabricks_job_id}' AS value, 'genomics' AS module) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
"""
spark.sql(query)

# COMMAND ----------

# Grant the app SP CAN_MANAGE_RUN so the app can launch this job immediately after deploy.
from genesis_workbench.workbench import set_app_permissions_for_job
import os
_app_names_raw = dbutils.widgets.get("databricks_app_names")
os.environ["DATABRICKS_APP_NAMES"] = ",".join([n.strip() for n in _app_names_raw.replace(":", ",").split(",") if n.strip()])  # UI + MCP

set_app_permissions_for_job(job_id=run_parabricks_job_id, user_email=user_email)

# COMMAND ----------

# Register Parabricks as a batch model so it appears in the Deployed Models tab and the UI
# can launch it on demand (jobs run-now on run_parabricks_job_id).
from genesis_workbench.models import register_batch_model

register_batch_model(
    model_name="parabricks",
    model_display_name="NVIDIA Parabricks",
    model_description="GPU-accelerated germline variant calling (pbrun fq2bam + deepvariant) on serverless GPU",
    model_category="genomics",
    module="genomics",
    job_id=run_parabricks_job_id,
    job_name="run_parabricks",
    cluster_type="GPU",
    added_by=user_email,
)

print("Genomics Parabricks module initialization complete")
