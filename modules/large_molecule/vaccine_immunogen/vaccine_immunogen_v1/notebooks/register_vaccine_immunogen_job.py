# Databricks notebook source
# MAGIC %md
# MAGIC ### Register Vaccine Immunogen Orchestrator Job
# MAGIC
# MAGIC Runs once per deploy (after the orchestrator job exists):
# MAGIC 1. Persist `run_vaccine_immunogen_job_id` to the `settings` table so
# MAGIC    `grant_app_permissions.py` (which queries `key LIKE '%_job_id'`) keeps the
# MAGIC    app SP's `CAN_MANAGE_RUN` grant in sync on subsequent redeploys.
# MAGIC 2. Grant the app SP `CAN_MANAGE_RUN` on the orchestrator job now (so the app
# MAGIC    can dispatch immediately) and `WRITE` on the cache volume (the app uploads
# MAGIC    each run's epitope-motif PDB there before dispatching).
# MAGIC 3. Register Vaccine Immunogen Design in `batch_models` (the orchestrator is the
# MAGIC    user-launchable workflow).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("run_vaccine_immunogen_job_id", "", "Orchestrator Job ID")
dbutils.widgets.text("user_email", "a@b.com", "User email")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App Name")
dbutils.widgets.text("databricks_app_names", "genesis-workbench:mcp-genesis-workbench", "Databricks App Names (colon/comma-separated, UI + MCP)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(f"GWB library: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
g = dbutils.widgets.get
catalog, schema = g("catalog"), g("schema")
run_vaccine_immunogen_job_id = g("run_vaccine_immunogen_job_id")
user_email, sql_warehouse_id = g("user_email"), g("sql_warehouse_id")
databricks_app_name = g("databricks_app_name")
databricks_app_names = g("databricks_app_names") or databricks_app_name

# set BEFORE importing helpers
os.environ["DATABRICKS_APP_NAMES"] = ",".join([n.strip() for n in databricks_app_names.replace(":", ",").split(",") if n.strip()])  # UI + MCP
os.environ["DATABRICKS_APP_NAME"] = databricks_app_name  # legacy single-app fallback

from genesis_workbench.workbench import (
    initialize,
    set_app_permissions_for_job,
    set_app_permissions_for_volume,
)
from genesis_workbench.models import register_batch_model, ModelCategory

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# 1. Persist the orchestrator job id to settings (key LIKE '%_job_id' is what
#    grant_app_permissions queries to keep the grant in sync on redeploys).
spark.sql(f"""
    MERGE INTO settings AS target
    USING (SELECT 'run_vaccine_immunogen_job_id' AS key, '{run_vaccine_immunogen_job_id}' AS value, 'vaccine_immunogen' AS module) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
""")
print(f"settings: run_vaccine_immunogen_job_id = {run_vaccine_immunogen_job_id}")

# COMMAND ----------

# 2. Grant the app SP CAN_MANAGE_RUN on the orchestrator job + WRITE on the cache
#    volume (the app uploads each run's epitope-motif PDB to it before dispatching —
#    without WRITE_VOLUME the dispatcher fails with "User does not have WRITE
#    VOLUME privilege on VOLUME ...").
set_app_permissions_for_job(job_id=run_vaccine_immunogen_job_id, user_email=user_email)
set_app_permissions_for_volume(
    volume_full_name=f"{catalog}.{schema}.vaccine_immunogen",
    write=True,
)
print("Granted app SP CAN_MANAGE_RUN on the orchestrator job + WRITE on the motif-upload volume.")

# COMMAND ----------

# 3. Register Vaccine Immunogen Design as a batch model (the orchestrator is the user-launchable workflow).
register_batch_model(
    model_name="vaccine_immunogen",
    model_display_name="Vaccine Immunogen Design",
    model_description="Design + reward-optimize stable scaffold proteins that present a conserved epitope motif (epitope-focused immunogen) with RFD4-Proteina motif-scaffolding (in-process, H100), scored on epitope-presentation fidelity (motif RMSD), scaffold fold confidence, and manufacturability.",
    model_category=str(ModelCategory.LARGE_MOLECULE),
    module="vaccine_immunogen",
    job_id=run_vaccine_immunogen_job_id,
    job_name="run_vaccine_immunogen_gwb",
    cluster_type="GPU",
    added_by=user_email,
)
print("registered Vaccine Immunogen Design in batch_models")
