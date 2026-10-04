# Databricks notebook source
# MAGIC %md
# MAGIC # KERMT — Register jobs (Layer 2)
# MAGIC Runs once per deploy (after the orchestrator jobs exist):
# MAGIC 1. Persist `kermt_finetune_job_id` / `kermt_deploy_job_id` to the `settings` table so
# MAGIC    `grant_app_permissions` keeps the app SP's CAN_MANAGE_RUN grant in sync on redeploys.
# MAGIC 2. Grant the app SP CAN_MANAGE_RUN on both jobs now (so the app can dispatch immediately).
# MAGIC 3. Register KERMT as a batch model in `batch_models`.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("kermt_finetune_job_id", "", "KERMT finetune orchestrator job id")
dbutils.widgets.text("kermt_deploy_job_id", "", "KERMT deploy orchestrator job id")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App Name")
dbutils.widgets.text("databricks_app_names", "genesis-workbench:mcp-genesis-workbench", "Databricks App Names (colon/comma-separated, UI + MCP)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
g = dbutils.widgets.get
catalog, schema = g("catalog"), g("schema")
sql_warehouse_id, user_email = g("sql_warehouse_id"), g("user_email")
kermt_finetune_job_id = g("kermt_finetune_job_id")
kermt_deploy_job_id = g("kermt_deploy_job_id")
databricks_app_name = g("databricks_app_name")
databricks_app_names = dbutils.widgets.get("databricks_app_names") or databricks_app_name

# set BEFORE importing helpers
os.environ["DATABRICKS_APP_NAMES"] = ",".join([n.strip() for n in databricks_app_names.replace(":", ",").split(",") if n.strip()])  # UI + MCP
os.environ["DATABRICKS_APP_NAME"] = databricks_app_name  # legacy single-app fallback

from genesis_workbench.workbench import initialize, set_app_permissions_for_job
from genesis_workbench.models import register_batch_model, ModelCategory

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# 1. Persist job ids to settings (key LIKE '%_job_id' is what grant_app_permissions queries).
for key, val in (("kermt_finetune_job_id", kermt_finetune_job_id),
                 ("kermt_deploy_job_id", kermt_deploy_job_id)):
    spark.sql(f"""
        MERGE INTO settings AS target
        USING (SELECT '{key}' AS key, '{val}' AS value, 'kermt' AS module) AS source
        ON target.key = source.key AND target.module = source.module
        WHEN MATCHED THEN UPDATE SET target.value = source.value
        WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
    """)
    print(f"settings: {key} = {val}")

# COMMAND ----------

# 2. Grant the app SP CAN_MANAGE_RUN on both jobs.
set_app_permissions_for_job(job_id=kermt_finetune_job_id, user_email=user_email)
set_app_permissions_for_job(job_id=kermt_deploy_job_id, user_email=user_email)
print("app permissions granted on finetune + deploy jobs")

# COMMAND ----------

# 3. Register KERMT as a batch model (the finetune job is the user-launchable workflow).
register_batch_model(
    model_name="kermt",
    model_display_name="KERMT (Kinetic GROVER Multi-Task)",
    model_description="GNN for small-molecule ADMET/tox property prediction; fine-tune + serve.",
    model_category=str(ModelCategory.SMALL_MOLECULE),
    module="kermt",
    job_id=kermt_finetune_job_id,
    job_name="kermt_finetune_job",
    cluster_type="GPU",
    added_by=user_email,
)
# Also register the deploy job so the "Deploy KERMT" canvas node shows deployed
# unconditionally (batch_models are always in the palette's available set), rather
# than depending on the app SP seeing the job via a cached jobs.list().
register_batch_model(
    model_name="kermt_deploy",
    model_display_name="Deploy KERMT",
    model_description="Deploy a fine-tuned KERMT checkpoint as an ADMET serving endpoint.",
    model_category=str(ModelCategory.SMALL_MOLECULE),
    module="kermt",
    job_id=kermt_deploy_job_id,
    job_name="kermt_deploy_job",
    cluster_type="GPU",
    added_by=user_email,
)
print("registered KERMT in batch_models")
