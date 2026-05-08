# Databricks notebook source
# MAGIC %md
# MAGIC ### Register Enzyme Optimization Orchestrator Jobs
# MAGIC
# MAGIC Persists the two orchestrator job IDs (Fast / Accurate) into the
# MAGIC `settings` table and grants the Genesis Workbench app's service
# MAGIC principal `CAN_MANAGE_RUN` on each — without this, the Streamlit
# MAGIC dispatcher's `WorkspaceClient().jobs.list(name=...)` returns empty
# MAGIC because the app SP has no view on the job, and the form errors with
# MAGIC "Orchestrator job 'run_enzyme_optimization_gwb' not found".

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("run_enzyme_optimization_job_id", "", "Fast (CPU) Orchestrator Job ID")
dbutils.widgets.text("run_enzyme_optimization_inprocess_ame_job_id", "", "Accurate (A10) Orchestrator Job ID")
dbutils.widgets.text("user_email", "a@b.com", "User email")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App Name")

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

print(f"GWB library: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
run_enzyme_optimization_job_id = dbutils.widgets.get("run_enzyme_optimization_job_id")
run_enzyme_optimization_inprocess_ame_job_id = dbutils.widgets.get("run_enzyme_optimization_inprocess_ame_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
databricks_app_name = dbutils.widgets.get("databricks_app_name")

import os
os.environ["DATABRICKS_APP_NAME"] = databricks_app_name

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Persist both orchestrator job IDs in the settings table so the next run of
# `grant_app_permissions.py` (which iterates `key LIKE '%_job_id'`) keeps the
# app SP's CAN_MANAGE_RUN grant in sync after subsequent redeploys.
spark.sql(f"""
MERGE INTO settings AS target
USING (SELECT 'run_enzyme_optimization_job_id' AS key, '{run_enzyme_optimization_job_id}' AS value, 'small_molecule' AS module) AS source
ON target.key = source.key AND target.module = source.module
WHEN MATCHED THEN UPDATE SET target.value = source.value
WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
""")

spark.sql(f"""
MERGE INTO settings AS target
USING (SELECT 'run_enzyme_optimization_inprocess_ame_job_id' AS key, '{run_enzyme_optimization_inprocess_ame_job_id}' AS value, 'small_molecule' AS module) AS source
ON target.key = source.key AND target.module = source.module
WHEN MATCHED THEN UPDATE SET target.value = source.value
WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
""")

# COMMAND ----------

from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=run_enzyme_optimization_job_id, user_email=user_email)
set_app_permissions_for_job(job_id=run_enzyme_optimization_inprocess_ame_job_id, user_email=user_email)

print("Granted app SP CAN_MANAGE_RUN on both orchestrator jobs.")
