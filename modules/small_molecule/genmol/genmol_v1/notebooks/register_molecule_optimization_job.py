# Databricks notebook source
# MAGIC %md
# MAGIC ### Register the Guided Molecule Design orchestrator job
# MAGIC Persists the `run_molecule_optimization` job id in the `settings` table and
# MAGIC grants the Genesis Workbench app's service principal CAN_MANAGE_RUN on it, so
# MAGIC the app can dispatch the optimization loop (`jobs.run_now`). Mirrors
# MAGIC `register_enzyme_optimization_job.py`. Re-run any time the job id changes;
# MAGIC `grant_app_permissions.py` (which iterates `key LIKE '%_job_id'`) keeps the
# MAGIC grant in sync on subsequent full deploys.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("run_molecule_optimization_job_id", "", "Orchestrator Job ID")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App name")

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
run_molecule_optimization_job_id = dbutils.widgets.get("run_molecule_optimization_job_id")
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

# Persist the orchestrator job id in settings so grant_app_permissions.py
# (iterates key LIKE '%_job_id') keeps the app SP's CAN_MANAGE_RUN grant in sync.
spark.sql(f"""
MERGE INTO settings AS target
USING (SELECT 'run_molecule_optimization_job_id' AS key, '{run_molecule_optimization_job_id}' AS value, 'small_molecule' AS module) AS source
ON target.key = source.key AND target.module = source.module
WHEN MATCHED THEN UPDATE SET target.value = source.value
WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
""")

# COMMAND ----------

# Grant the app SP CAN_MANAGE_RUN now (resolves the SP from the app name — no
# hardcoded id), so the app can find + dispatch the orchestrator immediately.
from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=run_molecule_optimization_job_id, user_email=user_email)
print(f"Granted app CAN_MANAGE_RUN on job {run_molecule_optimization_job_id}")
