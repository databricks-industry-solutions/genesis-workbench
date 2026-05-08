# Databricks notebook source
# MAGIC %md
# MAGIC ### Grant App Permissions
# MAGIC
# MAGIC Grants the Genesis Workbench app service principal:
# MAGIC - `CAN_QUERY` on all serving endpoints
# MAGIC - `CAN_MANAGE_RUN` on all registered jobs (from the settings table)
# MAGIC - `READ VOLUME` on all volumes in the schema
# MAGIC - `EXECUTE` on all registered models in Unity Catalog

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App Name")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
databricks_app_name = dbutils.widgets.get("databricks_app_name")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3

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
databricks_app_name = dbutils.widgets.get("databricks_app_name")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

import os
os.environ["DATABRICKS_APP_NAME"] = databricks_app_name

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant CAN_QUERY on all serving endpoints

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel

w = WorkspaceClient()
app = w.apps.get(name=databricks_app_name)
app_sp_id = app.service_principal_client_id

print(f"App: {databricks_app_name}, SP: {app_sp_id}")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

endpoint_names_df = spark.sql(
    "SELECT DISTINCT model_endpoint_name FROM model_deployments WHERE is_active = true AND model_endpoint_name IS NOT NULL"
).collect()

print(f"Found {len(endpoint_names_df)} deployed endpoints in model_deployments")

for row in endpoint_names_df:
    ep_name = row["model_endpoint_name"]
    try:
        ep = w.serving_endpoints.get(ep_name)
        perms = w.serving_endpoints.get_permissions(serving_endpoint_id=ep.id)
        already_granted = any(
            acl.user_name == app_sp_id
            for acl in (perms.access_control_list or [])
            for p in (acl.all_permissions or [])
            if p.permission_level and "CAN_QUERY" in str(p.permission_level)
        )
        if not already_granted:
            w.serving_endpoints.update_permissions(
                serving_endpoint_id=ep.id,
                access_control_list=[
                    ServingEndpointAccessControlRequest(
                        user_name=app_sp_id,
                        permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
                    )
                ],
            )
            print(f"  Granted CAN_QUERY on {ep_name}")
        else:
            print(f"  Already has CAN_QUERY on {ep_name}, skipping")
    except Exception as e:
        print(f"  Warning: Could not set permissions on {ep_name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant CAN_MANAGE_RUN on all registered jobs

# COMMAND ----------

from databricks.sdk.service.jobs import JobAccessControlRequest, JobPermissionLevel

job_ids_df = spark.sql("SELECT value FROM settings WHERE key LIKE '%_job_id'").collect()

print(f"Found {len(job_ids_df)} registered jobs")

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().getOrElse(None)

for row in job_ids_df:
    job_id = row["value"]
    try:
        perms = w.jobs.get_permissions(job_id=job_id)
        already_granted = any(
            acl.user_name == app_sp_id
            for acl in (perms.access_control_list or [])
            for p in (acl.all_permissions or [])
            if p.permission_level and "CAN_MANAGE_RUN" in str(p.permission_level)
        )
        if not already_granted:
            w.jobs.update_permissions(
                job_id=job_id,
                access_control_list=[
                    JobAccessControlRequest(
                        user_name=app_sp_id,
                        permission_level=JobPermissionLevel.CAN_MANAGE_RUN,
                    )
                ],
            )
            print(f"  Granted CAN_MANAGE_RUN on job {job_id}")
        else:
            print(f"  Already has CAN_MANAGE_RUN on job {job_id}, skipping")
    except Exception as e:
        print(f"  Warning: Could not set permissions on job {job_id}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant READ VOLUME and WRITE VOLUME on all volumes in the schema

# COMMAND ----------

try:
    spark.sql(f"GRANT READ VOLUME ON SCHEMA {catalog}.{schema} TO `{app_sp_id}`")
    print(f"Granted READ VOLUME on schema {catalog}.{schema}")
except Exception as e:
    print(f"Warning: Could not grant READ VOLUME on schema: {e}")

# WRITE VOLUME is required so the app can upload per-run inputs (e.g. the
# motif PDB the Guided Enzyme Optimization form persists to UC before
# dispatching the orchestrator job — otherwise the form errors with
# "Failed to dispatch job: [Errno 13] Permission denied: '/Volumes'").
try:
    spark.sql(f"GRANT WRITE VOLUME ON SCHEMA {catalog}.{schema} TO `{app_sp_id}`")
    print(f"Granted WRITE VOLUME on schema {catalog}.{schema}")
except Exception as e:
    print(f"Warning: Could not grant WRITE VOLUME on schema: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grant EXECUTE on all registered models in Unity Catalog

# COMMAND ----------

model_names_df = spark.sql(
    f"SELECT DISTINCT deploy_model_uc_name FROM model_deployments WHERE is_active = true AND deploy_model_uc_name IS NOT NULL"
).collect()

print(f"Found {len(model_names_df)} registered models in model_deployments")

for row in model_names_df:
    model_uc_name = row["deploy_model_uc_name"]
    try:
        spark.sql(f"GRANT EXECUTE ON FUNCTION {model_uc_name} TO `{app_sp_id}`")
        print(f"  Granted EXECUTE on {model_uc_name}")
    except Exception as e:
        if "already has" in str(e).lower() or "inherited" in str(e).lower():
            print(f"  Already has EXECUTE on {model_uc_name}, skipping")
        else:
            print(f"  Warning: Could not grant EXECUTE on {model_uc_name}: {e}")

# COMMAND ----------

print("App permissions grant complete.")
