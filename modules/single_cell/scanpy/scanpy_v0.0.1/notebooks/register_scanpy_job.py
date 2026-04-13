# Databricks notebook source
dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "genesis_schema", "Schema")
dbutils.widgets.text("run_scanpy_job_id", "", "Scanpy Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")
run_scanpy_job_id = dbutils.widgets.get("run_scanpy_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

spark.sql(f"""
MERGE INTO settings AS target
USING (SELECT 'run_scanpy_job_id' AS key, '{run_scanpy_job_id}' AS value, 'single_cell' AS module) AS source
ON target.key = source.key AND target.module = source.module
WHEN MATCHED THEN UPDATE SET target.value = source.value
WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
""")

# COMMAND ----------

#Grant app permission to run this job
from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=run_scanpy_job_id, user_email=user_email)

# COMMAND ----------

# Register as batch model so it appears in the Deployed Models tab
from genesis_workbench.models import register_batch_model

register_batch_model(
    model_name="scanpy",
    model_display_name="Scanpy Single Cell Analysis",
    model_description="CPU-based single-cell QC, clustering, UMAP, marker gene detection, and optional pseudotime",
    model_category="single_cell",
    module="single_cell",
    job_id=run_scanpy_job_id,
    job_name="run_scanpy",
    cluster_type="CPU",
    added_by=user_email,
)

print(f"Successfully registered scanpy job with ID: {run_scanpy_job_id}")
print(f"App permissions granted for job execution")

