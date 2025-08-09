# Databricks notebook source

# COMMAND ----------

#parameters to the notebook
dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "genesis_schema", "Schema")
dbutils.widgets.text("bionemo_esm_finetune_job_id", "1234", "BioNeMo ESM Fine Tune Job ID")
dbutils.widgets.text("bionemo_esm_inference_job_id", "1234", "BioNeMo ESM Inference Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

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
bionemo_esm_finetune_job_id = dbutils.widgets.get("bionemo_esm_finetune_job_id")
bionemo_esm_inference_job_id = dbutils.widgets.get("bionemo_esm_inference_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")

spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS bionemo_weights")

spark.sql(f"""
CREATE TABLE  bionemo_weights (
    ft_id BIGINT ,
    ft_label STRING,
    model_type STRING,
    variant STRING,
    experiment_name STRING,
    run_id STRING,
    weights_volume_location STRING,
    created_by STRING,
    created_datetime TIMESTAMP,
    is_active BOOLEAN,
    deactivated_timestamp TIMESTAMP
)
""")

# COMMAND ----------

query= f"""
    INSERT INTO settings VALUES
    ('bionemo_esm_finetune_job_id', '{bionemo_esm_finetune_job_id}', 'bionemo'),
    ('bionemo_esm_inference_job_id', '{bionemo_esm_inference_job_id}' , 'bionemo')
"""

spark.sql(query)

# COMMAND ----------

#Grant app permission to run this job
from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=bionemo_esm_finetune_job_id, user_email=user_email)
set_app_permissions_for_job(job_id=bionemo_esm_inference_job_id, user_email=user_email)