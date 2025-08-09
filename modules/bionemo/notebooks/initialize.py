# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.61.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#parameters to the notebook
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("bionemo_esm_finetune_job_id", "1234", "BioNeMo ESM Fine Tune Job ID")
dbutils.widgets.text("bionemo_esm_inference_job_id", "1234", "BioNeMo ESM Inference Job ID")
dbutils.widgets.text("dev_user_prefix", "abc", "Prefix for resources")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "Email of the user running the deploy")

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

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
bionemo_esm_finetune_job_id = dbutils.widgets.get("bionemo_esm_finetune_job_id")
bionemo_esm_inference_job_id = dbutils.widgets.get("bionemo_esm_inference_job_id")
dev_user_prefix = dbutils.widgets.get("dev_user_prefix")
user_email = dbutils.widgets.get("user_email")

dev_user_prefix = None if dev_user_prefix.strip() == "" or dev_user_prefix.strip().lower()=="none" else dev_user_prefix

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

assert catalog and schema, "Catalog and schema must be provided"

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