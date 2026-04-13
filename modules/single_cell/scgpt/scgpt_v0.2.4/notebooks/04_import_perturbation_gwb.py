# Databricks notebook source
## USE SERVERLESS

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("perturb_model_name", "scgpt_perturbation", "Perturbation Model Name")
dbutils.widgets.text("experiment_name", "scgpt_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "scgpt_cache_dir", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

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
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
model_name = dbutils.widgets.get("perturb_model_name")
workload_type = dbutils.widgets.get("workload_type")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      deploy_model,
                                      get_latest_model_version)
from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)
print(f"Model: {model_uc_name} v{model_version}")

gwb_model_id = import_model_from_uc(
    user_email=user_email,
    model_category=ModelCategory.SINGLE_CELL,
    model_uc_name=model_uc_name,
    model_uc_version=model_version,
    model_name="scgpt_Perturbation",
    model_display_name="scGPT Perturbation Predictor",
    model_source_version="v0.2.4",
    model_description_url="https://github.com/bowang-lab/scGPT/blob/main/README.md",
)

# COMMAND ----------

run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="scGPT Perturbation",
    deployment_description="Zero-shot gene perturbation prediction using scGPT transformer",
    input_adapter_str="none",
    output_adapter_str="none",
    sample_input_data_dict_as_json="none",
    sample_params_as_json="none",
    workload_type=workload_type,
    workload_size="Small",
)

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)
