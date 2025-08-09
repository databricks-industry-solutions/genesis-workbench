# Databricks notebook source
## USE SERVERLESS? 

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_yyang_genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "scgpt", "Model Name")
dbutils.widgets.text("experiment_name", "scgpt_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "yang.yang@databricks.com", "User Id/Email")
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
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
model_name = dbutils.widgets.get("model_name")
workload_type = dbutils.widgets.get("workload_type")

# COMMAND ----------

#Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)


# COMMAND ----------

from genesis_workbench.models import (ModelCategory, 
                                      import_model_from_uc,
                                      deploy_model,
                                      get_latest_model_version)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# List models in the schema
models = w.registered_models.list(
    catalog_name=catalog,
    schema_name=schema
)

# for model in models:
#     print(f"Model: {model.name}")
#     print(f"Created: {model.created_at}")
#     print("---")

model_list = [model.name for model in models];

model_list

# COMMAND ----------

model_uc_name=f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(model_uri)

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SINGLE_CELL,
                    model_uc_name=f"{catalog}.{schema}.{model_name}",
                    model_uc_version=model_version,
                    model_name=f"{model_name}_Get_Embedding",
                    model_display_name=f"{model_name}_Get_Embedding",
                    model_source_version="v0.2.4",
                    model_description_url="https://github.com/bowang-lab/scGPT/blob/main/README.md")



# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"scGPT",
                deployment_description="Initial deployment",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout = 3600)
