# Databricks notebook source
# DBTITLE 1,gwb_paramsNvariables
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog") 
dbutils.widgets.text("schema", "dev_mmt_core_test", "Schema") #dev_mmt_core_test# gets overwritten during DAB deployment 
dbutils.widgets.text("user_email", "may.merkletan@databricks.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# DBTITLE 1,install standard dependencies
# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC # dbutils.library.restartPython() ## run after installing gwb wheel below

# COMMAND ----------

# DBTITLE 1,extract gwb library from core app wheel file
gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

# COMMAND ----------

# DBTITLE 1,install gwb library & dependencies
try:
    %pip install {gwb_library_path} --force-reinstall
    dbutils.library.restartPython()
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# DBTITLE 1,get UC variables & set env auth
import os

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")


# COMMAND ----------

#Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)


# COMMAND ----------

# DBTITLE 1,import class/functions from gwb library
from genesis_workbench.models import (ModelCategory, 
                                      import_model_from_uc,
                                      deploy_model,
                                      get_latest_model_version)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

# DBTITLE 1,scimilarity_gene_order
model_uc_name=f"{catalog}.{schema}.scimilarity_gene_order"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(model_uri)


import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SINGLE_CELL,
                    model_uc_name=f"{catalog}.{schema}.scimilarity_gene_order",
                    model_uc_version=model_version,
                    model_name="SCimilarity_Gene_Order",
                    model_display_name="SCimilarity:GeneOrder",
                    model_source_version="v0.4.0_weights_v1.1",
                    model_description_url="https://genentech.github.io/scimilarity/index.html")

# COMMAND ----------

# DBTITLE 1,scimilarity_get_embedding
model_uc_name=f"{catalog}.{schema}.scimilarity_get_embedding"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(model_uri)

gwb_model_id_get_embedding = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SINGLE_CELL,
                    model_uc_name=f"{catalog}.{schema}.scimilarity_get_embedding",
                    model_uc_version=model_version,
                    model_name="SCimilarity_Get_Embedding",
                    model_display_name="SCimilarity:GetEmbedding",
                    model_source_version="v0.4.0_weights_v1.1",
                    model_description_url="https://genentech.github.io/scimilarity/index.html")

# COMMAND ----------

run_id_get_embedding = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id_get_embedding,
                deployment_name=f"Scimilarity_Get_Embedding",
                deployment_description="Initial deployment",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type="GPU_SMALL",
                workload_size="Small")

# COMMAND ----------

# DBTITLE 1,scimilarity_search_nearest
model_uc_name=f"{catalog}.{schema}.scimilarity_search_nearest"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(model_uri)

gwb_model_id_search = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SINGLE_CELL,
                    model_uc_name=f"{catalog}.{schema}.scimilarity_search_nearest",
                    model_uc_version=model_version,
                    model_name="SCimilarity_Search_Nearest",
                    model_display_name="SCimilarity:SearchNearest",
                    model_source_version="v0.4.0_weights_v1.1",
                    model_description_url="https://genentech.github.io/scimilarity/index.html")

# COMMAND ----------

run_id_search = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id_search,
                deployment_name=f"Scimilarity_Search_Nearest",
                deployment_description="Initial deployment",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type="GPU_SMALL",
                workload_size="Small")

# COMMAND ----------

result1 = wait_for_job_run_completion(run_id_get_embedding, timeout = 3600)


# COMMAND ----------

result2 = wait_for_job_run_completion(run_id_search, timeout = 3600)

