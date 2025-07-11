# Databricks notebook source
## USE SERVERLESS? 

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
# dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("schema", "mmt_test", "Schema")
dbutils.widgets.text("user_email", "may.merkletan@databricks.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# MAGIC # dbutils.library.restartPython()

# COMMAND ----------

### TEMP FIX to update code because I can't get core to deploy

# COMMAND ----------

# DBTITLE 1,when empty
# # Create the volume in Unity Catalog if it does not exist
# spark.sql(f"""
# CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.libraries
# COMMENT 'Volume for libraries'
# """)

# dbutils.fs.ls(f"dbfs:/Volumes/{catalog}/{schema}/libraries")

# COMMAND ----------

# DBTITLE 1,check files
# # catalog
# schema_SCN = "dev_srijit_nair_dbx_genesis_workbench_core"
# dbutils.fs.ls(f"/Volumes/{catalog}/{schema_SCN}/libraries")

# COMMAND ----------

# DBTITLE 1,copy over libraries
# # Define the source and destination paths
# source_path = f"/Volumes/{catalog}/{schema_SCN}/libraries/genesis_workbench-0.1.0-py3-none-any.whl"
# destination_path = f"/Volumes/{catalog}/{schema}/libraries/genesis_workbench-0.1.0-py3-none-any.whl"

# # Create the volume in Unity Catalog if it does not exist
# spark.sql(f"""
# CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.libraries
# COMMENT 'Volume for libraries'
# """)

# # Create the volume path if it does not exist
# volume_path = f"/Volumes/{catalog}/{schema}/libraries"
# dbutils.fs.mkdirs(volume_path)

# # Copy the file
# dbutils.fs.cp(source_path, destination_path)

# COMMAND ----------

# DBTITLE 1,check destination
# dbutils.fs.ls(destination_path)

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

import os

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["SQL_WAREHOUSE"]=sql_warehouse_id
os.environ["IS_TOKEN_AUTH"]="Y"
os.environ["DATABRICKS_TOKEN"]=databricks_token


# COMMAND ----------

from genesis_workbench.models import (ModelCategory, 
                                      import_model_from_uc,
                                      get_latest_model_version)

from genesis_workbench.workbench import AppContext

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# List models in the schema
models = w.registered_models.list(
    catalog_name="genesis_workbench",
    schema_name="mmt_test"
)

# for model in models:
#     print(f"Model: {model.name}")
#     print(f"Created: {model.created_at}")
#     print("---")

model_list = [model.name for model in models];

model_list

# COMMAND ----------

# DBTITLE 1,gene_order
model_uc_name=f"{catalog}.{schema}.scimilarity_gene_order"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(model_uri)

app_context = AppContext(
        core_catalog_name=catalog,
        core_schema_name=schema
    )

import_model_from_uc(app_context,user_email=user_email,
                    model_category=ModelCategory.SINGLE_CELL,
                    model_uc_name=f"{catalog}.{schema}.scimilarity_gene_order",
                    model_uc_version=model_version,
                    model_name="SCimilarity_Gene_Order",
                    model_display_name="SCimilarity:GeneOrder",
                    model_source_version="v0.4.0",
                    model_description_url="hhttps://genentech.github.io/scimilarity/index.html")

# COMMAND ----------

model_uc_name=f"{catalog}.{schema}.scimilarity_get_embedding"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(model_uri)

app_context = AppContext(
        core_catalog_name=catalog,
        core_schema_name=schema
    )

import_model_from_uc(app_context,user_email=user_email,
                    model_category=ModelCategory.SINGLE_CELL,
                    model_uc_name=f"{catalog}.{schema}.scimilarity_get_embedding",
                    model_uc_version=model_version,
                    model_name="SCimilarity_Get_Embedding",
                    model_display_name="SCimilarity_Get_Embedding",
                    model_source_version="v0.4.0",
                    model_description_url="hhttps://genentech.github.io/scimilarity/index.html")

# COMMAND ----------

model_uc_name=f"{catalog}.{schema}.scimilarity_search_nearest"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"
print(model_uri)

app_context = AppContext(
        core_catalog_name=catalog,
        core_schema_name=schema
    )

import_model_from_uc(app_context,user_email=user_email,
                    model_category=ModelCategory.SINGLE_CELL,
                    model_uc_name=f"{catalog}.{schema}.scimilarity_search_nearest",
                    model_uc_version=model_version,
                    model_name="SCimilarity_Search_Nearest",
                    model_display_name="SCimilarity_Search_Nearest",
                    model_source_version="v0.4.0",
                    model_description_url="hhttps://genentech.github.io/scimilarity/index.html")

# COMMAND ----------


