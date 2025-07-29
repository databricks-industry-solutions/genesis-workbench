# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

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

model_uc_name=f"{catalog}.{schema}.rfdiffusion_unconditional"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"

import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.PROTEIN_STUDIES,
                    model_uc_name=f"{catalog}.{schema}.rfdiffusion_unconditional",
                    model_uc_version=model_version,
                    model_name="RFdiffusion_Unconditional",
                    model_display_name="RFdiffusion Unconditional",
                    model_source_version="v1.1.0",
                    model_description_url="https://github.com/RosettaCommons/RFdiffusion")

# COMMAND ----------

model_uc_name=f"{catalog}.{schema}.rfdiffusion_inpainting"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"

import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.PROTEIN_STUDIES,
                    model_uc_name=f"{catalog}.{schema}.rfdiffusion_inpainting",
                    model_uc_version=model_version,
                    model_name="RFdiffusion_Inpainting",
                    model_display_name="RFdiffusion Inpainting",
                    model_source_version="v1.1.0",
                    model_description_url="https://github.com/RosettaCommons/RFdiffusion")
