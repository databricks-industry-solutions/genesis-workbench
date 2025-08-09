# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
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

import os

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
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

model_uc_name=f"{catalog}.{schema}.rfdiffusion_unconditional"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"

gwb_model_id_unconditional = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.PROTEIN_STUDIES,
                    model_uc_name=f"{catalog}.{schema}.rfdiffusion_unconditional",
                    model_uc_version=model_version,
                    model_name="RFdiffusion_Unconditional",
                    model_display_name="RFdiffusion Unconditional",
                    model_source_version="v1.1.0",
                    model_description_url="https://github.com/RosettaCommons/RFdiffusion")

# COMMAND ----------

run_id_unconditional = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id_unconditional,
                deployment_name=f"RFdiffusion_Unconditional",
                deployment_description="Initial deployment",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

model_uc_name=f"{catalog}.{schema}.rfdiffusion_inpainting"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"

gwb_model_id_inpainting = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.PROTEIN_STUDIES,
                    model_uc_name=f"{catalog}.{schema}.rfdiffusion_inpainting",
                    model_uc_version=model_version,
                    model_name="RFdiffusion_Inpainting",
                    model_display_name="RFdiffusion Inpainting",
                    model_source_version="v1.1.0",
                    model_description_url="https://github.com/RosettaCommons/RFdiffusion")

# COMMAND ----------

run_id_inpainting = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id_inpainting,
                deployment_name=f"RFdiffusion_Inpainting",
                deployment_description="Initial deployment",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result1 = wait_for_job_run_completion(run_id_unconditional, timeout = 3600)

# COMMAND ----------

result2 = wait_for_job_run_completion(run_id_inpainting, timeout = 3600)
