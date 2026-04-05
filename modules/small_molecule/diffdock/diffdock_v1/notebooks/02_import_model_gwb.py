# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
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

model_name = "diffdock_v1"
model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SMALL_MOLECULES,
                    model_uc_name=model_uc_name,
                    model_uc_version=model_version,
                    model_name="DiffDock",
                    model_display_name="DiffDock Molecular Docking",
                    model_source_version="v1.0 (commit 0f9c419)",
                    model_description_url="https://github.com/gcorso/DiffDock")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"DiffDock Molecular Docking",
                deployment_description="DiffDock molecular docking — predicts 3D binding poses for protein-ligand complexes using diffusion generative modeling with confidence ranking",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)
