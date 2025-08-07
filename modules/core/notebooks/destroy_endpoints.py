# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("model_category", "single_cell", "Model Category for which endpoints will be destroyed")
dbutils.widgets.text("destroy_user_email", "a@b.com", "User Id")

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
# MAGIC

# COMMAND ----------

#parameters to the notebook
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
model_category = dbutils.widgets.get("model_category")
destroy_user_email = dbutils.widgets.get("destroy_user_email")


# COMMAND ----------

print(f"catalog: {catalog}")
print(f"schema: {schema}")
print(f"sql_warehouse_id: {sql_warehouse_id}")
print(f"model_category: {model_category}")
print(f"destroy_user_email: {destroy_user_email}")


# COMMAND ----------

#Setup the env variables required for Genesis Workbench library to load settings
from genesis_workbench.workbench import initialize
from genesis_workbench.models import (GWBModelInfo, 
                                      ModelDeploymentInfo, 
                                      get_gwb_model_info,
                                      get_deployed_models, 
                                      get_gwb_model_deployment_info,
                                      delete_endpoint)

from genesis_workbench.workbench import (execute_select_query,
                                         execute_non_select_query)

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)

# COMMAND ----------

deployed_models = get_deployed_models(model_category)

# COMMAND ----------

deployed_models

# COMMAND ----------

for index, row in deployed_models[["model_id", "deployment_id", "model_endpoint_name"]].iterrows():
    model_id = row['model_id']
    deployment_id = row['deployment_id']
    delete_endpoint(catalog, schema, deployment_id)

# COMMAND ----------

#lets deactivate all the models registered in Genesis Workbench
spark.sql("""
     UPDATE {catalog}.{schema}.models SET
        is_active = 'false',
        deactivated_timestamp = current_timestamp()
     WHERE 
        model_category = '{model_category}'  AND 
        is_active = 'true'  
""")
