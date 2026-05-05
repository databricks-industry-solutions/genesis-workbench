# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("module", "single_cell", "Model Category for which endpoints will be destroyed")
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
module = dbutils.widgets.get("module")
destroy_user_email = dbutils.widgets.get("destroy_user_email")


# COMMAND ----------

print(f"catalog: {catalog}")
print(f"schema: {schema}")
print(f"sql_warehouse_id: {sql_warehouse_id}")
print(f"module: {module}")
print(f"destroy_user_email: {destroy_user_email}")


# COMMAND ----------

#Setup the env variables required for Genesis Workbench library to load settings
from genesis_workbench.workbench import initialize
from genesis_workbench.models import (get_deployed_models, 
                                      delete_endpoint)

from genesis_workbench.workbench import (execute_select_query,
                                         execute_non_select_query)

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)

# COMMAND ----------

deployed_models = get_deployed_models(module)

# COMMAND ----------

deployed_models

# COMMAND ----------

for index, row in deployed_models[["model_id", "deployment_id", "model_endpoint_name"]].iterrows():
    model_id = row['model_id']
    deployment_id = row['deployment_id']
    delete_endpoint(catalog, schema, deployment_id)

# COMMAND ----------

# TEMPORARILY DISABLED — re-enable after testing the rest of the destroy flow.
# Per-module Vector Search resources created at runtime by submodule notebooks
# (06c for scimilarity, 04 for sequence_search). The bundle doesn't own these,
# so they have to be cleaned up explicitly here. Source Delta tables are
# intentionally left in place — storage is cheap and the data is reusable.
#
# MODULE_VS_RESOURCES = {
#     "single_cell": [
#         {
#             "endpoint": "gwb_scimilarity_vs_endpoint",
#             "index": f"{catalog}.{schema}.scimilarity_cell_index",
#         },
#     ],
#     "protein_studies": [
#         {
#             "endpoint": "gwb_sequence_search_vs_endpoint",
#             "index": f"{catalog}.{schema}.sequence_embedding_index",
#         },
#     ],
# }
#
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.errors import NotFound
#
# w = WorkspaceClient()
#
# for entry in MODULE_VS_RESOURCES.get(module, []):
#     index_name = entry["index"]
#     endpoint_name = entry["endpoint"]
#
#     print(f"⏩️ Deleting Vector Search index {index_name}")
#     try:
#         w.vector_search_indexes.delete_index(index_name=index_name)
#         print(f"  Deleted index {index_name}")
#     except NotFound:
#         print(f"  Index {index_name} does not exist — skipping")
#     except Exception as e:
#         print(f"  Error deleting index {index_name}: {e}. Delete it manually")
#
#     print(f"⏩️ Deleting Vector Search endpoint {endpoint_name}")
#     try:
#         w.vector_search_endpoints.delete_endpoint(name=endpoint_name)
#         print(f"  Deleted endpoint {endpoint_name}")
#     except NotFound:
#         print(f"  Endpoint {endpoint_name} does not exist — skipping")
#     except Exception as e:
#         print(f"  Error deleting endpoint {endpoint_name}: {e}. Delete it manually")

# COMMAND ----------

#lets deactivate all the models registered in Genesis Workbench
spark.sql(f"""
     UPDATE {catalog}.{schema}.models SET
        is_active = 'false',
        deactivated_timestamp = current_timestamp()
     WHERE 
        model_category = '{module}'  AND 
        is_active = 'true'  
""")

# COMMAND ----------

#Remove all module settings
spark.sql(f"""
     DELETE FROM {catalog}.{schema}.settings 
     WHERE 
        module = '{module}' 
""")

