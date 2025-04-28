# Databricks notebook source
#parameters to the notebook
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
workspace_artifact_location = dbutils.widgets.get("workspace_artifact_location")
artifacts = dbutils.widgets.get("artifacts")

# COMMAND ----------

#for testing
catalog = "genesis_workbench"
schema = "dev_srijit_nair_dbx_genesis_workbench_core"
volume = "libraries"
workspace_artifact_location = "dbfs/Workspace/Users/srijit.nair@databricks.com/.bundle/genesis_workbench/core/dev/artifacts/.internal"
artifacts = "genesis_workbench-0.1.0-py3-none-any.whl"

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Volume: {volume}")
print(f"Workspace Artifact Location: {workspace_artifact_location}")
print(f"Artifacts: {artifacts}")

# COMMAND ----------

assert catalog and schema and volume, "Catalog, schema and volume must be provided"
assert workspace_artifact_location, "Workspace Artifact Location must be provided"
assert artifacts, "Artifacts must be provided"

# COMMAND ----------

# MAGIC %pip install databricks-sql-connector
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks import sql
import os

# COMMAND ----------


def get_warehouse_details_from_id(warehouse_id):
    w = WorkspaceClient()
    details = w.warehouses.get(warehouse_id)
    return {
        "id": details.id,
        "cluster_size" : details.cluster_size,
        "name": details.name,
        "hostname": details.odbc_params.hostname,
        "http_path": details.odbc_params.path
    }

def db_connect():
    warehouse_id = os.getenv("SQL_WAREHOUSE")
    warehouse_details = get_warehouse_details_from_id(warehouse_id)
    print(warehouse_details)    
    return sql.connect(server_hostname = f"https://{warehouse_details['hostname']}",
        http_path = warehouse_details['http_path'],
        access_token=dbutils.secrets.get(scope="multi_agent", key="pat"))

# COMMAND ----------

os.environ["SQL_WAREHOUSE"]="b721abe0cc790b1f"

# COMMAND ----------

import pandas as pd

def execute_query(query)-> pd.DataFrame:
    with(db_connect()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [ col_desc[0] for col_desc in cursor.description]
            result = pd.DataFrame.from_records(cursor.fetchall(),columns=columns)
            return result

display(execute_query("SELECT * FROM  srijit_nair.coverage.cpt_codes LIMIT 10"))

   

# COMMAND ----------

type(result)

# COMMAND ----------



# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat
import io

w = WorkspaceClient()

artifact_paths = [ f"{workspace_artifact_location}/{artifact}" for artifact in artifacts.split(",")]

for artifact_path in artifact_paths:
    destination_path = f"/Volumes/{catalog}/{schema}/{volume}/"
    print(f"Copying {artifact_path} to {destination_path}")
    
        artifact_content = w.workspace.export(
            path=artifact_path, 
            format=ExportFormat.AUTO  
        ).content
        
        w.files.upload(
            destination_path,
            contents=io.BytesIO(artifact_content), 
            overwrite=True
        )

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al /Workspace/Users/srijit.nair@databricks.com/.bundle

# COMMAND ----------

context_str = '--var="dev_user_prefix=scn,core_catalog_name=genesis_workbench,core_schema_name=dev_srijit_nair_dbx_genesis_workbench_core"'

# COMMAND ----------

ctx_items = {}
context_str = context_str.replace("--var=","").replace("\"","")
[(lambda x : ctx_items.update({x[0]:x[1]}) )(ctx_item.split("=")) for ctx_item in context_str.split(",")] 

# COMMAND ----------

ctx_items

# COMMAND ----------

lambda x : {x[0]:x[1]}

# COMMAND ----------

from databricks.sdk.core import Config, 

# COMMAND ----------


