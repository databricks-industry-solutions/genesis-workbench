# Databricks notebook source
#parameters to the notebook
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_name = dbutils.widgets.get("sql_warehouse_name")
tags = dbutils.widgets.get("tags")

# COMMAND ----------

#for testing 
#catalog = "genesis_workbench"
#schema = "dev_scn_dbx_genesis_workbench_core"
#sql_warehouse_name = "dev_scn_dbx_genesis_workbench_warehouse"
#tags = '{"owner":"genesis_workbench","tag_name1":"value1"}'
#current_user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
import datetime
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create SQL Warehouse
# MAGIC

# COMMAND ----------

resource_tags = json.loads(tags)
sql_warehouse_tags = [sql.EndpointTagPair(key=key, value=value) for key, value in resource_tags.items()]

w = WorkspaceClient()
gwb_warehouse = w.warehouses.create_and_wait(name=sql_warehouse_name,
                cluster_size="2X-Small",
                enable_serverless_compute=True,
                max_num_clusters=1,
                auto_stop_mins=10,
                tags=sql.EndpointTags(custom_tags=sql_warehouse_tags),
                timeout= datetime.timedelta(minutes=20)
            )

# COMMAND ----------

spark.sql(f"""
          INSERT INTO {catalog}.{schema}.non_dab_resources (
              resource_name ,
              resource_type,
              resource_metadata,
              created_date,
              created_by
          )          
          VALUES (
              '{sql_warehouse_name}',
              'sql_warehouse',
              PARSE_JSON('{json.dumps(gwb_warehouse.as_dict())}'),
              CURRENT_TIMESTAMP,
              '{current_user_name}'
              )
              
""")

# COMMAND ----------


