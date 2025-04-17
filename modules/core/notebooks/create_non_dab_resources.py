# Databricks notebook source
#parameters to the notebook
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_name = dbutils.widgets.get("sql_warehouse_name")
tags = dbutils.widgets.get("tags")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

# COMMAND ----------

print(tags)
