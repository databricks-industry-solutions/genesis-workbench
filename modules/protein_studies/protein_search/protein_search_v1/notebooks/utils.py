# Databricks notebook source
# DBTITLE 1,Declare notebook widgets (pre-filled by DAB job parameters at deploy time)

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "ai_driven_drug_discovery", "Schema")
dbutils.widgets.text("volume_name", "protein_seq", "Volume Name")
dbutils.widgets.text("foundation_model_endpoint", "databricks-claude-sonnet-4-5", "Foundation Model Endpoint")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")

# COMMAND ----------

# DBTITLE 1,Read widget values and ensure UC resources exist

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")


volume_location = f"/Volumes/{catalog}/{schema}/{volume_name}"

print(f"catalog:         {catalog}")
print(f"schema:          {schema}")
print(f"volume_name:     {volume_name}")
print(f"volume_location: {volume_location}")
