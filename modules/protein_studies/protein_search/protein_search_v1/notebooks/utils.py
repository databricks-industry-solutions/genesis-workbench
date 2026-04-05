# Databricks notebook source
# DBTITLE 1,Declare notebook widgets (pre-filled by DAB job parameters at deploy time)

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "ai_driven_drug_discovery", "Schema")
dbutils.widgets.text("volume_name", "protein_seq", "Volume Name")
dbutils.widgets.text("external_endpoint_name", "az_openai_gpt4o", "AI Gateway Endpoint")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")

# COMMAND ----------

# DBTITLE 1,Read widget values and ensure UC resources exist

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume_name}")

volume_location = f"/Volumes/{catalog}/{schema}/{volume_name}"

print(f"catalog:         {catalog}")
print(f"schema:          {schema}")
print(f"volume_name:     {volume_name}")
print(f"volume_location: {volume_location}")
