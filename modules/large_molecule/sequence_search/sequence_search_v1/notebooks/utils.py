# Databricks notebook source
# DBTITLE 1,Declare notebook widgets (pre-filled by DAB job parameters at deploy time)

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "ai_driven_drug_discovery", "Schema")
dbutils.widgets.text("volume_name", "sequence_search", "Volume Name")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")

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
