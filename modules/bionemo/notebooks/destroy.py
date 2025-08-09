# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.61.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#parameters to the notebook
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

assert catalog and schema, "Catalog and schema must be provided"

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")

spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------
spark.sql("DROP TABLE IF EXISTS bionemo_weights")



