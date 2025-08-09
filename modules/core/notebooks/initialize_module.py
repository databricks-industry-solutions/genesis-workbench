# Databricks notebook source

# COMMAND ----------

#parameters to the notebook
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("module", "single_cell", "Model Category for which endpoints will be destroyed")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "Email of the user running the deploy")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
module = dbutils.widgets.get("module")
user_email = dbutils.widgets.get("user_email")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

query= f"""
    INSERT INTO {catalog}.{schema}.settings VALUES
    ('{module}_deployed', 'true', '{module}')
"""

spark.sql(query)


# COMMAND ----------
#Add any module specific logic here

# COMMAND ----------
#If any resources access need to be granted by app, add here