# Databricks notebook source

# COMMAND ----------

#parameters to the notebook
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("module", "single_cell", "Model Category for which endpoints will be destroyed")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
module = dbutils.widgets.get("module")
user_email = dbutils.widgets.get("user_email")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

query= f"""
    MERGE INTO {catalog}.{schema}.settings AS target
    USING (SELECT '{module}_deployed' AS key, 'true' AS value, '{module}' AS module) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
"""

spark.sql(query)


# COMMAND ----------
#Add any module specific logic here

# COMMAND ----------
#If any resources access need to be granted by app, add here