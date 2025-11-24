# Databricks notebook source

# COMMAND ----------

#parameters to the notebook
dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "genesis_schema", "Schema")
dbutils.widgets.text("parabricks_cluster_name", "1234", "Parabricks Cluster Name")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")

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

# COMMAND ----------

catalog = dbutils.widgets.get("core_catalog")
schema = dbutils.widgets.get("core_schema")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
parabricks_cluster_name = dbutils.widgets.get("parabricks_cluster_name")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")

spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

query= f"""
    INSERT INTO settings VALUES
    ('parabricks_cluster_name', '{parabricks_cluster_name}', 'parabricks')
"""

spark.sql(query)

