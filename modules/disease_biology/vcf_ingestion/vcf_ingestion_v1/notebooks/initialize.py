# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("vcf_ingestion_job_id", "1234", "VCF Ingestion Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vcf_ingestion_job_id = dbutils.widgets.get("vcf_ingestion_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"VCF Ingestion Job ID: {vcf_ingestion_job_id}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

query = f"""
    MERGE INTO settings AS target
    USING (SELECT 'vcf_ingestion_job_id' AS key, '{vcf_ingestion_job_id}' AS value, 'disease_biology' AS module) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
"""

spark.sql(query)

# COMMAND ----------

from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=vcf_ingestion_job_id, user_email=user_email)

# COMMAND ----------

print("Disease Biology VCF Ingestion module initialization complete")
