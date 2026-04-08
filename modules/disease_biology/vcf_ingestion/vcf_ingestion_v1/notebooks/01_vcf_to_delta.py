# Databricks notebook source
# MAGIC %md
# MAGIC ### VCF to Delta Conversion
# MAGIC Reads a VCF file using [Glow](https://glow.readthedocs.io/) and persists it as a Delta table
# MAGIC for downstream analysis (GWAS, variant annotation, etc.).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("vcf_path", "", "VCF File Path")
dbutils.widgets.text("output_table_name", "", "Output Delta Table Name")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %pip install databricks-sdk>=0.50.0 databricks-sql-connector>=4.0.2 mlflow>=2.15

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
vcf_path = dbutils.widgets.get("vcf_path")
output_table_name = dbutils.widgets.get("output_table_name")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
user_email = dbutils.widgets.get("user_email")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register Glow and load VCF

# COMMAND ----------

import glow
spark = glow.register(spark)

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

full_table_name = f"{catalog}.{schema}.{output_table_name}"

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_param("vcf_path", vcf_path)
    mlflow.log_param("output_table", full_table_name)
    mlflow.set_tag("job_status", "ingestion_started")

    # Read VCF with Glow
    vcf_df = spark.read.format("vcf").load(vcf_path)

    row_count = vcf_df.count()
    mlflow.log_metric("total_variants", row_count)
    mlflow.log_param("variant_count", str(row_count))

    # Write to Delta table
    vcf_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)

    mlflow.set_tag("job_status", "ingestion_complete")
    mlflow.set_tag("output_table", full_table_name)

# COMMAND ----------

print(f"VCF ingestion complete: {row_count} variants written to {full_table_name}")
