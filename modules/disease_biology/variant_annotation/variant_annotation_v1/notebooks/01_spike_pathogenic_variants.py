# Databricks notebook source
# MAGIC %md
# MAGIC ### Spike Pathogenic Variants (Optional)
# MAGIC Adds known pathogenic variants (e.g., BRCA1/BRCA2) into the variant dataset
# MAGIC for demonstration purposes. If no `pathogenic_vcf_path` is provided, this step
# MAGIC passes through the source table unchanged.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("variants_table", "", "Source Variants Table")
dbutils.widgets.text("pathogenic_vcf_path", "", "Pathogenic VCF Path (optional)")
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
variants_table = dbutils.widgets.get("variants_table")
pathogenic_vcf_path = dbutils.widgets.get("pathogenic_vcf_path")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
user_email = dbutils.widgets.get("user_email")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

import mlflow
import pyspark.sql.functions as F

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load source variants and optionally spike pathogenic variants

# COMMAND ----------

existing_variants = spark.table(variants_table)
original_count = existing_variants.count()

output_table = f"{catalog}.{schema}.variant_annotation_variants_with_pathogenic"

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_param("source_variants_table", variants_table)
    mlflow.log_metric("original_variant_count", original_count)

    if pathogenic_vcf_path and pathogenic_vcf_path.strip():
        import glow
        spark_glow = glow.register(spark)

        pathogenic_variants = spark.read.format("vcf").load(pathogenic_vcf_path.strip())
        pathogenic_count = pathogenic_variants.count()

        enhanced_variants = existing_variants.unionByName(
            pathogenic_variants,
            allowMissingColumns=True
        )

        enhanced_variants.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)

        final_count = spark.table(output_table).count()
        mlflow.log_param("pathogenic_vcf_path", pathogenic_vcf_path)
        mlflow.log_metric("pathogenic_variants_added", pathogenic_count)
        mlflow.log_metric("enhanced_variant_count", final_count)
        mlflow.set_tag("spike_status", "spiked")

        print(f"Spiked {pathogenic_count} pathogenic variants. Total: {final_count}")
    else:
        # No spiking — pass through the source table
        existing_variants.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
        mlflow.set_tag("spike_status", "passthrough")
        print(f"No pathogenic VCF provided. Passed through {original_count} variants.")

    mlflow.set_tag("output_table", output_table)
