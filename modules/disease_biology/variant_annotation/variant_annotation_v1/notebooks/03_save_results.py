# Databricks notebook source
# MAGIC %md
# MAGIC ### Save Annotation Results
# MAGIC Enriches pathogenic variants with disease names, computes summary
# MAGIC statistics, and logs final metrics to MLflow.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
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
# MAGIC #### Enrich pathogenic variants with disease context

# COMMAND ----------

pathogenic_table = f"{catalog}.{schema}.variant_annotation_pathogenic"
clinical_annotated_table = f"{catalog}.{schema}.variant_annotation_clinical_annotated"

pathogenic_df = spark.table(pathogenic_table)
annotated_df = spark.table(clinical_annotated_table)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compute summary statistics

# COMMAND ----------

total_annotated = annotated_df.count()
total_pathogenic = pathogenic_df.count()

genes_analyzed = annotated_df.select("gene").distinct().collect()
gene_names = [row["gene"] for row in genes_analyzed]

gene_counts = annotated_df.groupBy("gene").count().collect()
gene_summary = {row["gene"]: row["count"] for row in gene_counts}

# COMMAND ----------

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_metric("total_annotated_variants", total_annotated)
    mlflow.log_metric("total_pathogenic_variants", total_pathogenic)
    mlflow.log_metric("genes_analyzed", len(gene_names))

    for gene, count in gene_summary.items():
        mlflow.log_metric(f"variants_{gene}", count)

    mlflow.set_tag("genes_analyzed", ",".join(gene_names))
    mlflow.set_tag("job_status", "annotation_complete")
    mlflow.set_tag("annotated_table", clinical_annotated_table)
    mlflow.set_tag("pathogenic_table", pathogenic_table)

# COMMAND ----------

print(f"Results summary:")
print(f"  Total annotated variants: {total_annotated}")
print(f"  Pathogenic variants: {total_pathogenic}")
print(f"  Genes analyzed: {', '.join(gene_names)}")
for gene, count in gene_summary.items():
    print(f"    {gene}: {count} variants")
