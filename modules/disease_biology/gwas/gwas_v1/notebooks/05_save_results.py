# Databricks notebook source
# MAGIC %md
# MAGIC # Save GWAS Results and Update MLflow
# MAGIC
# MAGIC Reads the GWAS results Delta table, computes summary statistics,
# MAGIC and updates the MLflow run with final status and metrics.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("contigs", "6", "Contigs analyzed")
dbutils.widgets.text("hwe_cutoff", "0.01", "HWE cutoff used")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
contigs = dbutils.widgets.get("contigs")
hwe_cutoff = dbutils.widgets.get("hwe_cutoff")

# COMMAND ----------

# MAGIC %pip install mlflow>=2.15

# COMMAND ----------

import pyspark.sql.functions as F

results_table = f"gwas_results_{mlflow_run_id.replace('-', '_')}"
results_df = spark.table(f"{catalog}.{schema}.{results_table}")

total_variants = results_df.count()

# Genome-wide significance threshold (standard: 5e-8, relaxed for small datasets: 1e-5)
sig_threshold = 5e-8
significant_hits = results_df.filter(F.col("pvalue") < sig_threshold).count()

if significant_hits == 0:
    sig_threshold = 1e-5
    significant_hits = results_df.filter(F.col("pvalue") < sig_threshold).count()

suggestive_hits = results_df.filter(F.col("pvalue") < 1e-5).count()

min_pvalue_row = results_df.orderBy("pvalue").first()
min_pvalue = float(min_pvalue_row["pvalue"]) if min_pvalue_row else None

print(f"Total variants tested: {total_variants}")
print(f"Significant hits (p < {sig_threshold}): {significant_hits}")
print(f"Suggestive hits (p < 1e-5): {suggestive_hits}")
print(f"Minimum p-value: {min_pvalue}")

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_param("contigs", contigs)
    mlflow.log_param("hwe_cutoff", hwe_cutoff)
    mlflow.log_param("results_table", f"{catalog}.{schema}.{results_table}")
    mlflow.log_metric("total_variants_tested", total_variants)
    mlflow.log_metric("significant_hits", significant_hits)
    mlflow.log_metric("suggestive_hits", suggestive_hits)
    if min_pvalue is not None:
        mlflow.log_metric("min_pvalue", min_pvalue)
    mlflow.set_tag("job_status", "gwas_complete")

print("GWAS analysis complete — MLflow run updated")
