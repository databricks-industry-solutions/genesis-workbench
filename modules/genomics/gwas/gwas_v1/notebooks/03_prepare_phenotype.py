# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare Phenotype Dataset
# MAGIC
# MAGIC Reads user-supplied phenotype data (CSV or TSV) and writes a Delta table
# MAGIC with `sampleId` and `phenotype` columns suitable for Glow GWAS regression.
# MAGIC
# MAGIC Adapted from mini-glow-demo notebook 2.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("phenotype_path", "", "Phenotype file path (UC Volume)")
dbutils.widgets.text("phenotype_column", "phenotype", "Column name for phenotype labels")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
phenotype_path = dbutils.widgets.get("phenotype_path")
phenotype_column = dbutils.widgets.get("phenotype_column")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")

# COMMAND ----------

import pyspark.sql.functions as F

is_tsv = phenotype_path.endswith(".tsv")
sep = "\t" if is_tsv else ","

pheno_df = spark.read.csv(phenotype_path, sep=sep, header=True, inferSchema=True)

print(f"Loaded phenotype file: {phenotype_path}")
print(f"Columns: {pheno_df.columns}")
print(f"Rows: {pheno_df.count()}")

# COMMAND ----------

# Ensure we have a sampleId column (try common variants)
sample_col = None
for candidate in ["sampleId", "Sample name", "sample_id", "SampleID", "sample_name"]:
    if candidate in pheno_df.columns:
        sample_col = candidate
        break

if sample_col is None:
    sample_col = pheno_df.columns[0]
    print(f"No standard sample ID column found, using first column: {sample_col}")

if sample_col != "sampleId":
    pheno_df = pheno_df.withColumnRenamed(sample_col, "sampleId")

# Clean column names: replace spaces with underscores
pheno_df = pheno_df.select([F.col(col).alias(col.replace(' ', '_')) for col in pheno_df.columns])

# COMMAND ----------

# Ensure the phenotype column exists and is integer-typed
pheno_col = phenotype_column.replace(' ', '_')
if pheno_col not in pheno_df.columns:
    distinct_values = pheno_df.select(pheno_col if pheno_col in pheno_df.columns
                                       else pheno_df.columns[-1]).distinct().collect()
    raise ValueError(f"Phenotype column '{phenotype_column}' not found. Available: {pheno_df.columns}")

col_type = dict(pheno_df.dtypes).get(pheno_col, "string")
if col_type == "string":
    distinct_vals = [row[0] for row in pheno_df.select(pheno_col).distinct().collect()]
    label_map = {v: i for i, v in enumerate(sorted(distinct_vals))}
    print(f"Mapping string phenotype to integers: {label_map}")

    from pyspark.sql.types import IntegerType
    map_udf = F.udf(lambda x: label_map.get(x, -1), IntegerType())
    pheno_df = pheno_df.withColumn("phenotype", map_udf(F.col(pheno_col)))
elif pheno_col != "phenotype":
    pheno_df = pheno_df.withColumn("phenotype", F.col(pheno_col).cast("int"))

# COMMAND ----------

phenotype_table = f"gwas_phenotype_{mlflow_run_id.replace('-', '_')}"
pheno_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{phenotype_table}")
print(f"Phenotype table written: {catalog}.{schema}.{phenotype_table}")

# COMMAND ----------

# MAGIC %pip install mlflow==2.22.0

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_param("phenotype_path", phenotype_path)
    mlflow.log_param("phenotype_column", phenotype_column)
    mlflow.log_param("phenotype_table", f"{catalog}.{schema}.{phenotype_table}")
    mlflow.log_param("phenotype_row_count", pheno_df.count())
    mlflow.set_tag("job_status", "phenotype_prepared")
