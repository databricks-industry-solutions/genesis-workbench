# Databricks notebook source
# MAGIC %md
# MAGIC ### Filter Gene Regions & Annotate with ClinVar
# MAGIC Filters variants to specified gene regions (e.g., BRCA1/BRCA2), then
# MAGIC joins with ClinVar to add clinical significance annotations and
# MAGIC identify pathogenic variants.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("variants_table", "", "Variants Table (from spike step)")
dbutils.widgets.text("gene_regions", '[{"name":"BRCA1","contig":"chr17","start":43044292,"end":43170327},{"name":"BRCA2","contig":"chr13","start":32315086,"end":32400268}]', "Gene Regions JSON")
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
gene_regions_json = dbutils.widgets.get("gene_regions")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
user_email = dbutils.widgets.get("user_email")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

import json
import mlflow
import pyspark.sql.functions as F
from functools import reduce

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

gene_regions = json.loads(gene_regions_json)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter variants to gene regions

# COMMAND ----------

# Use the output from the spike step
source_table = f"{catalog}.{schema}.variant_annotation_variants_with_pathogenic"
vcf_df = spark.table(source_table)

# Build filter condition for each gene region
region_filters = []
for region in gene_regions:
    region_filter = (
        (F.col("contigName") == region["contig"]) &
        (F.col("start") >= region["start"]) &
        (F.col("end") <= region["end"])
    )
    region_filters.append(region_filter)

combined_filter = reduce(lambda a, b: a | b, region_filters)
gene_variants = vcf_df.where(combined_filter)

# Add gene name based on region
gene_case = F.when(F.lit(False), F.lit(""))
for region in gene_regions:
    gene_case = gene_case.when(
        (F.col("contigName") == region["contig"]) &
        (F.col("start") >= region["start"]) &
        (F.col("end") <= region["end"]),
        F.lit(region["name"])
    )
gene_case = gene_case.otherwise(F.lit("Unknown"))

gene_variants = gene_variants.withColumn("gene", gene_case)

# Add derived columns
gene_variants = gene_variants.withColumn(
    "chromosome", F.col("contigName")
).withColumn(
    "ref", F.col("referenceAllele")
).withColumn(
    "alt", F.col("alternateAlleles")[0]
).withColumn(
    "zygosity",
    F.when(
        F.col("genotypes")[0]["calls"][0] == F.col("genotypes")[0]["calls"][1],
        F.lit("Homozygous Alt")
    ).otherwise(F.lit("Heterozygous"))
)

gene_variant_count = gene_variants.count()
print(f"Found {gene_variant_count} variants in {len(gene_regions)} gene region(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Annotate with ClinVar

# COMMAND ----------

clinvar_table = f"{catalog}.{schema}.clinvar_variants"
clinvar_df = spark.table(clinvar_table)

# Join variants with ClinVar (handle chr prefix mismatch)
annotated = gene_variants.alias("gv").join(
    clinvar_df.alias("cv"),
    (F.regexp_replace(F.col("gv.contigName"), "chr", "") == F.col("cv.contigName")) &
    (F.col("gv.start") == F.col("cv.start")) &
    (F.col("gv.referenceAllele") == F.col("cv.referenceAllele")) &
    (F.col("gv.alternateAlleles")[0] == F.col("cv.alternateAlleles")[0]),
    "left_outer"
).select(
    F.col("gv.gene"),
    F.col("gv.chromosome"),
    F.col("gv.start"),
    F.col("gv.ref"),
    F.col("gv.alt"),
    F.col("gv.zygosity"),
    F.col("gv.qual"),
    F.col("cv.INFO_CLNSIG").alias("clinical_significance"),
    F.col("cv.INFO_CLNDN").alias("disease_name"),
    F.col("cv.INFO_CLNVARID").alias("variant_ids"),
    F.col("gv.genotypes"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save annotated variants and identify pathogenic

# COMMAND ----------

clinical_annotated_table = f"{catalog}.{schema}.variant_annotation_clinical_annotated"
annotated.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(clinical_annotated_table)

# Filter pathogenic variants
pathogenic = annotated.where(
    F.array_contains(F.col("clinical_significance"), "Pathogenic") |
    F.array_contains(F.col("clinical_significance"), "Likely_pathogenic")
)

pathogenic_table = f"{catalog}.{schema}.variant_annotation_pathogenic"
pathogenic.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(pathogenic_table)

pathogenic_count = spark.table(pathogenic_table).count()
annotated_count = spark.table(clinical_annotated_table).count()

# COMMAND ----------

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_param("gene_regions", gene_regions_json)
    mlflow.log_param("clinvar_table", clinvar_table)
    mlflow.log_metric("gene_region_variants", annotated_count)
    mlflow.log_metric("pathogenic_variants", pathogenic_count)
    mlflow.set_tag("annotated_table", clinical_annotated_table)
    mlflow.set_tag("pathogenic_table", pathogenic_table)
    mlflow.set_tag("job_status", "annotation_complete")

print(f"Annotation complete: {annotated_count} annotated, {pathogenic_count} pathogenic")
