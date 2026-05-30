# Databricks notebook source
# MAGIC %md
# MAGIC ### Download Reference Databases
# MAGIC Downloads the ClinVar GRCh38 VCF from NCBI and the ACMG SF v3.2 gene panel BED file,
# MAGIC then saves both as Delta tables for use in variant clinical annotation.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")

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

glow_whl_path = None
glow_jar_path = None
glow_libs = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in glow_libs:
    if lib.name.startswith("glow") and lib.name.endswith(".whl"):
        glow_whl_path = lib.path.replace("dbfs:", "")
    if lib.name.startswith("glow") and lib.name.endswith(".jar"):
        glow_jar_path = lib.path.replace("dbfs:", "")

print(f"GWB library: {gwb_library_path}")
print(f"Glow wheel: {glow_whl_path}")
print(f"Glow JAR: {glow_jar_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} {glow_whl_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download ClinVar VCF from NCBI FTP

# COMMAND ----------

import os
import subprocess

clinvar_volume = f"/Volumes/{catalog}/{schema}/variant_annotation_reference/clinvar"
os.makedirs(clinvar_volume, exist_ok=True)

clinvar_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
clinvar_tbi_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi"

clinvar_vcf_path = os.path.join(clinvar_volume, "clinvar_GRCh38.vcf.gz")
clinvar_tbi_path = os.path.join(clinvar_volume, "clinvar_GRCh38.vcf.gz.tbi")

if not os.path.exists(clinvar_vcf_path):
    print("Downloading ClinVar VCF...")
    subprocess.run(["wget", "-q", "-O", clinvar_vcf_path, clinvar_url], check=True)
    subprocess.run(["wget", "-q", "-O", clinvar_tbi_path, clinvar_tbi_url], check=True)
    print(f"Downloaded ClinVar to {clinvar_vcf_path}")
else:
    print(f"ClinVar already exists at {clinvar_vcf_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load ClinVar into Delta table

# COMMAND ----------

import glow
spark = glow.register(spark)

# COMMAND ----------

clinvar_df = spark.read.format("vcf").load(clinvar_vcf_path)

clinvar_table = f"{catalog}.{schema}.clinvar_variants"
clinvar_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(clinvar_table)

row_count = spark.table(clinvar_table).count()
print(f"ClinVar loaded: {row_count} variants written to {clinvar_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download and Load ACMG SF v3.2 Gene Panel
# MAGIC
# MAGIC The ACMG Secondary Findings v3.2 list contains 81 medically actionable genes:
# MAGIC - **Cancer** (29 genes): BRCA1, BRCA2, TP53, APC, MLH1, MSH2, etc.
# MAGIC - **Cardiovascular** (44 genes): MYBPC3, SCN5A, KCNQ1, TTN, etc.
# MAGIC - **Metabolic** (4 genes): GAA, GLA, OTC, BTD
# MAGIC - **Miscellaneous** (4 genes): HFE, HNF1A, ATP7B, RPE65

# COMMAND ----------

# The BED file is uploaded to the UC Volume by deploy.sh
acmg_bed_path = f"/Volumes/{catalog}/{schema}/variant_annotation_reference/acmg/ACMG_SFv3.2_GRCh38.bed"
print(f"Reading ACMG BED file from {acmg_bed_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load ACMG Gene Panel into Delta table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType

acmg_schema = StructType([
    StructField("chrom", StringType(), False),
    StructField("chromStart", LongType(), False),
    StructField("chromEnd", LongType(), False),
    StructField("name", StringType(), False),
    StructField("score", IntegerType(), True),
    StructField("strand", StringType(), True),
    StructField("category", StringType(), True),
    StructField("condition", StringType(), True),
])

acmg_df = spark.read.csv(acmg_bed_path, sep="\t", comment="#", schema=acmg_schema)

acmg_table = f"{catalog}.{schema}.acmg_gene_panel"
acmg_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(acmg_table)

acmg_count = spark.table(acmg_table).count()
print(f"ACMG gene panel loaded: {acmg_count} gene regions written to {acmg_table}")
