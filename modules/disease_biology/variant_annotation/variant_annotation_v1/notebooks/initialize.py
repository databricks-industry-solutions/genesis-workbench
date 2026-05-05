# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("variant_annotation_job_id", "1234", "Variant Annotation Job ID")
dbutils.widgets.text("variant_annotation_dashboard_id", "1234", "Variant Annotation Dashboard ID")
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
variant_annotation_job_id = dbutils.widgets.get("variant_annotation_job_id")
variant_annotation_dashboard_id = dbutils.widgets.get("variant_annotation_dashboard_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Variant Annotation Job ID: {variant_annotation_job_id}")
print(f"Dashboard ID: {variant_annotation_dashboard_id}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# Schema migration: ensure variant annotation tables exist with the run_name column
va_tables = {
    "variant_annotation_variants_with_pathogenic": """
        CREATE TABLE IF NOT EXISTS {fq} (
            contigName STRING, start LONG, end LONG, names ARRAY<STRING>,
            referenceAllele STRING, alternateAlleles ARRAY<STRING>,
            qual DOUBLE, filters ARRAY<STRING>, splitFromMultiAllelic BOOLEAN,
            INFO_CLINVAR STRING, genotypes ARRAY<STRUCT<sampleId:STRING, calls:ARRAY<INT>>>,
            run_name STRING
        )
    """,
    "variant_annotation_clinical_annotated": """
        CREATE TABLE IF NOT EXISTS {fq} (
            gene STRING, category STRING, condition STRING,
            chromosome STRING, start LONG, ref STRING, alt STRING,
            zygosity STRING, qual DOUBLE,
            clinical_significance ARRAY<STRING>, disease_name ARRAY<STRING>,
            variant_ids ARRAY<STRING>,
            genotypes ARRAY<STRUCT<sampleId:STRING, calls:ARRAY<INT>>>,
            run_name STRING
        )
    """,
    "variant_annotation_pathogenic": """
        CREATE TABLE IF NOT EXISTS {fq} (
            gene STRING, category STRING, condition STRING,
            chromosome STRING, start LONG, ref STRING, alt STRING,
            zygosity STRING, qual DOUBLE,
            clinical_significance ARRAY<STRING>, disease_name ARRAY<STRING>,
            variant_ids ARRAY<STRING>,
            genotypes ARRAY<STRUCT<sampleId:STRING, calls:ARRAY<INT>>>,
            run_name STRING
        )
    """,
}

for tbl, ddl in va_tables.items():
    fq = f"{catalog}.{schema}.{tbl}"
    if spark.catalog.tableExists(fq):
        cols = [c.name for c in spark.table(fq).schema]
        if "run_name" not in cols:
            spark.sql(f"ALTER TABLE {fq} ADD COLUMNS (run_name STRING)")
            print(f"Added run_name column to {tbl}")
        else:
            print(f"{tbl} already has run_name column")
    else:
        spark.sql(ddl.format(fq=fq))
        print(f"Created table {tbl}")

# COMMAND ----------

query = f"""
    MERGE INTO settings AS target
    USING (
        SELECT * FROM VALUES
            ('variant_annotation_job_id', '{variant_annotation_job_id}', 'disease_biology'),
            ('variant_annotation_dashboard_id', '{variant_annotation_dashboard_id}', 'disease_biology'),
            ('sample_fastq_r1', '/Volumes/{catalog}/{schema}/gwas_data/sample_fastq/sample_1.fq.gz', 'disease_biology'),
            ('sample_fastq_r2', '/Volumes/{catalog}/{schema}/gwas_data/sample_fastq/sample_2.fq.gz', 'disease_biology')
        AS src(key, value, module)
    ) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
"""

spark.sql(query)

# COMMAND ----------

from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=variant_annotation_job_id, user_email=user_email)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dashboard catalog/schema
# MAGIC The dashboard uses `dataset_catalog` and `dataset_schema` parameters
# MAGIC (set in variant_annotation_dashboard.yml) so queries resolve to the correct catalog/schema
# MAGIC at deploy time. No runtime substitution needed.


# COMMAND ----------

# Download sample FASTQ files from 1000 Genomes for Variant Calling demo
import subprocess
import os

sample_fastq_dir = f"/Volumes/{catalog}/{schema}/gwas_data/sample_fastq"
os.makedirs(sample_fastq_dir, exist_ok=True)

fastq_base_url = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read"
fastq_files = {
    "sample_1.fq.gz": f"{fastq_base_url}/SRR062634_1.filt.fastq.gz",
    "sample_2.fq.gz": f"{fastq_base_url}/SRR062634_2.filt.fastq.gz",
}

for dest_name, url in fastq_files.items():
    dest_path = os.path.join(sample_fastq_dir, dest_name)
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_name}...")
        subprocess.run(["wget", "-q", "-O", dest_path, url], check=True)
        print(f"Downloaded {dest_name}")
    else:
        print(f"{dest_name} already exists, skipping")

# COMMAND ----------

print("Disease Biology Variant Annotation module initialization complete")
