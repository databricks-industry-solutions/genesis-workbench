# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("variant_annotation_job_id", "1234", "Variant Annotation Job ID")
dbutils.widgets.text("variant_annotation_dashboard_id", "1234", "Variant Annotation Dashboard ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("databricks_app_names", "genesis-workbench:mcp-genesis-workbench", "Databricks App Names (colon/comma-separated, UI + MCP)")

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
            ('variant_annotation_job_id', '{variant_annotation_job_id}', 'genomics'),
            ('variant_annotation_dashboard_id', '{variant_annotation_dashboard_id}', 'genomics'),
            ('sample_fastq_r1', '/Volumes/{catalog}/{schema}/gwas_data/sample_fastq/sample_1.fq.gz', 'genomics'),
            ('sample_fastq_r2', '/Volumes/{catalog}/{schema}/gwas_data/sample_fastq/sample_2.fq.gz', 'genomics')
        AS src(key, value, module)
    ) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
"""

spark.sql(query)

# COMMAND ----------

from genesis_workbench.workbench import set_app_permissions_for_job

import os
_app_names_raw = dbutils.widgets.get("databricks_app_names")
os.environ["DATABRICKS_APP_NAMES"] = ",".join([n.strip() for n in _app_names_raw.replace(":", ",").split(",") if n.strip()])  # UI + MCP

set_app_permissions_for_job(job_id=variant_annotation_job_id, user_email=user_email)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dashboard catalog/schema
# MAGIC The dashboard uses `dataset_catalog` and `dataset_schema` parameters
# MAGIC (set in variant_annotation_dashboard.yml) so queries resolve to the correct catalog/schema
# MAGIC at deploy time. No runtime substitution needed.


# COMMAND ----------

# Download sample FASTQ files from 1000 Genomes for Variant Calling demo.
# Use HTTPS, not FTP: this task runs on serverless, whose egress does not reliably
# allow FTP, so the old `ftp://` wget would stall with no progress until the task
# timeout (the cause of the variant_annotation_init timeouts). curl over HTTPS with
# retries + stall detection (--speed-limit/--speed-time) pulls the ~3.6 GB reliably
# in a few minutes. On failure the partial file is removed so a re-run doesn't treat
# a truncated download as complete (the os.path.exists guard would otherwise skip it).
import subprocess
import os

sample_fastq_dir = f"/Volumes/{catalog}/{schema}/gwas_data/sample_fastq"
os.makedirs(sample_fastq_dir, exist_ok=True)

fastq_base_url = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read"
fastq_files = {
    "sample_1.fq.gz": f"{fastq_base_url}/SRR062634_1.filt.fastq.gz",
    "sample_2.fq.gz": f"{fastq_base_url}/SRR062634_2.filt.fastq.gz",
}

for dest_name, url in fastq_files.items():
    dest_path = os.path.join(sample_fastq_dir, dest_name)
    if os.path.exists(dest_path):
        print(f"{dest_name} already exists, skipping")
        continue
    print(f"Downloading {dest_name} from {url} ...")
    try:
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10",
             "--connect-timeout", "30", "--speed-limit", "10000", "--speed-time", "60",
             "-o", dest_path, url],
            check=True,
        )
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)  # don't leave a partial a re-run would treat as complete
        raise
    print(f"Downloaded {dest_name}")

# COMMAND ----------

print("Genomics Variant Annotation module initialization complete")
