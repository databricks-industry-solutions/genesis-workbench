# Databricks notebook source

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("parabricks_alignment_job_id", "1234", "Parabricks Alignment Job ID")
dbutils.widgets.text("gwas_analysis_job_id", "1234", "GWAS Analysis Job ID")
dbutils.widgets.text("user_email", "a@b.com", "Email of the user running the deploy")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")

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

print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
parabricks_alignment_job_id = dbutils.widgets.get("parabricks_alignment_job_id")
gwas_analysis_job_id = dbutils.widgets.get("gwas_analysis_job_id")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Parabricks Alignment Job ID: {parabricks_alignment_job_id}")
print(f"GWAS Analysis Job ID: {gwas_analysis_job_id}")

# COMMAND ----------

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

ref_genome_path = f"/Volumes/{catalog}/{schema}/gwas_reference/genomes/GRCh38_full_analysis_set_plus_decoy_hla.fa"

query = f"""
    MERGE INTO settings AS target
    USING (
        SELECT * FROM VALUES
            ('parabricks_alignment_job_id', '{parabricks_alignment_job_id}', 'disease_biology'),
            ('gwas_analysis_job_id', '{gwas_analysis_job_id}', 'disease_biology'),
            ('gwas_reference_genome_path', '{ref_genome_path}', 'disease_biology'),
            ('gwas_sample_vcf_path', '/Volumes/{catalog}/{schema}/gwas_data/sample_vcf/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz', 'disease_biology'),
            ('gwas_sample_phenotype_path', '/Volumes/{catalog}/{schema}/gwas_data/sample_phenotype/breast_cancer_phenotype.tsv', 'disease_biology')
        AS src(key, value, module)
    ) AS source
    ON target.key = source.key AND target.module = source.module
    WHEN MATCHED THEN UPDATE SET target.value = source.value
    WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
"""

spark.sql(query)

# COMMAND ----------

from genesis_workbench.workbench import set_app_permissions_for_job

set_app_permissions_for_job(job_id=parabricks_alignment_job_id, user_email=user_email)
set_app_permissions_for_job(job_id=gwas_analysis_job_id, user_email=user_email)

# COMMAND ----------

# Copy bundled sample phenotype data to the gwas_data volume
import os, shutil

sample_data_dir = f"/Volumes/{catalog}/{schema}/gwas_data/sample_phenotype"
os.makedirs(sample_data_dir, exist_ok=True)

current_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
workspace_root = os.sep.join(current_path.split(os.sep)[:-1])
source_tsv = f"/Workspace{workspace_root}/../data/example_sample_list.tsv"

if os.path.exists(source_tsv):
    dest = os.path.join(sample_data_dir, "example_sample_list.tsv")
    shutil.copy2(source_tsv, dest)
    print(f"Copied sample phenotype data to {dest}")
else:
    print(f"Sample data file not found at {source_tsv}, skipping")

source_phenotype = f"/Workspace{workspace_root}/../data/breast_cancer_phenotype.tsv"
if os.path.exists(source_phenotype):
    dest = os.path.join(sample_data_dir, "breast_cancer_phenotype.tsv")
    shutil.copy2(source_phenotype, dest)
    print(f"Copied breast cancer phenotype data to {dest}")
else:
    print(f"Phenotype file not found at {source_phenotype}, skipping")

# COMMAND ----------

# Register Glow as a batch model so it appears in the Deployed Models tab
from genesis_workbench.models import register_batch_model

register_batch_model(
    model_name="glow",
    model_display_name="Glow Genomics",
    model_description="GPU-accelerated genomics library for VCF ingestion, variant annotation, and GWAS analysis on Spark",
    model_category="disease_biology",
    module="disease_biology",
    job_id=gwas_analysis_job_id,
    job_name="gwas_analysis",
    cluster_type="CPU",
    added_by=user_email,
)

print("Disease Biology GWAS module initialization complete")
