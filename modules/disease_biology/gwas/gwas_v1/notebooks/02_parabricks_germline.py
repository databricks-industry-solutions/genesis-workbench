# Databricks notebook source
# MAGIC %md
# MAGIC # Parabricks Germline Variant Calling
# MAGIC
# MAGIC Runs `pbrun germline` on user-supplied FASTQ files to produce
# MAGIC aligned BAM and germline VCF output.
# MAGIC
# MAGIC This notebook runs on a Parabricks-enabled Docker cluster (GPU).

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("fastq_r1", "", "FASTQ Read 1 path")
dbutils.widgets.text("fastq_r2", "", "FASTQ Read 2 path")
dbutils.widgets.text("reference_genome_path", "", "Reference genome FASTA path")
dbutils.widgets.text("output_volume_path", "", "Output UC Volume path")
dbutils.widgets.text("mlflow_run_id", "", "MLflow Run ID")
dbutils.widgets.text("user_email", "a@b.com", "User Email")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
fastq_r1 = dbutils.widgets.get("fastq_r1")
fastq_r2 = dbutils.widgets.get("fastq_r2")
reference_genome_path = dbutils.widgets.get("reference_genome_path")
output_volume_path = dbutils.widgets.get("output_volume_path")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
user_email = dbutils.widgets.get("user_email")

# COMMAND ----------

# MAGIC %pip install mlflow>=2.15 databricks-sdk>=0.50.0

# COMMAND ----------

import os
import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

run_output_dir = os.path.join(output_volume_path, "alignment", mlflow_run_id)
os.makedirs(run_output_dir, exist_ok=True)

output_bam = os.path.join(run_output_dir, "output.bam")
output_vcf = os.path.join(run_output_dir, "germline.vcf")
output_recal = os.path.join(run_output_dir, "recal.txt")

print(f"Input FASTQ R1: {fastq_r1}")
print(f"Input FASTQ R2: {fastq_r2}")
print(f"Reference: {reference_genome_path}")
print(f"Output dir: {run_output_dir}")

# COMMAND ----------

import subprocess

cmd = [
    "pbrun", "germline",
    "--ref", reference_genome_path,
    "--in-fq", fastq_r1, fastq_r2,
    "--out-bam", output_bam,
    "--out-variants", output_vcf,
    "--out-recal-file", output_recal,
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

if result.returncode != 0:
    print(f"STDERR: {result.stderr[:2000]}")
    raise RuntimeError(f"pbrun germline failed with exit code {result.returncode}")

print("Parabricks germline completed successfully")

# COMMAND ----------

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_param("fastq_r1", fastq_r1)
    mlflow.log_param("fastq_r2", fastq_r2)
    mlflow.log_param("reference_genome", reference_genome_path)
    mlflow.log_param("output_bam", output_bam)
    mlflow.log_param("output_vcf", output_vcf)
    mlflow.set_tag("job_status", "alignment_complete")

print(f"Alignment complete. VCF: {output_vcf}")
