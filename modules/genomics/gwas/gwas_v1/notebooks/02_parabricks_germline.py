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

# MAGIC %sh python -m ensurepip --upgrade && python -m pip install --upgrade pip && python -m pip install mlflow>=2.15 databricks-sdk>=0.50.0

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
fastq_r1 = dbutils.widgets.get("fastq_r1")
fastq_r2 = dbutils.widgets.get("fastq_r2")
reference_genome_path = dbutils.widgets.get("reference_genome_path")
output_volume_path = dbutils.widgets.get("output_volume_path")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
user_email = dbutils.widgets.get("user_email")

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

# Validate input files exist
for label, path in [("FASTQ R1", fastq_r1), ("FASTQ R2", fastq_r2), ("Reference", reference_genome_path)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    print(f"  {label}: {os.path.getsize(path) / (1024*1024):.0f} MB")

# COMMAND ----------

import subprocess

def run_pbrun(cmd_str, description, log_path):
    """Run a pbrun command, streaming output in real-time."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd_str}")
    print(f"{'='*60}")
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log_fh.write(line)
        proc.wait()
    if proc.returncode != 0:
        with open(log_path) as f:
            lines = f.readlines()
        print(f"\nLast 50 lines of output:")
        for line in lines[-50:]:
            print(line, end="")
        raise RuntimeError(f"{description} failed with exit code {proc.returncode}")
    print(f"\n{description} completed successfully")

# Verify GPU and Parabricks are available
run_pbrun("nvidia-smi", "GPU check", os.path.join(run_output_dir, "gpu.log"))
run_pbrun("pbrun --version", "Parabricks version", os.path.join(run_output_dir, "version.log"))

# Step 1: fq2bam (alignment)
fq2bam_cmd = (
    f"pbrun fq2bam "
    f"--ref {reference_genome_path} "
    f"--in-fq {fastq_r1} {fastq_r2} "
    f"--out-bam {output_bam} "
    f"--low-memory"
)
run_pbrun(fq2bam_cmd, "fq2bam (alignment)", os.path.join(run_output_dir, "fq2bam.log"))

# Step 2: haplotypecaller (variant calling)
haplotype_cmd = (
    f"pbrun haplotypecaller "
    f"--ref {reference_genome_path} "
    f"--in-bam {output_bam} "
    f"--out-variants {output_vcf}"
)
run_pbrun(haplotype_cmd, "haplotypecaller (variant calling)", os.path.join(run_output_dir, "haplotypecaller.log"))

print("\nParabricks pipeline completed successfully")

# COMMAND ----------

with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.log_param("fastq_r1", fastq_r1)
    mlflow.log_param("fastq_r2", fastq_r2)
    mlflow.log_param("reference_genome", reference_genome_path)
    mlflow.log_param("output_bam", output_bam)
    mlflow.log_param("output_vcf", output_vcf)
    mlflow.set_tag("job_status", "alignment_complete")

print(f"Alignment complete. VCF: {output_vcf}")
