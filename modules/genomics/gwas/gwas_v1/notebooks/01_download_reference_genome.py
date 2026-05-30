# Databricks notebook source
# MAGIC %md
# MAGIC # Download Reference Genomes
# MAGIC
# MAGIC Downloads the GRCh38 reference genome and index files to a UC Volume
# MAGIC for use by Parabricks alignment and Glow variant normalization.
# MAGIC
# MAGIC Also downloads a sample VCF from 1000 Genomes for demo purposes.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

import os

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.gwas_reference")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.gwas_data")

genome_path = f"/Volumes/{catalog}/{schema}/gwas_reference/genomes"
os.makedirs(genome_path, exist_ok=True)

os.environ['GENOME_PATH'] = genome_path
os.environ['CATALOG'] = catalog
os.environ['SCHEMA'] = schema

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download GRCh38 reference genome

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /local_disk0/
# MAGIC mkdir -p tmp_ref
# MAGIC
# MAGIC if [ ! -f "$GENOME_PATH/GRCh38_full_analysis_set_plus_decoy_hla.fa" ]; then
# MAGIC   echo "Downloading GRCh38 reference genome..."
# MAGIC   wget -q https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa -P tmp_ref/
# MAGIC   cp tmp_ref/GRCh38_full_analysis_set_plus_decoy_hla.fa $GENOME_PATH/
# MAGIC   echo "Downloaded GRCh38 FASTA"
# MAGIC else
# MAGIC   echo "GRCh38 FASTA already exists, skipping"
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC if [ ! -f "$GENOME_PATH/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai" ]; then
# MAGIC   echo "Downloading GRCh38 FASTA index..."
# MAGIC   wget -q https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai -P /local_disk0/tmp_ref/
# MAGIC   cp /local_disk0/tmp_ref/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai $GENOME_PATH/
# MAGIC   echo "Downloaded GRCh38 FASTA index"
# MAGIC else
# MAGIC   echo "GRCh38 FASTA index already exists, skipping"
# MAGIC fi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download BWA index files (required by Parabricks fq2bam)

# COMMAND ----------

# MAGIC %sh
# MAGIC BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome"
# MAGIC REF_BASE="GRCh38_full_analysis_set_plus_decoy_hla.fa"
# MAGIC
# MAGIC for ext in bwt sa ann amb pac; do
# MAGIC   if [ ! -f "$GENOME_PATH/${REF_BASE}.${ext}" ]; then
# MAGIC     echo "Downloading BWA index: ${REF_BASE}.${ext}..."
# MAGIC     wget -q "${BASE_URL}/${REF_BASE}.${ext}" -P /local_disk0/tmp_ref/
# MAGIC     cp "/local_disk0/tmp_ref/${REF_BASE}.${ext}" "$GENOME_PATH/"
# MAGIC     echo "Downloaded ${REF_BASE}.${ext}"
# MAGIC   else
# MAGIC     echo "BWA index ${REF_BASE}.${ext} already exists, skipping"
# MAGIC   fi
# MAGIC done
# MAGIC echo "BWA index files complete"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download sample VCF from 1000 Genomes (for demo)

# COMMAND ----------

import os, subprocess

sample_vcf_dir = f"/Volumes/{catalog}/{schema}/gwas_data/sample_vcf"
os.makedirs(sample_vcf_dir, exist_ok=True)

vcf_filename = "ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"
vcf_dest = os.path.join(sample_vcf_dir, vcf_filename)

if not os.path.exists(vcf_dest):
    vcf_url = f"https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/{vcf_filename}"
    print(f"Downloading sample VCF to {vcf_dest}...")
    # Download to local disk first, then copy to Volume (avoids FUSE write issues)
    local_tmp = f"/local_disk0/tmp_ref/{vcf_filename}"
    os.makedirs("/local_disk0/tmp_ref", exist_ok=True)
    result = subprocess.run(["wget", "-q", "-O", local_tmp, vcf_url], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: wget failed: {result.stderr}")
    elif os.path.exists(local_tmp) and os.path.getsize(local_tmp) > 1000:
        import shutil
        shutil.copy2(local_tmp, vcf_dest)
        print(f"Downloaded sample VCF: {os.path.getsize(vcf_dest) / (1024*1024):.0f} MB")
    else:
        print(f"ERROR: Downloaded file is missing or too small")
else:
    print(f"Sample VCF already exists at {vcf_dest}, skipping")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy reference to DBFS (required by Glow for variant normalization)

# COMMAND ----------

# MAGIC %sh
# MAGIC DBFS_REF="/dbfs/genesis_workbench/gwas/reference/grch38"
# MAGIC mkdir -p $DBFS_REF
# MAGIC
# MAGIC if [ ! -f "$DBFS_REF/GRCh38_full_analysis_set_plus_decoy_hla.fa" ]; then
# MAGIC   cp $GENOME_PATH/GRCh38_full_analysis_set_plus_decoy_hla.fa $DBFS_REF/
# MAGIC   cp $GENOME_PATH/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai $DBFS_REF/
# MAGIC   echo "Reference genome copied to DBFS for Glow"
# MAGIC else
# MAGIC   echo "Reference genome already exists in DBFS, skipping"
# MAGIC fi

# COMMAND ----------

print("Reference genome setup complete:")
for f in os.listdir(genome_path):
    size_mb = os.path.getsize(os.path.join(genome_path, f)) / (1024 * 1024)
    print(f"  {f} ({size_mb:.0f} MB)")
