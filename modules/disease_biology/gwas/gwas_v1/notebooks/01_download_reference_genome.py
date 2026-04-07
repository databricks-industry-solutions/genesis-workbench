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
# MAGIC   echo "Downloading GRCh38 index..."
# MAGIC   wget -q https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai -P /local_disk0/tmp_ref/
# MAGIC   cp /local_disk0/tmp_ref/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai $GENOME_PATH/
# MAGIC   echo "Downloaded GRCh38 index"
# MAGIC else
# MAGIC   echo "GRCh38 index already exists, skipping"
# MAGIC fi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download sample VCF from 1000 Genomes (for demo)

# COMMAND ----------

# MAGIC %sh
# MAGIC SAMPLE_VCF_DIR="/Volumes/$CATALOG/$SCHEMA/gwas_data/sample_vcf"
# MAGIC mkdir -p $SAMPLE_VCF_DIR
# MAGIC
# MAGIC if [ ! -f "$SAMPLE_VCF_DIR/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz" ]; then
# MAGIC   echo "Downloading sample VCF (chr6) from 1000 Genomes..."
# MAGIC   wget -q ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz -P $SAMPLE_VCF_DIR/
# MAGIC   echo "Downloaded sample VCF"
# MAGIC else
# MAGIC   echo "Sample VCF already exists, skipping"
# MAGIC fi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy reference to DBFS (required by Glow for variant normalization)

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /dbfs/genesis_workbench/gwas/reference/grch38
# MAGIC cp $GENOME_PATH/GRCh38_full_analysis_set_plus_decoy_hla.fa /dbfs/genesis_workbench/gwas/reference/grch38/
# MAGIC cp $GENOME_PATH/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai /dbfs/genesis_workbench/gwas/reference/grch38/
# MAGIC echo "Reference genome copied to DBFS for Glow"

# COMMAND ----------

print("Reference genome setup complete:")
for f in os.listdir(genome_path):
    size_mb = os.path.getsize(os.path.join(genome_path, f)) / (1024 * 1024)
    print(f"  {f} ({size_mb:.0f} MB)")
