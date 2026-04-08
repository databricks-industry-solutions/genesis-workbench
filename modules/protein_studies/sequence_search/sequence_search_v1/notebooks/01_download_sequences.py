# Databricks notebook source
# MAGIC %md
# MAGIC ## Download UniRef90 FASTA
# MAGIC
# MAGIC Downloads the UniRef90 FASTA file from [UniProt](https://www.uniprot.org/help/downloads)
# MAGIC and saves it to a Unity Catalog Volume.
# MAGIC
# MAGIC UniRef90 contains ~150M protein sequences clustered at 90% sequence identity.
# MAGIC Source: https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz

# COMMAND ----------

# DBTITLE 1,Install biopython
# MAGIC %pip install biopython==1.86
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run utils (declares widgets, creates UC resources)
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Read widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume_name = dbutils.widgets.get("volume_name")

# COMMAND ----------

# DBTITLE 1,Create volume if not exists
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume_name}")

# COMMAND ----------

# DBTITLE 1,Download UniRef90 FASTA to UC Volume
import requests
import gzip

url = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
output_path = f"/Volumes/{catalog}/{schema}/{volume_name}/uniref90.fasta"

print(f"Downloading UniRef90 from {url}")
print(f"Output: {output_path}")
print("This may take 30-60 minutes depending on network speed (~35GB compressed)...")

response = requests.get(url, stream=True)
response.raise_for_status()

bytes_written = 0
with gzip.open(response.raw, "rt") as gz_file, open(output_path, "w") as out_file:
    for line in gz_file:
        out_file.write(line)
        bytes_written += len(line)
        if bytes_written % (1024 * 1024 * 1000) == 0:
            print(f"  Written ~{bytes_written // (1024 * 1024)} MB...")

print(f"Downloaded FASTA to {output_path}")

# COMMAND ----------

# DBTITLE 1,Verify downloaded FASTA file
from Bio import SeqIO

fasta_path = f"/Volumes/{catalog}/{schema}/{volume_name}/uniref90.fasta"

# Count first 100 records to verify
count = 0
for record in SeqIO.parse(fasta_path, "fasta"):
    count += 1
    if count == 1:
        print(f"First record ID: {record.id}")
        print(f"First record sequence: {record.seq[:50]}...")
    if count >= 100:
        break

print(f"Verified at least {count} records in FASTA file")
