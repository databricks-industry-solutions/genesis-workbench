# Databricks notebook source
# MAGIC %md
# MAGIC ## Download FASTA file from UniProt
# MAGIC
# MAGIC Downloads the Swiss-Prot FASTA file from [UniProt](https://www.uniprot.org/help/downloads)
# MAGIC and saves it to a Unity Catalog Volume.
# MAGIC
# MAGIC Source: https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz

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

# DBTITLE 1,Download FASTA file to UC Volume
import requests
import gzip

url = "https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
output_path = f"/Volumes/{catalog}/{schema}/{volume_name}/uniprot_sprot.fasta"

response = requests.get(url, stream=True)
response.raise_for_status()

with gzip.open(response.raw, "rt") as gz_file, open(output_path, "w") as out_file:
    for line in gz_file:
        out_file.write(line)

print(f"Downloaded FASTA to {output_path}")

# COMMAND ----------

# DBTITLE 1,Verify downloaded FASTA file
from Bio import SeqIO

fasta_path = f"/Volumes/{catalog}/{schema}/{volume_name}/uniprot_sprot.fasta"

records = list(SeqIO.parse(fasta_path, "fasta"))

print(f"Number of sequences: {len(records)}")
print(f"First record ID: {records[0].id}")
print(f"First record sequence: {records[0].seq[:50]}")
