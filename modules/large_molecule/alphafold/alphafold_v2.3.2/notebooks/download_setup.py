# Databricks notebook source
# MAGIC %md
# MAGIC This can be called with a %run command for other download scripts as this process is required for seperate download tasks

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_volume", "alphafold", "Model Volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_VOLUME = dbutils.widgets.get("model_volume")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{MODEL_VOLUME}")

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC apt-get update
# MAGIC apt-get --no-install-recommends -y install aria2

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC mkdir -p /app
# MAGIC cd /app
# MAGIC git clone https://github.com/google-deepmind/alphafold.git
# MAGIC cd alphafold
# MAGIC git checkout v2.3.2
# MAGIC
# MAGIC # Patch FTP URLs to HTTPS — v2.3.2 scripts use FTP which is blocked on
# MAGIC # many cloud environments (AWS Databricks clusters block outbound FTP).
# MAGIC # The same files are available over HTTPS from the same hosts.
# MAGIC cd scripts
# MAGIC sed -i 's|ftp://ftp.ebi.ac.uk/|https://ftp.ebi.ac.uk/|g' download_uniprot.sh
# MAGIC sed -i 's|ftp://ftp.uniprot.org/|https://ftp.uniprot.org/|g' download_uniref90.sh
# MAGIC sed -i 's|ftp://ftp.wwpdb.org/|https://ftp.wwpdb.org/|g' download_pdb_mmcif.sh download_pdb_seqres.sh
# MAGIC
# MAGIC # Replace rsync-based PDB mmCIF download with HTTPS — rsync on port 33444
# MAGIC # is also blocked in most cloud environments. Use wget against the EBI
# MAGIC # HTTPS mirror instead. This is slower than rsync but works through firewalls.

# COMMAND ----------

# Create the HTTPS-based PDB mmCIF download script via Python to avoid
# heredoc quoting issues in %sh MAGIC cells.
script = r'''#!/bin/bash
set -e
if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi
DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/pdb_mmcif"
RAW_DIR="${ROOT_DIR}/raw"
MMCIF_DIR="${ROOT_DIR}/mmcif_files"
BASE_URL="https://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/divided/mmCIF"

echo "Downloading PDB mmCIF files over HTTPS from EBI mirror..."
mkdir -p "${RAW_DIR}" "${MMCIF_DIR}"

# Get list of 2-letter subdirectories from the index page
SUBDIRS=$(curl -s "${BASE_URL}/" | grep -o 'href="[a-z0-9][a-z0-9]/"' | cut -d'"' -f2 | tr -d '/' | sort -u)
TOTAL=$(echo "$SUBDIRS" | wc -w)
echo "Found $TOTAL subdirectories to download."
COUNT=0
for subdir in $SUBDIRS; do
  COUNT=$((COUNT + 1))
  mkdir -p "${RAW_DIR}/${subdir}"

  # Get list of .gz files in this subdirectory
  FILES=$(curl -s "${BASE_URL}/${subdir}/" | grep -o 'href="[^"]*\.cif\.gz"' | sed 's/href="//;s/"$//' | sort -u)
  NFILES=$(echo "$FILES" | grep -c . || true)
  echo "[$COUNT/$TOTAL] Downloading ${subdir}/ ($NFILES files)..."

  # Build aria2c input file for batch download
  INPUT_FILE="/tmp/aria2_${subdir}.txt"
  > "${INPUT_FILE}"
  for f in $FILES; do
    echo "${BASE_URL}/${subdir}/${f}" >> "${INPUT_FILE}"
    echo "  dir=${RAW_DIR}/${subdir}" >> "${INPUT_FILE}"
  done

  # Download all files in this subdir with aria2c (16 connections)
  if [ -s "${INPUT_FILE}" ]; then
    aria2c -i "${INPUT_FILE}" -j 16 --auto-file-renaming=false \
      --allow-overwrite=true --console-log-level=warn 2>&1 || true
  fi
  rm -f "${INPUT_FILE}"
done

echo "Verifying downloads..."
FILE_COUNT=$(find "${RAW_DIR}" -name "*.cif.gz" | wc -l)
echo "Downloaded $FILE_COUNT .cif.gz files."

if [ "$FILE_COUNT" -eq 0 ]; then
  echo "ERROR: No files were downloaded. Check network connectivity."
  exit 1
fi

echo "Unzipping all mmCIF files..."
find "${RAW_DIR}/" -type f -iname "*.gz" -exec gunzip {} +

echo "Flattening all mmCIF files..."
mkdir -p "${MMCIF_DIR}"
for subdir in "${RAW_DIR}"/*; do
  if [ -d "$subdir" ]; then
    mv "${subdir}/"*.cif "${MMCIF_DIR}" 2>/dev/null || true
  fi
done

# Clean up raw directory
rm -rf "${RAW_DIR}"

aria2c "https://files.wwpdb.org/pub/pdb/data/status/obsolete.dat" --dir="${ROOT_DIR}"
echo "PDB mmCIF download complete. $(ls "${MMCIF_DIR}" | wc -l) structures."
'''

with open("/app/alphafold/scripts/download_pdb_mmcif_https.sh", "w") as f:
    f.write(script)

import os
os.chmod("/app/alphafold/scripts/download_pdb_mmcif_https.sh", 0o755)
print("Created download_pdb_mmcif_https.sh")

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC cd /local_disk0
# MAGIC mkdir -p downloads

# COMMAND ----------

import os
os.environ["CATALOG"] = CATALOG
os.environ["SCHEMA"] = SCHEMA
os.environ["MODEL_VOLUME"] = f"/Volumes/{CATALOG}/{SCHEMA}/{MODEL_VOLUME}"

