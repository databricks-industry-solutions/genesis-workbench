# Databricks notebook source
# MAGIC %run ./download_setup

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/uniprot" ]; then
# MAGIC   export TAR_OPTIONS="--no-same-owner"
# MAGIC   echo "Downloading uniprot"
# MAGIC   cd /app/alphafold/scripts
# MAGIC   ./download_uniprot.sh /local_disk0/downloads
# MAGIC   cd /
# MAGIC   echo "Copying to $MODEL_VOLUME"
# MAGIC   cp -r /local_disk0/downloads/uniprot $MODEL_VOLUME/datasets/
# MAGIC fi
