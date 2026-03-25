# Databricks notebook source
# MAGIC %run ./download_setup

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/params" ]; then
# MAGIC   export TAR_OPTIONS="--no-same-owner"
# MAGIC   cd /app/alphafold/scripts
# MAGIC   ./download_alphafold_params.sh /local_disk0/downloads
# MAGIC   cd /
# MAGIC   echo "Copying to $MODEL_VOLUME"
# MAGIC   cp -r /local_disk0/downloads/params $MODEL_VOLUME/datasets/
# MAGIC fi
