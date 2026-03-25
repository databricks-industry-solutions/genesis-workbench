# Databricks notebook source
# MAGIC %run ./download_setup

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/uniref90" ]; then
# MAGIC   export TAR_OPTIONS="--no-same-owner"
# MAGIC   echo "Downloading Uniref90"
# MAGIC   cd /app/alphafold/scripts
# MAGIC   ./download_uniref90.sh /local_disk0/downloads
# MAGIC   cd /
# MAGIC   echo "Copying to $MODEL_VOLUME"
# MAGIC   cp -r /local_disk0/downloads/uniref90 $MODEL_VOLUME/datasets/
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/uniref30" ]; then
# MAGIC   export TAR_OPTIONS="--no-same-owner"
# MAGIC   echo "Downloading Uniref30"
# MAGIC   cd /app/alphafold/scripts
# MAGIC   ./download_uniref30.sh /local_disk0/downloads
# MAGIC   cd /
# MAGIC   echo "Copying to $MODEL_VOLUME"
# MAGIC   cp -r /local_disk0/downloads/uniref30 $MODEL_VOLUME/datasets/
# MAGIC fi
