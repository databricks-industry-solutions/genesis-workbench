# Databricks notebook source
# MAGIC %run ./download_setup

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/pdb_mmcif" ]; then
# MAGIC     export TAR_OPTIONS="--no-same-owner"
# MAGIC     echo "Downloading pdb_mmcif"
# MAGIC     cd /app/alphafold/scripts
# MAGIC     # Use HTTPS download script instead of rsync-based original —
# MAGIC     # rsync on port 33444 is blocked on AWS Databricks clusters.
# MAGIC     # The HTTPS script was created by download_setup.py.
# MAGIC     ./download_pdb_mmcif_https.sh /local_disk0/downloads
# MAGIC     cd /
# MAGIC     echo "Copying to $MODEL_VOLUME"
# MAGIC     cp -r /local_disk0/downloads/pdb_mmcif $MODEL_VOLUME/datasets/
# MAGIC fi

# COMMAND ----------


