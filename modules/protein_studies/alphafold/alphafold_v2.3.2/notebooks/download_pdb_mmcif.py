# Databricks notebook source
# MAGIC %run ./download_setup

# COMMAND ----------

# MAGIC %sh
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/pdb_mmcif" ]; then
# MAGIC     echo "Downloading pdb_mmcif"
# MAGIC     cd /app/alphafold/scripts
# MAGIC     ./download_pdb_mmcif.sh /local_disk0/downloads
# MAGIC     cd /
# MAGIC     echo "Copying to $MODEL_VOLUME"
# MAGIC     cp -r /local_disk0/downloads/pdb_mmcif $MODEL_VOLUME/datasets/
# MAGIC fi

# COMMAND ----------


