# Databricks notebook source
# MAGIC %md
# MAGIC ## Download MGNIFY, small_bfd, pdb70 and pdb_seqres datasets

# COMMAND ----------

# MAGIC %run ./download_setup

# COMMAND ----------

# MAGIC %sh
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/mgnify" ]; then
# MAGIC     echo "Downloading mgnify"
# MAGIC     cd /app/alphafold/scripts
# MAGIC     ./download_mgnify.sh /local_disk0/downloads
# MAGIC     cd /
# MAGIC     echo "Copying to $MODEL_VOLUME"
# MAGIC     cp -r /local_disk0/downloads/mgnify $MODEL_VOLUME/datasets/
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/small_bfd" ]; then
# MAGIC     echo "Downloading small_bfd"
# MAGIC     cd /app/alphafold/scripts
# MAGIC     ./download_small_bfd.sh /local_disk0/downloads
# MAGIC     cd /
# MAGIC     echo "Copying to $MODEL_VOLUME"
# MAGIC     cp -r /local_disk0/downloads/small_bfd $MODEL_VOLUME/datasets/
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/pdb70" ]; then
# MAGIC   echo "Downloading pdb70"
# MAGIC     cd /app/alphafold/scripts
# MAGIC     ./download_pdb70.sh /local_disk0/downloads
# MAGIC     cd /
# MAGIC     echo "Copying to $MODEL_VOLUME"
# MAGIC     cp -r /local_disk0/downloads/pdb70 $MODEL_VOLUME/datasets/
# MAGIC fi

# COMMAND ----------

# MAGIC %sh
# MAGIC if [ ! -d "$MODEL_VOLUME/datasets/pdb_seqres" ]; then
# MAGIC     echo "Downloading pdb_seqres"
# MAGIC
# MAGIC     cd /app/alphafold/scripts
# MAGIC     NEWLINE='SOURCE_URL="https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"'
# MAGIC     sed -i "s|^SOURCE_URL=.*|$NEWLINE|" download_pdb_seqres.sh
# MAGIC     ./download_pdb_seqres.sh /local_disk0/downloads
# MAGIC     cd /
# MAGIC     echo "Copying to $MODEL_VOLUME"
# MAGIC     cp -r /local_disk0/downloads/pdb_seqres $MODEL_VOLUME/datasets/
# MAGIC fi
