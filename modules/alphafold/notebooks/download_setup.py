# Databricks notebook source
# MAGIC %md
# MAGIC This can be called with a %run command for other download scripts as this process is required for seperate download tasks

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_volume", "alphafold_cache_dir", "Model Volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_VOLUME = dbutils.widgets.get("model_volume")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{MODEL_VOLUME}")

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get update
# MAGIC apt-get --no-install-recommends -y install aria2

# COMMAND ----------

# MAGIC %sh 
# MAGIC mkdir -p /app
# MAGIC cd /app
# MAGIC git clone https://github.com/google-deepmind/alphafold.git
# MAGIC cd alphafold
# MAGIC git checkout v2.3.2

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /local_disk0
# MAGIC mkdir -p downloads

# COMMAND ----------

import os
os.environ["CATALOG"] = CATALOG
os.environ["SCHEMA"] = SCHEMA
os.environ["MODEL_VOLUME"] = f"/Volumes/{CATALOG}/{SCHEMA}/{MODEL_VOLUME}"

