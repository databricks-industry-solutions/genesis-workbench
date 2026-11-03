# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_volume", "alphafold", "Model Volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_VOLUME = dbutils.widgets.get("model_volume")

# COMMAND ----------

import os
os.environ["CATALOG"] = CATALOG
os.environ["SCHEMA"] = SCHEMA
os.environ["MODEL_VOLUME"] = f"/Volumes/{CATALOG}/{SCHEMA}/{MODEL_VOLUME}"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{MODEL_VOLUME}")

# COMMAND ----------

# MAGIC %sh
# MAGIC set -euo pipefail
# MAGIC export TAR_OPTIONS="--no-same-owner"
# MAGIC mkdir -p /local_disk0/common/
# MAGIC wget -q --tries=5 --timeout=30 -P /local_disk0/common/ \
# MAGIC   https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/openstructure/stereo_chemical_props.txt
# MAGIC
# MAGIC mkdir -p $MODEL_VOLUME/datasets/common
# MAGIC cp /local_disk0/common/stereo_chemical_props.txt $MODEL_VOLUME/datasets/common/
