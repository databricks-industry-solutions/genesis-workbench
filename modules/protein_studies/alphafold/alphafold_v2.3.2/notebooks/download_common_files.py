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
# MAGIC mkdir -p /local_disk0/common/
# MAGIC wget -q -P /local_disk0/common/ \
# MAGIC   https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
# MAGIC
# MAGIC mkdir -p $MODEL_VOLUME/datasets/common
# MAGIC cp /local_disk0/common/stereo_chemical_props.txt $MODEL_VOLUME/datasets/common/
