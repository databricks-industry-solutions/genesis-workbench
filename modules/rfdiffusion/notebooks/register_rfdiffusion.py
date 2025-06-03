# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_name", "rfdiffusion", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "cache_dir", "Cache dir")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
USER_EMAIL = dbutils.widgets.get("user_email")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")
CACHE_DIR = dbutils.widgets.get("cache_dir")

print(f"Cache dir: {CACHE_DIR}")
cache_full_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

import os

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["SQL_WAREHOUSE"]=SQL_WAREHOUSE_ID
os.environ["IS_TOKEN_AUTH"]="Y"
os.environ["DATABRICKS_TOKEN"]=databricks_token

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download the RFDiffusion code and model weights to Unity Catalog

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /local_disk0/
# MAGIC git clone https://github.com/RosettaCommons/RFdiffusion.git
# MAGIC cd RFdiffusion
# MAGIC mkdir models && cd models
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
# MAGIC wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS protein_folding.rfdiffusion.repo_w_models")

# COMMAND ----------

# MAGIC %sh
# MAGIC cp -r /local_disk0/RFdiffusion /Volumes/protein_folding/rfdiffusion/repo_w_models/

# COMMAND ----------

# MAGIC %sh ls /Volumes/protein_folding/rfdiffusion/repo_w_models/RFdiffusion
