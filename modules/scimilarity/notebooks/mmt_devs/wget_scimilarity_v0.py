# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_mmt_core_gwb", "Schema")
dbutils.widgets.text("model_name", "scimilarity", "Model Name") ## use this as a prefix for the model name ?
dbutils.widgets.text("experiment_name", "gwb_modules_mmt", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id") # ??
dbutils.widgets.text("user_email", "may.merkletan@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "scimilarity", "Cache dir") ## VOLUME NAME | MODEL_FAMILY 

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

CATALOG = "mmt"
DB_SCHEMA = "genesiswb"

# VOLUME_NAME | PROJECT 
MODEL_FAMILY = "scimilarity"

# # Create widgets for catalog, db_schema, and model_family
# dbutils.widgets.text("catalog", "mmt")#, "CATALOG")
# dbutils.widgets.text("db_schema", "genesiswb")#, "DB_SCHEMA")
# dbutils.widgets.text("model_family", "scimilarity")#, "MODEL_FAMILY")

# # Get the values from the widgets
# CATALOG = dbutils.widgets.get("catalog")
# DB_SCHEMA = dbutils.widgets.get("db_schema")
# MODEL_FAMILY = dbutils.widgets.get("model_family")

print("CATALOG :", CATALOG)
print("DB_SCHEMA :", DB_SCHEMA)
print("MODEL_FAMILY :", MODEL_FAMILY)

# COMMAND ----------

# DBTITLE 1,Model File Paths
model_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/model_v1.1"
geneOrder_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/model_v1.1/gene_order.tsv"
sampledata_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"

print("model_path :", model_path)
print("geneOrder_path :", geneOrder_path)
print("sampledata_path :", sampledata_path)

# COMMAND ----------

# DBTITLE 1,get model & data files
## using serverless compute for this

## REF https://genentech.github.io/scimilarity/install.html

# COMMAND ----------

# Downloading the pretrained models
# You can download the following pretrained models for use with SCimilarity from Zenodo: https://zenodo.org/records/10685499

# COMMAND ----------

# Create the target directory if it doesn't exist
# !mkdir -p /Volumes/mmt/genesiswb/scimilarity/downloads
!mkdir -p {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/downloads"}

# COMMAND ----------

# !ls -lah /Volumes/mmt/genesiswb/scimilarity/downloads
!ls -lah {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/downloads"}

# COMMAND ----------

# DBTITLE 1,download model
# MAGIC %sh
# MAGIC wget -O {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/downloads/model_v1.1.tar.gz"} http://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1 --verbose
# MAGIC # wget -O /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz http://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1 --verbose

# COMMAND ----------

# !ls -lah /Volumes/mmt/genesiswb/scimilarity/downloads
!ls -lah {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/downloads"}

# COMMAND ----------



# COMMAND ----------

# Create the target directory if it doesn't exist
# !mkdir -p /Volumes/mmt/genesiswb/scimilarity/model
!mkdir -p {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model"}

# COMMAND ----------

# !ls -lah -p /Volumes/mmt/genesiswb/scimilarity/model
!ls -lah -p {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model"}

# COMMAND ----------

# MAGIC %sh
# MAGIC chmod u+rx {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/downloads/model_v1.1.tar.gz"}
# MAGIC # chmod u+rx /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz

# COMMAND ----------

# DBTITLE 1,untargzip to model folder
# MAGIC %sh
# MAGIC tar --no-same-owner -xzvf {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/downloads/model_v1.1.tar.gz"} -C /Volumes/mmt/genesiswb/scimilarity/model --verbose
# MAGIC # tar --no-same-owner -xzvf /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz -C /Volumes/mmt/genesiswb/scimilarity/model --verbose

# COMMAND ----------

# %sh
# tar --no-same-owner -xzvf /Volumes/mmt/genesiswb/scimilarity/downloads/model_v1.1.tar.gz -C /Volumes/mmt/genesiswb/scimilarity/model --wildcards '*/gene_order.tsv'

# COMMAND ----------

# !ls -lah -p /Volumes/mmt/genesiswb/scimilarity/model/*
!ls -lah -p {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/*"}

# COMMAND ----------



# COMMAND ----------

# Query data. We will use Adams et al., 2020 healthy and IPF lung scRNA-seq data. Download tutorial data.
# https://zenodo.org/records/13685881

# COMMAND ----------

# Create the target directory if it doesn't exist
# !mkdir -p /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020
!mkdir -p {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020"}

# COMMAND ----------

# !ls -lah /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020
!ls -lah {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020"}

# COMMAND ----------

# DBTITLE 1,get sample data
# MAGIC %sh
# MAGIC wget -O {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"} https://zenodo.org/records/13685881/files/GSE136831_subsample.h5ad?download=1 --verbose
# MAGIC # wget -O /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020/GSE136831_subsample.h5ad https://zenodo.org/records/13685881/files/GSE136831_subsample.h5ad?download=1 --verbose

# COMMAND ----------

# !ls -lah /Volumes/mmt/genesiswb/scimilarity/data/adams_etal_2020
!ls -lah {f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020"}
