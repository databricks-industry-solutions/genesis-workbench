# Databricks notebook source
# MAGIC %md
# MAGIC #### SETUP Requirements: 
# MAGIC - Ref: https://genentech.github.io/scimilarity/install.html#conda-environment-setup
# MAGIC - Requires: `python=3.10`
# MAGIC - Databricks Runtime `14.3 LTS` supports `Python 3.10`
# MAGIC - MLflow: `2.22.0`; NB: `v3.0` has breaking changes wrt `runs` --> `models` vs `artifact_paths` --> `name` etc.

# COMMAND ----------

# DBTITLE 1,[gwb] pip install from requirements list
#install all dependencies
%pip install scimilarity==0.4.0 typing_extensions>=4.14.0 scanpy==1.11.2 numcodecs==0.13.1 numpy==1.26.4 pandas==1.5.3 mlflow==2.22.0 cloudpickle==2.0.0 tbb>=2021.6.0 uv 

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configs etc.

# COMMAND ----------

# DBTITLE 1,(?) Specify Threading Option
# Import numba after installing the required packages
import numba

# Set this at the beginning of your notebook/script
numba.config.THREADING_LAYER = 'workqueue'  # Most compatible option
# Other options: 'omp' (OpenMP) or 'tbb' (default)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Unity Catalog / Volumes Paths  

# COMMAND ----------

# DBTITLE 1,gwb_variablesNparams
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "SCimilarity", "Model Name") 
dbutils.widgets.text("experiment_name", "gwb_modules_scimilarity", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id") 
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "scimilarity", "Cache dir")

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

# DBTITLE 1,CATALOG | SCHEMA | VOLS
CATALOG = CATALOG 
DB_SCHEMA = SCHEMA 
MODEL_FAMILY = CACHE_DIR 

print("CATALOG :", CATALOG)
print("DB_SCHEMA :", DB_SCHEMA)
print("MODEL_FAMILY :", MODEL_FAMILY)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `Model` / `geneOrder` / `sampledata` 

# COMMAND ----------

# DBTITLE 1,Model File Paths
model_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/model_v1.1"
geneOrder_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/model/model_v1.1/gene_order.tsv"
sampledata_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"

print("model_path :", model_path)
print("geneOrder_path :", geneOrder_path)
print("sampledata_path :", sampledata_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initial (Setup) Data Processing 
# MAGIC _use as inputs to local initialized model + model context_

# COMMAND ----------

# DBTITLE 1,Initial Data Processing for Deployment/Debugging/Testing
from scimilarity import CellQuery, CellEmbedding
from scimilarity import align_dataset, lognorm_counts

import scanpy as sc
from scipy import sparse
import numpy as np
import pandas as pd

from collections.abc import MutableMapping  # 

## READ GeneOrder from GeneOrder file --> substitue with endpoint in testing nb/task
def derive_gene_order() -> list[str]:    
    return pd.read_csv(geneOrder_path, header=None).squeeze().tolist()

gene_order = derive_gene_order()

## READ sample czi dataset H5AD file + align + lognorm 
adams = sc.read(sampledata_path)
aligned = align_dataset(adams, gene_order)  ## adams_aligned = align_dataset(adams, cq.gene_order); cq = CellQuery(model_path)
lognorm = lognorm_counts(aligned)

### Filter samle data to "Disease" | celltype: "myofibroblast cell" | sample_ref: "DS000011735-GSM4058950"  

disease_name = "IPF"
celltype_name = "myofibroblast cell" 
sample_refid = "DS000011735-GSM4058950"
subsample_refid = "123942"

diseasetype_ipf = lognorm[lognorm.obs["Disease"] == disease_name].copy()

celltype_myofib = diseasetype_ipf[
                                  diseasetype_ipf.obs["celltype_name"] == celltype_name
                                 ].copy()

## Extract list for sample_ref
celltype_sample = celltype_myofib[
                                  celltype_myofib.obs["sample"] == sample_refid 
                                 ].copy()

## extract specific index in celltype_sample 
celltype_subsample = celltype_sample[celltype_sample.obs.index == subsample_refid]

## extract celltype_sample query (1d array or list)
X_vals: sparse.csr_matrix = celltype_subsample.X

## get cell embeddings --> substitue with endpoint in testing nb/task
ce = CellEmbedding(model_path)
cell_embeddings = ce.get_embeddings(X_vals)

## search nearest k neighbors --> substitue with endpoint in testing nb/task
cq = CellQuery(model_path)
nn_idxs,nn_dists,results_metadata  = cq.search_nearest(cell_embeddings, k=100) # where k is a parameter for N of nearest neighbors to search 

print("Initial Data Processing using SCimilarity Complete")

# COMMAND ----------

# DBTITLE 1,get_latest_model_version
import mlflow
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

def get_model_uri(model_name):
  return f"models:/{model_name}/{get_latest_model_version(model_name)}"


def set_mlflow_experiment(experiment_tag, user_email):    
    w = WorkspaceClient()
    mlflow_experiment_base_path = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(experiment_path)