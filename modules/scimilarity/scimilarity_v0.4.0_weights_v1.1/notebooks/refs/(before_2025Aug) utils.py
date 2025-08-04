# Databricks notebook source
# MAGIC %md
# MAGIC #### SETUP Requirements: 
# MAGIC - Ref: https://genentech.github.io/scimilarity/install.html#conda-environment-setup
# MAGIC - Requires: `python=3.10`
# MAGIC - Databricks Runtime `14.3 LTS` supports `Python 3.10`
# MAGIC - MLflow: `2.22.0`; NB: `v3.0` has breaking changes wrt `runs` --> `models` vs `artifact_paths` --> `name` etc.

# COMMAND ----------

# DBTITLE 1,(initial) gwb_variablesNparams
# dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
# dbutils.widgets.text("schema", "dev_mmt_core_gwb", "Schema")
# # dbutils.widgets.text("model_name", "SCimilarity", "Model Name") ## use this as a prefix for the model name ?
# # dbutils.widgets.text("experiment_name", "gwb_modules_scimilarity", "Experiment Name")
# # dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id") # ??
# # dbutils.widgets.text("user_email", "may.merkletan@databricks.com", "User Id/Email")
# dbutils.widgets.text("cache_dir", "scimilarity", "Cache dir") ## VOLUME NAME | MODEL_FAMILY 

# CATALOG = dbutils.widgets.get("catalog")
# SCHEMA = dbutils.widgets.get("schema")
# # MODEL_NAME = dbutils.widgets.get("model_name")
# # EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
# # USER_EMAIL = dbutils.widgets.get("user_email")
# # SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")
# CACHE_DIR = dbutils.widgets.get("cache_dir")

# print(f"Cache dir: {CACHE_DIR}")
# cache_full_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
# print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

# DBTITLE 1,Requirements & noted version
# scimilarity==0.4.0
# typing_extensions>=4.14.0 #>=4.0.0 
## scanpy==1.11.2
## numcodecs==0.13.1
## zarr>=2.6.1
## numpy==1.26.4
## pandas==1.5.3
# mlflow==2.22.0 ## pin to this for now since v3 has breaking changes... 
# tbb>=2021.6.0
# uv

# COMMAND ----------

# DBTITLE 1,CATALOG | SCHEMA | VOLS
## causes concurrency conflicts wrt multiple tasks writing to Vols 

# import os

# # Create a requirements.txt file with the necessary dependencies
# requirements = """
# scimilarity==0.4.0
# typing_extensions>=4.14.0
# numpy==1.26.4
# pandas==1.5.3
# mlflow
# tbb>=2021.6.0
# uv
# """

# CATALOG = CATALOG #"mmt"
# # DB_SCHEMA = "genesiswb"
# DB_SCHEMA = SCHEMA #"tests"

# # VOLUME_NAME | PROJECT 
# MODEL_FAMILY = CACHE_DIR #"scimilarity"

# # Define the volumes path
# # scimilarity_ws_requirements_path = "/Volumes/mmt/genesiswb/scimilarity/workspace_requirements/requirements.txt"
# scimilarity_ws_requirements_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/workspace_requirements/requirements.txt"

# # Create the directory if it does not exist
# os.makedirs(os.path.dirname(scimilarity_ws_requirements_path), exist_ok=True)

# # Write the requirements to a file in the volumes path
# with open(scimilarity_ws_requirements_path, 'w') as f:
#     f.write(requirements)

# COMMAND ----------

# DBTITLE 1,check requirements.txt
# !cat /Volumes/mmt/genesiswb/scimilarity/workspace_requirements/requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ##### pip install `requirements` 
# MAGIC <!-- | `requirements.txt`  -->
# MAGIC <!-- - to Volumes for easier `sys.append()` -->

# COMMAND ----------

# DBTITLE 1,notebook -- pip install requirements.txt
# # Install required packages/dependencies 
# %pip install -r {scimilarity_ws_requirements_path} --upgrade -v

# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,[gwb] pip install from requirements list
## avoid concurrency issues wrt multiple tasks writing to Vols

# Define the list of requirements
requirements = [
    "scimilarity==0.4.0",
    "typing_extensions>=4.14.0",
    "scanpy==1.11.2", #
    "numcodecs==0.13.1", #
    "numpy==1.26.4",
    "pandas==1.5.3",
    "mlflow==2.22.0",
    "cloudpickle==2.0.0", #
    "tbb>=2021.6.0",
    "uv"
]

# Install the required packages
for package in requirements:
    %pip install {package} --upgrade -v

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

# DBTITLE 1,? Ignore warning...
# import warnings

## Ignore specific Numba warning
# warnings.filterwarnings("ignore", message="The TBB threading layer requires TBB version 2021 update 6 or later") 
# ## theoretically installed? 

# import warnings
# warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Unity Catalog / Volumes Paths  

# COMMAND ----------

# DBTITLE 1,gwb_variablesNparams
## for nb devs -- these get overwritten wrt deployment args
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_mmt_core_test", "Schema") 

dbutils.widgets.text("model_name", "SCimilarity", "Model Name") ## use this as a prefix for the model name ?
dbutils.widgets.text("experiment_name", "gwb_modules_scimilarity", "Experiment Name")
# dbutils.widgets.text("experiment_name", "gwb_modules", "Experiment Name") ## mlflow expt folder_name

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

# DBTITLE 1,CATALOG | SCHEMA | VOLS
CATALOG = CATALOG #"mmt"
DB_SCHEMA = SCHEMA #"tests" | "genesiswb"

# VOLUME_NAME | PROJECT 
MODEL_FAMILY = CACHE_DIR ## CACHE_DIR #"scimilarity"

# MODEL_NAME #"SCimilarity" 

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

# DBTITLE 1,check
# display(dbutils.fs.ls("/Volumes/genesis_workbench/dev_mmt_core_gwb/scimilarity/model/"))


# COMMAND ----------

# MAGIC %md
# MAGIC ##### `Workspace User Paths`

# COMMAND ----------

# Get the workspace / user path
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
user_path = f"/Users/{user}"
workspace_user_path = f"/Workspace{user_path}"

print("user_path :", user_path)
print("workspace_user_path: ", f'{workspace_user_path}')

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
adams_ipf = lognorm[lognorm.obs["Disease"] == "IPF"].copy()

adams_myofib = adams_ipf[
                          adams_ipf.obs["celltype_name"] == "myofibroblast cell"
                        ].copy()

## Extract list for sample_ref
subsample = adams_myofib[
                          adams_myofib.obs["sample"] == "DS000011735-GSM4058950" # sample ref 
                        ].copy()

## extract specific index in subsample 
query_cell = subsample[subsample.obs.index == "123942"]

## extract subsample query (1d array or list)
X_vals: sparse.csr_matrix = query_cell.X

## get cell embeddings --> substitue with endpoint in testing nb/task
ce = CellEmbedding(model_path)
cell_embeddings = ce.get_embeddings(X_vals)

## search nearest k neighbors --> substitue with endpoint in testing nb/task
cq = CellQuery(model_path)
nn_idxs,nn_dists,results_metadata  = cq.search_nearest(cell_embeddings, k=100) # where k is a parameter for N of nearest neighbors to search 

# results_metadata

print("Initial Data Processing using SCimilarity Complete")

# COMMAND ----------

# DBTITLE 1,check / review outputs
# adams_myofib.obs#["sample"].unique()
# subsample.obs
# X_vals
# cell_embeddings | # cell_embeddings.shape #(1, 128)
# nn_idxs,nn_dists,results_metadata 

# results_metadata

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initiate `HelperFunctions`

# COMMAND ----------

# DBTITLE 1,Define add_model_alias
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
import time

def add_model_alias(full_model_name, alias="Champion"):
    # Define the model name
    # full_model_name = f"{CATALOG}.{DB_SCHEMA}.{model_name}"
    # full_model_name = f"{CATALOG}.{DB_SCHEMA}.{MODEL_NAME}"

    # Initialize the MLflow client
    client = MlflowClient()

    # Fetch the latest model version
    model_version_infos = client.search_model_versions(f"name='{full_model_name}'")

    # Check if there are any model versions available
    if not model_version_infos:
        raise ValueError(f"No versions found for model '{full_model_name}'")

    latest_model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])

    # Set alias for the latest model version
    client.set_registered_model_alias(
        name=full_model_name,
        alias=alias,
        version=latest_model_version
    )

    # Function to wait until the model is ready
    def wait_until_ready(full_model_name, model_version, alias, timeout=300):
        start_time = time.time()
        while time.time() - start_time < timeout:
            model_version_info = client.get_model_version(name=full_model_name, version=model_version)
            if model_version_info.status == "READY":
                print(f"Model {full_model_name} version {model_version} with alias '{alias}' is ready.")
                return
            time.sleep(5)
        raise TimeoutError(f"Model {full_model_name} version {model_version} with alias '{alias}' did not become ready within {timeout} seconds.")

    # Wait until the model is ready
    wait_until_ready(full_model_name, latest_model_version, alias)

## Example usage
# add_model_alias(model_name, alias="Champion")
# add_model_alias("SCimilarity_GeneOrder")

# COMMAND ----------

# DBTITLE 1,get_latest_model_version
from mlflow.tracking import MlflowClient

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

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,get_endpoint_link
# databricks_instance = "e2-demo-field-eng.cloud.databricks.com"  # Replace with your Databricks instance URL -- this is NOT workspace_url

def get_endpoint_link(databricks_instance, model_name, latest_model_version):
  # Construct the link to the endpoint
  # databricks_instance = "e2-demo-field-eng.cloud.databricks.com"  # Replace with your Databricks instance URL -- this is NOT workspace_url
  
  model_name = model_name  # Replace with your model name
  model_version = latest_model_version  # Replace with your model version

  endpoint_url = f"https://{databricks_instance}/ml/endpoints/{endpoint_name}"

  print(f"Endpoint link: {endpoint_url}")
  return endpoint_url

# COMMAND ----------

# DBTITLE 1,check_endpoint_status
import time
from requests.exceptions import HTTPError
from mlflow.exceptions import RestException
from mlflow.deployments import get_deploy_client

def check_endpoint_status(endpoint_name, max_checks=20, sleep_time=30):
    client = get_deploy_client("databricks")

    # Verify the endpoint exists
    try:
        endpoint_status = client.get_endpoint(endpoint_name)
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            print(f"Endpoint with name '{endpoint_name}' does not exist.")
            return
        else:
            raise

    # Wait for the endpoint to be ready
    check_count = 0  # Initialize check counter

    while check_count < max_checks:
        endpoint_status = client.get_endpoint(endpoint_name)
        state = endpoint_status["state"]
        if state == {'ready': 'READY', 'config_update': 'NOT_UPDATING', 'suspend': 'NOT_SUSPENDED'}:
            print(f"Endpoint {endpoint_name} is READY!")
            break    
        elif state["config_update"] == "UPDATING":
            print(f"Endpoint {endpoint_name} config_update is updating...")
        elif state["config_update"] == "IN_PROGRESS":
            print(f"Endpoint {endpoint_name} config_update is in progress...")
            print(endpoint_status['pending_config']['served_models'][0]['state'])        
        else:
            print(f"Endpoint {endpoint_name} is in state: {state}")
        
        check_count += 1
        time.sleep(sleep_time)  # Wait for sleep_time seconds before checking again

    if check_count == max_checks:
        print(f"Reached maximum number of checks ({max_checks}). Exiting loop.")

## Example usage
# check_endpoint_status(endpoint_name, max_checks=20, sleep_time=30)

# COMMAND ----------

# DBTITLE 1,[1] create_serving_json | score_model
# import os 
# import json
# import requests

## databricks_instance = "e2-demo-field-eng.cloud.databricks.com"

# def create_serving_json(data):
#     if isinstance(data, dict):
#         return {'inputs': {name: data[name].tolist() if hasattr(data[name], 'tolist') else data[name] for name in data.keys()}}
      
#     elif isinstance(data, list):
#         return {'inputs': data}



# def score_model(databricks_instance, endpoint_name, dataset):
#     # url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/gwb_SCimilarity_GeneOrder_mmt/invocations'
#     url = f'https://{databricks_instance}/serving-endpoints/{endpoint_name}/invocations'
    
#     headers = {
#         'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
#         'Content-Type': 'application/json'
#     }
#     ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_serving_json(dataset)

#     data_json = json.dumps(ds_dict, allow_nan=True)
    
#     response = requests.request(method='POST', headers=headers, url=url, data=data_json)
#     return response.json()

# COMMAND ----------

# DBTITLE 1,[2] create_serving_json | score_model
import os
import json
import requests
import pandas as pd

def create_serving_json(data, params=None):
    result = {}
    if isinstance(data, dict):
        result['inputs'] = {name: data[name].tolist() if hasattr(data[name], 'tolist') else data[name] for name in data.keys()}
    elif isinstance(data, list):
        result['inputs'] = data
    if params:
        result['params'] = params
    return result

def score_model(databricks_instance, endpoint_name, dataset, params=None):
    # url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/gwb_SCimilarity_SearchNearest_mmt/invocations'
    url = f'https://{databricks_instance}/serving-endpoints/{endpoint_name}/invocations'
    headers = {
        'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        'Content-Type': 'application/json'
    }
    
    if isinstance(dataset, pd.DataFrame):
        ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
        if params:
            ds_dict['params'] = params
    else:
        ds_dict = create_serving_json(dataset, params)
    
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.post(url, headers=headers, data=data_json)
    
    ### Debug information
    # print(f"Response status code: {response.status_code}")
    # print(f"Response headers: {response.headers}")
    # print(f"Response text: {response.text}")
    
    # Check if the response is successful and contains JSON
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            print("Response is not valid JSON")
            return {"error": "Invalid JSON response", "response_text": response.text}
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {"error": f"HTTP {response.status_code}", "response_text": response.text}


## Example usage

# DATABRICKS_TOKEN = dbutils.secrets.get("<scope>", "<secret-key>")
# os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# databricks_instance = "e2-demo-field-eng.cloud.databricks.com"
# databricks_instance = "adb-830292400663869.9.azuredatabricks.net"

# endpoint_name = "mmt_scimilarity_gene_order"
# example_input = pd.DataFrame({"input": ["get_gene_order"]})

# result = score_model(databricks_instance, endpoint_name, example_input)
# display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,IF NEEDED: reset-delete-modelNversions
import mlflow
from mlflow.tracking.client import MlflowClient
from typing import Optional, List

def delete_model_versions(CATALOG, DB_SCHEMA, model_name, delete_all=False, specific_versions:Optional[List[int]]=None):
    """
    Delete all versions or specific versions of a registered model.

    Parameters:
    model_name (str): The name of the model to delete.
    delete_all (bool): If True, delete all versions of the model. If False, delete specific versions.
    specific_versions (list): List of specific versions to delete if delete_all is False.
    """
    full_model_name = f"{CATALOG}.{DB_SCHEMA}.{model_name}"

    # Initialize the MLflow client
    client = MlflowClient()

    # Fetch all versions of the model
    model_version_infos = client.search_model_versions(f"name='{full_model_name}'")

    if delete_all:
        # Delete all versions of the model
        for model_version_info in model_version_infos:
            client.delete_model_version(name=full_model_name, version=model_version_info.version)
        
        # Delete the registered model
        client.delete_registered_model(name=full_model_name)
        print(f"Deleted model '{full_model_name}' and all its versions.")

    else:
        if specific_versions is None:
            raise ValueError("specific_versions must be provided if delete_all is False")
        # Delete specific versions of the model
        for version in specific_versions:
            client.delete_model_version(name=full_model_name, version=version)

        print(f"Deleted model '{full_model_name}' and version {specific_versions}.")

# Example usage
# delete_model_versions("SCimilarity_GeneOrder", delete_all=True)
# delete_model_versions("SCimilarity_GeneOrder", delete_all=False, specific_versions=["3", "4", "5"])  


# COMMAND ----------

print("HelperFunctions Initiated!")
