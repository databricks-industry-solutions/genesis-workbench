# Databricks notebook source
# to-do 
# add a task to check the deployed endpoints can be called to make a KNN for subsample of data 
# maybe test using ai_query()

# COMMAND ----------



# COMMAND ----------

## avoid concurrency issues wrt multiple tasks writing to Vols

# Define the list of requirements
requirements = [
    "scimilarity==0.4.0",
    "typing_extensions>=4.14.0",
    "numpy==1.26.4",
    "pandas==1.5.3",
    "mlflow==2.22.0",
    "tbb>=2021.6.0",
    "uv"
]

# Install the required packages
for package in requirements:
    %pip install {package} --upgrade -v

dbutils.library.restartPython()

# COMMAND ----------

# Import numba after installing the required packages
import numba

# Set this at the beginning of your notebook/script
numba.config.THREADING_LAYER = 'workqueue'  # Most compatible option
# Other options: 'omp' (OpenMP) or 'tbb' (default)

# COMMAND ----------

# dbutils.widgets.removeAll()

# COMMAND ----------

## for nb devs -- these get overwritten wrt deployment args
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_mmt_core_test", "Schema") 

dbutils.widgets.text("model_name", "SCimilarity", "Model Name") ## use this as a prefix for the model name ?
# dbutils.widgets.text("experiment_name", "gwb_modules_scimilarity", "Experiment Name")
# dbutils.widgets.text("experiment_name", "gwb_modules", "Experiment Name") ## mlflow expt folder_name

# dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id") # ??
# dbutils.widgets.text("user_email", "may.merkletan@databricks.com", "User Id/Email")

dbutils.widgets.text("cache_dir", "scimilarity", "Cache dir") ## VOLUME NAME | MODEL_FAMILY 

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

MODEL_NAME = dbutils.widgets.get("model_name")
# EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
# USER_EMAIL = dbutils.widgets.get("user_email")
# SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")

CACHE_DIR = dbutils.widgets.get("cache_dir")

print(f"Cache dir: {CACHE_DIR}")
cache_full_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

CATALOG = CATALOG #"mmt"
DB_SCHEMA = SCHEMA #"tests" | "genesiswb"

# VOLUME_NAME | PROJECT 
MODEL_FAMILY = CACHE_DIR ## CACHE_DIR #"scimilarity"

# MODEL_NAME #"SCimilarity" 

print("CATALOG :", CATALOG)
print("DB_SCHEMA :", DB_SCHEMA)
print("MODEL_FAMILY :", MODEL_FAMILY)

# COMMAND ----------

# DBTITLE 1,Still require some Scimilarity + Dependencies imports
# from scimilarity import CellQuery, CellEmbedding
from scimilarity import align_dataset, lognorm_counts

import scanpy as sc
from scipy import sparse
import numpy as np
import pandas as pd

from collections.abc import MutableMapping  # 

# COMMAND ----------

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
# databricks_instance = "adb-830292400663869.9.azuredatabricks.net"
# endpoint_name = "mmt_scimilarity_gene_order"
# example_input = pd.DataFrame({"input": ["get_gene_order"]})

# result = score_model(databricks_instance, endpoint_name, example_input)
# display(result)

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()
# w.secrets.put_secret(scope="mmt", key="databricks_token", string_value="my-secret-value")

# COMMAND ----------

import os 
DATABRICKS_TOKEN = dbutils.secrets.get("mmt","databricks_token")
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# COMMAND ----------

databricks_instance = "adb-830292400663869.9.azuredatabricks.net"

# COMMAND ----------

# DBTITLE 1,gene_order
# Example usage
# databricks_instance = "adb-830292400663869.9.azuredatabricks.net"

# ## READ GeneOrder from GeneOrder file --> substitue with endpoint in testing nb/task
# def derive_gene_order() -> list[str]:    
#     return pd.read_csv(geneOrder_path, header=None).squeeze().tolist()
# gene_order = derive_gene_order()

def derive_gene_order(endpoint_name = "mmt_scimilarity_gene_order") -> list[str]:     
    
    example_input = pd.DataFrame({"input": ["get_gene_order"]})

    get_gene_order = score_model(databricks_instance, endpoint_name, example_input)
    
    return get_gene_order['predictions']

gene_order = derive_gene_order()
gene_order

# COMMAND ----------

# import pandas as pd
# import requests
# import json
# import os

# databricks_instance = "adb-830292400663869.9.azuredatabricks.net"
# endpoint_name = "mmt_scimilarity_gene_order"
# example_input = pd.DataFrame({"input": ["get_gene_order"]})

# # Convert DataFrame to the required JSON format
# example_input_json = json.dumps({"dataframe_split": json.loads(example_input.to_json(orient="split"))})

# # Define the function to score the model
# def score_model0(databricks_instance, endpoint_name, input_data):
#     url = f"https://{databricks_instance}/serving-endpoints/{endpoint_name}/invocations"
#     headers = {
#         "Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}",
#         "Content-Type": "application/json"
#     }
#     response = requests.post(url, headers=headers, data=input_data)
#     if response.status_code != 200:
#         raise Exception(f"Request failed with status code: {response.status_code}, {response.text}")
#     return response.json()

# # Call the function with the JSON input
# result = score_model0(databricks_instance, endpoint_name, example_input_json)
# print(result)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,read sample data + extract subsample
sampledata_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/data/adams_etal_2020/GSE136831_subsample.h5ad"

## READ sample czi dataset H5AD file + align + lognorm 
adams = sc.read(sampledata_path)

## use the gene_order from endpoint 
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

# COMMAND ----------

# DBTITLE 1,there are actually 3 sets
subsample.obs

# COMMAND ----------

X_vals2 = subsample[subsample.obs.index == "126138"].X
X_vals2

# COMMAND ----------

X_vals

# COMMAND ----------

combi_Xvals = [X_vals.toarray(),X_vals2.toarray()]
combi_Xvals

# COMMAND ----------

# {'error': 'HTTP 504', 'response_text': 'upstream request timeout'} -- scaling from zero takes a while...? 
# do we need a larger compute? 

# COMMAND ----------

example_input = pd.DataFrame([{'subsample_query_array': X_vals.toarray()[0].tolist() }]) ## required input formatting 

endpoint_name = "mmt_scimilarity_get_embedding"

get_embeddings = score_model(databricks_instance, endpoint_name, example_input)
get_embeddings['predictions'][0]['embedding'][0]

# COMMAND ----------

example2_input = pd.DataFrame([{'subsample_query_array': X_vals2.toarray()[0].tolist() }]) ## required input formatting 

endpoint_name = "mmt_scimilarity_get_embedding"

get_embeddings2 = score_model(databricks_instance, endpoint_name, example2_input)
get_embeddings2['predictions'][0]['embedding'][0]

# COMMAND ----------

def derive_embeddings(data_input, endpoint_name) -> list[str]:  

    ## data_input: X_vals
    example_input = pd.DataFrame([{'subsample_query_array': data_input.toarray()[0].tolist() }]) ## required input formatting 

    get_embeddings = score_model(databricks_instance, endpoint_name, example_input)
    return get_embeddings['predictions'][0]['embedding'][0]

# COMMAND ----------

cell_embedding = derive_embeddings(data_input = X_vals, endpoint_name = "mmt_scimilarity_get_embedding")

cell_embedding

# COMMAND ----------

## get cell embeddings --> substitue with endpoint in testing nb/task
# ce = CellEmbedding(model_path)
# cell_embeddings = ce.get_embeddings(X_vals)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,to figure out batch processing
# example_input2 = pd.DataFrame([{'subsample_query_array': x.tolist()} for x in combi_Xvals])
# example_input2

# COMMAND ----------

# score_model(databricks_instance, endpoint_name, example_input2)

# COMMAND ----------



# COMMAND ----------

# {'error': 'HTTP 504', 'response_text': 'upstream request timeout'}

# COMMAND ----------

## search nearest k neighbors --> substitue with endpoint in testing nb/task
# cq = CellQuery(model_path)
# nn_idxs,nn_dists,results_metadata  = cq.search_nearest(cell_embeddings, k=100) # where k is a parameter for N of nearest neighbors to search 

# results_metadata

# COMMAND ----------

knn_results = score_model(databricks_instance, 
                          endpoint_name="mmt_scimilarity_search_nearest", 
                          dataset=cell_embedding, 
                          params = {'k': 1000}
                         )

pd.DataFrame(knn_results['predictions'])
# knn_results['predictions'][0]

# COMMAND ----------

pd.DataFrame(knn_results['predictions'][0]['results_metadata'])

# COMMAND ----------



# COMMAND ----------


