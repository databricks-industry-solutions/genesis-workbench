# Databricks notebook source
# MAGIC %md
# MAGIC #### Setup: `%run ./utils` 

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils

# COMMAND ----------

CATALOG, DB_SCHEMA, MODEL_FAMILY

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Define Custom PyFunc for: `SCimilarity_GeneOrder`

# COMMAND ----------

# DBTITLE 1,SCimilarity_GeneOrder
import csv
from typing import Any, List, Optional
import mlflow
import numpy as np
import scipy
import pandas as pd
from mlflow.pyfunc.model import PythonModelContext
from scimilarity import CellEmbedding, CellQuery

class SCimilarity_GeneOrder(mlflow.pyfunc.PythonModel):
    r"""Create MLFlow Pyfunc class for SCimilarity gene order."""

    def load_context(self, context: PythonModelContext):
        r"""Intialize pre-trained SCimilarity model.

        Parameters
        ----------
        context : PythonModelContext
            Context object for MLFlow model -- here we are using it to load the gene order from the model weights folder

        """
        gene_path = context.artifacts["geneOrder_path"]
        self.gene_order = []

        with open(gene_path) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for row in tsvreader:
                for gene in row:
                    self.gene_order.append(gene)
    
    def predict(
        self,        
        model_input: pd.DataFrame = Any | None         
    ):
        r"""Output prediction on model.

        Parameters
        ----------
        model_input : pd.DataFrame
        Input data (ignored - gene order is static)

        Returns
        -------
        PyFuncOutput
            The gene order.

        """
        return self.gene_order

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Local Context for Defined Model

# COMMAND ----------

# DBTITLE 1,TEST Local Context
# Create a temporary context to initialize the model
class TempContext:
    artifacts = {        
        "geneOrder_path": geneOrder_path  # need to include this now
    }

temp_context = TempContext()

# Initialize the model and test with temporary context
model = SCimilarity_GeneOrder()
model.load_context(temp_context)

# COMMAND ----------

# DBTITLE 1,Specify model_input
model_input: pd.DataFrame = Any | None
model_input

# COMMAND ----------

# DBTITLE 1,Test model_input
gene_order_output = model.predict(model_input)

len(gene_order_output)
# print(gene_order_output)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define MLflow Signature with local Model + Context

# COMMAND ----------

# DBTITLE 1,Define MLflow Signature
from mlflow.models import infer_signature

# Example input for the model
example_input = pd.DataFrame({
                                'input': ["get_gene_order"]  # Just to trigger getting GeneOrder list from model weights folder
                            })

# Ensure the example output is in a serializable format
example_output = model.predict(example_input)

# Infer the model signature
signature = infer_signature(example_input, example_output)

# COMMAND ----------

# example_input #, example_output

# COMMAND ----------

signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_GeneOrder`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "GeneOrder"

## Set the experiment
experiment_dir = f"{user_path}/mlflow_experiments/{MODEL_FAMILY}"
print(experiment_dir)

# experiment_name = f"{user_path}/mlflow_experiments/{MODEL_FAMILY}/{MODEL_TYPE}"
experiment_name = f"{experiment_dir}/{MODEL_TYPE}"
print(experiment_name)

# COMMAND ----------

# DBTITLE 1,create experiment_dir
from databricks.sdk import WorkspaceClient

# Initialize client (uses ~/.databrickscfg or env vars for auth)
client = WorkspaceClient()

# Create workspace folder
client.workspace.mkdirs(    
                        # path = f"{user_path}/mlflow_experiments/{MODEL_FAMILY}"
                        path = f"{experiment_dir}", 
                      )

# List to verify
folders = client.workspace.list(f"{user_path}/mlflow_experiments")
for folder in folders:
  if folder.path == experiment_dir:
    print(f"Name: {folder.path}, Type: {folder.object_type}")

# COMMAND ----------

# DBTITLE 1,Log SCimilarity_GeneOrder
import os 
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

# Log the model
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Check if the experiment_name (defined above) exists
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    exp_id = mlflow.create_experiment(experiment_name)
else:
    exp_id = experiment.experiment_id

mlflow.set_experiment(experiment_id=exp_id)

# Save and log the model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path=f"{MODEL_TYPE}",
        python_model=model, 
        artifacts={
                   "geneOrder_path": geneOrder_path ## defined in ./utils 
                  },
        input_example=example_input,
        signature=signature
    )

    run_id = run.info.run_id
    print("Model logged with run ID:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check `run_id` logged Model & Predictions

# COMMAND ----------

# run_id #= "<include to save for debugging>"

# COMMAND ----------

# DBTITLE 1,load MLflow Logged model + test
import mlflow
logged_model_run_uri = f'runs:/{run_id}/{MODEL_TYPE}'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model_run_uri) ## 

# COMMAND ----------

# DBTITLE 1,access input_example from loaded_model
loaded_model.input_example

# COMMAND ----------

# DBTITLE 1,Test logged + loaded model prediction
predictions = loaded_model.predict(loaded_model.input_example)

len(predictions)
# print(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### UC Register Custom PyFunc: `SCimilarity_GeneOrder`

# COMMAND ----------

# DBTITLE 1,Model Info
# Register the model
model_name = f"SCimilarity_{MODEL_TYPE}"  
full_model_name = f"{CATALOG}.{DB_SCHEMA}.{model_name}"
model_uri = f"runs:/{run_id}/{MODEL_TYPE}"

model_name, full_model_name, model_uri

# COMMAND ----------

# DBTITLE 1,Register SCimilarity_GeneOrder
# registered_model = 
mlflow.register_model(model_uri=model_uri, 
                      name=full_model_name,                      
                      await_registration_for=120,
                    )

# COMMAND ----------

# DBTITLE 1,Associate model version with @: add_model_alias
add_model_alias("SCimilarity_GeneOrder", "Champion")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load UC registered model using @lias & test predictions

# COMMAND ----------

# DBTITLE 1,Load uc-registered model using alias
## Load the model as a PyFunc model using alias
# model_uri = f"models:/{full_model_name}@Champion"
# loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# DBTITLE 1,test predictions
## Make predictions
# predictions = loaded_model.predict(loaded_model.input_example) 

# len(predictions)
# # print(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Deploy & Serve UC registered model: `SCimilarity_GeneOrder`

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deployment Parameters for Endpoint Config

# COMMAND ----------

# DBTITLE 1,Endpoint Deployment Parameters
## Databricks HOST & TOKEN info.
# Get the API endpoint and token for the current notebook context
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
# API_TOKEN = dbutils.secrets.get(scope="mmt", key="hls_fe_SP") ## for some reason this won't work for API inference 
DATABRICKS_TOKEN = dbutils.secrets.get(scope="mmt", key="databricks_token")

import os
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

## databricks_instance -- this is NOT the same as dbutils derived DATABRICKS_HOST / workspace_url
databricks_instance = "e2-demo-field-eng.cloud.databricks.com"  # Replace with your Databricks instance URL 

## UC namespace
catalog_name = CATALOG #"mmt"
schema_name = DB_SCHEMA #"genesiswb"
# model_name = "SCimilarity_{MODLE_TYPE}" # "SCimilarity_GeneOrder" 
full_model_name = f"{CATALOG}.{DB_SCHEMA}.{model_name}"

## model info. 
registered_model_name = full_model_name #f"{catalog_name}.{schema_name}.{model_name}"

latest_model_version = get_latest_model_version(registered_model_name)
general_model_name = f"{model_name}-{latest_model_version}" 

## endpoint names
endpoint_base_name = f"gwb_{model_name}"
endpoint_name = f"{endpoint_base_name}_mmt" #-mlflowsdk" # endpoint name

## workload types&sizes
workload_type = "CPU"
workload_size = "Small"

## show configs
config_list = [
    f"catalog: {catalog_name}",
    f"schema: {schema_name}",
    f"host: {DATABRICKS_HOST}",
    f"token: {DATABRICKS_TOKEN}",
    f"workload_type: {workload_type}",
    f"workload_size: {workload_size}",
    f"general_model_name: {general_model_name}",
    f"registered_model_name: {registered_model_name}",
    f"latest_model_version: {latest_model_version}",
    f"endpoint_name: {endpoint_name}"
]

[print(config) for config in config_list];

# COMMAND ----------

# MAGIC %md
# MAGIC #### Endpoint Config Specs

# COMMAND ----------

# DBTITLE 1,client endpoint_config
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

# Define the full API request payload
endpoint_config = {
    "name": general_model_name,
    "served_models": [
        {                
            "model_name": registered_model_name,
            "model_version": latest_model_version,
            "workload_size": workload_size,  # defines concurrency: Small/Medium/Large
            "workload_type": workload_type,  # defines compute: GPU_SMALL/GPU_MEDIUM/MULTIGPU_MEDIUM/GPU_LARGE
            "scale_to_zero_enabled": True
        }
    ],
    "routes": [
        {
            "served_model_name": general_model_name,
            "traffic_percentage": 100
        }
    ],
    "auto_capture_config": {
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        # "table_name_prefix": f"{endpoint_base_name}_unique_prefix",  # Specify a unique table prefix
        "enabled": True
    },
    "tags": [
        {"key": "project", "value": f"{schema_name}"},
        {"key": "removeAfter", "value": "2025-12-31"}
    ]
}

# Ensure endpoint_name is valid
endpoint_name = endpoint_name[:63]  # Truncate to 63 characters if necessary
endpoint_name = ''.join(e for e in endpoint_name if e.isalnum() or e in ['-', '_'])  # Remove invalid characters

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create or update the endpoint

# COMMAND ----------

# DBTITLE 1,Create or update the endpoint
# Create or update the endpoint
try:
    # Check if endpoint exists
    existing_endpoint = client.get_endpoint(endpoint_name)
    print(f"Endpoint {endpoint_name} exists, updating configuration...")
    client.update_endpoint_config(endpoint_name, config=endpoint_config)
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"Creating new endpoint {endpoint_name}...")
        client.create_endpoint(endpoint_name, config=endpoint_config)
    else:
        raise

# COMMAND ----------

# DBTITLE 1,get endpoint link
databricks_instance = "e2-demo-field-eng.cloud.databricks.com"  # Replace with your Databricks instance URL -- this is NOT the same as workspace_url derived with dbutils 

endpoint_url = get_endpoint_link(databricks_instance, model_name, latest_model_version);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check Endpoint Status 

# COMMAND ----------

# DBTITLE 1,check for endpoint updates
check_endpoint_status(endpoint_name, max_checks=20, sleep_time=30)

# COMMAND ----------

# DBTITLE 1,check endpoint status
# MAGIC %md
# MAGIC #### For endpoint status test/debug

# COMMAND ----------

# DBTITLE 1,check endpoint status

# from mlflow.deployments import get_deploy_client

# client = get_deploy_client("databricks")
# endpoint_status = client.get_endpoint(endpoint_name)
# endpoint_status

# COMMAND ----------

# endpoint_status['state']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Endpoint 

# COMMAND ----------

# model_input, example_input

# COMMAND ----------

# score_model(databricks_instance, endpoint_name, example_input)

# COMMAND ----------

# DBTITLE 1,input for inferencing
# {
#   "dataframe_split": {
#     "columns": [
#       "input"
#     ],
#     "data": [
#       [
#         "go"
#       ]
#     ]
#   }
# }