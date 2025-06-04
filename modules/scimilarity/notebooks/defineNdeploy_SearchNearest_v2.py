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
# MAGIC ### Define Custom PyFunc for: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,SCimilarity_SearchNearest
import csv
from typing import Any
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModelContext
from scimilarity import CellEmbedding, CellQuery
import torch

class SCimilarity_SearchNearest(mlflow.pyfunc.PythonModel):
    r"""Create MLFlow Pyfunc class for SCimilarity model."""

    def load_context(self, context: PythonModelContext):
        r"""Intialize pre-trained SCimilarity model.

        Parameters
        ----------
        context : PythonModelContext
            Context object for MLFlow model -- here we are loading the pretrained model weights.

        """
        self.cq = CellQuery(context.artifacts["model_path"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame, 
        params: dict[str, Any], 
    ) -> pd.DataFrame:
        r"""Output prediction on model.

        Parameters
        ----------
        context : PythonModelContext
            Context object for MLFlow model.
        model_input : pd.DataFrame
            DataFrame containing embeddings.

        Returns
        -------
        pd.DataFrame
            The predicted classes.

        """
        embeddings = model_input.embedding[0] 
        
        predictions = self.cq.search_nearest(embeddings, k=params["k"]) # external params dict

        results_dict = {
            "nn_idxs": [np_array.tolist() for np_array in predictions[0]],
            "nn_dists": [np_array.tolist() for np_array in predictions[1]],
            "results_metadata": predictions[2].to_dict()
        }
        results_df = pd.DataFrame([results_dict])
        return results_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Local Context for Defined Model

# COMMAND ----------

# DBTITLE 1,TEST Local Context
# Create a temporary context to initialize the model
class TempContext:
    artifacts = {
                 "model_path": model_path,        
                }

temp_context = TempContext()

# Initialize the model and test with temporary context
model = SCimilarity_SearchNearest()
model.load_context(temp_context)

# COMMAND ----------

# DBTITLE 1,Specify model_input
## Create a DataFrame containing the embeddings

# cell_embeddings.dtype #dtype('float32')
# cell_embeddings.shape #(1, 128)

model_input = pd.DataFrame([{"embedding": cell_embeddings.tolist()[0]}])
display(model_input)

# COMMAND ----------

# DBTITLE 1,arrow warning (?)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# COMMAND ----------

# DBTITLE 1,Test model_input
# Call the predict method
searchNearest_output = model.predict(temp_context, model_input, params={"k": 100})

display(searchNearest_output)

# COMMAND ----------

# searchNearest_output

# COMMAND ----------

# DBTITLE 1,test extracting results_metadata from searchNearest_output
## Extract results_metadata from the output
# results_metadata = searchNearest_output["results_metadata"].iloc[0]
# pd.DataFrame(results_metadata)

# COMMAND ----------

# DBTITLE 1,params
# import json

params: dict[str, Any] = dict({"k": 100})
params.values()

# pd.DataFrame([json.dumps(params)], columns=["params"])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define MLflow Signature with local Model + Context

# COMMAND ----------

# DBTITLE 1,Define MLflow Signature
from mlflow.models import infer_signature
import pandas as pd

# Define a concrete example input as a Pandas DataFrame
example_input = model_input ## we will add params separately to keep it simple... but make a note on the usage input patterns 

# Ensure the example output is in a serializable format
example_output = searchNearest_output

# Create a Dict for params
params: dict[str, Any] = dict({"k": 100})


# Infer the model signature
signature = infer_signature(
    model_input=model_input, #example_input,
    model_output=example_output,
    params=params
)

# COMMAND ----------

example_input, example_output

# COMMAND ----------

# DBTITLE 1,check signature
signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "SearchNearest"

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

# DBTITLE 1,specify mlflow requirements.txt
import os

# Create a requirements.txt file with the necessary dependencies
requirements = """
cloudpickle==2.0.0
scanpy==1.11.2
numcodecs==0.13.1
scimilarity==0.4.0
pandas==1.5.3
numpy==1.26.4 
"""

# model_name = "{MODEL_FAMILY}_{MODEL_NAME}" 
model_name = "SCimilarity_SearchNearest"  # to update class func

# Define the path to save the requirements file in the UV volumes
SCimilarity_SearchNearest_requirements_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/mlflow_requirements/{model_name}/requirements.txt"
# SCimilarity_SearchNearest_requirements_path = f"/Volumes/mmt/genesiswb/scimilarity/mlflow_requirements/{model_name}/requirements.txt"

# Create the directory if it does not exist
os.makedirs(os.path.dirname(SCimilarity_SearchNearest_requirements_path), exist_ok=True)

# Write the requirements to the file
with open(SCimilarity_SearchNearest_requirements_path, "w") as f:
    f.write(requirements)

print(f"Requirements written to {SCimilarity_SearchNearest_requirements_path}")

# COMMAND ----------

# DBTITLE 1,log SCimilarity_SearchNearest
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
        artifact_path="searchNearest_model",
        python_model=model, 
        artifacts={
                    "model_path": model_path,            
                  },
        # input_example = model_input, # without params -- to add separately during inference? 
        input_example = example_input, 
        signature = signature, ## params defined in signature https://mlflow.org/docs/latest/model/signatures/#inference-params
        pip_requirements=SCimilarity_SearchNearest_requirements_path,        
    )

    run_id = run.info.run_id
    print("Model logged with run ID:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check `run_id` logged Model & Predictions

# COMMAND ----------

# DBTITLE 1,load MLflow Logged model + test
import mlflow
logged_model_run_uri = f'runs:/{run_id}/searchNearest_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model_run_uri) ## 

# COMMAND ----------

# DBTITLE 1,access input_example from loaded_model
loaded_model.input_example

# COMMAND ----------

# DBTITLE 1,params
params['k'], params

# COMMAND ----------

# DBTITLE 1,Test logged + loaded model prediction with params
# loaded_model.predict(loaded_model.input_example, params={"k": 100})
predictions = loaded_model.predict(loaded_model.input_example, params)
predictions

# predictions = loaded_model.predict(loaded_model.input_example)
# print(predictions)

# loaded_model.predict(loaded_model.input_example) ## with k-hardcoded

# COMMAND ----------

# MAGIC %md 
# MAGIC ### UC Register Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,test
# run_id ="a0c7afe8d7894511a25e36f06f0a3101"

# COMMAND ----------

# DBTITLE 1,Model Info
# Register the model
model_name = "SCimilarity_SearchNearest"  # to update class func
full_model_name = f"mmt.genesiswb.{model_name}"
model_uri = f"runs:/{run_id}/searchNearest_model"

model_name, full_model_name, model_uri

# COMMAND ----------

# DBTITLE 1,register SCimilarity_SearchNearest
# registered_model = 
mlflow.register_model(model_uri=model_uri, 
                      name=full_model_name,                      
                      await_registration_for=120,
                    )

# COMMAND ----------

# DBTITLE 1,Associate model version with alias
add_model_alias("SCimilarity_SearchNearest", "Champion")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load UC registered model using @lias & test predict

# COMMAND ----------

# DBTITLE 1,test registered model version load
## Load the model as a PyFunc model using alias
# model_uri = f"models:/{full_model_name}@Champion"
# loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# DBTITLE 1,test make predictions
## Make predictions
# predictions = loaded_model.predict(loaded_model.input_example, params={"k":10}) 
# print(predictions)

# COMMAND ----------

# DBTITLE 1,extract meta_data
# # Load the model
# model = mlflow.pyfunc.load_model(model_uri)

# # Get model metadata
# model_metadata = model.metadata
# print("Model Metadata:", model_metadata)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Deploy & Serve UC registered model: `SCimilarity_SearchNearest`

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deployment Parameters for Endpoint Config

# COMMAND ----------

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
# model_name = "SCimilarity_{MODLE_TYPE}" # "SCimilarity_GetEmbeddings" 
full_model_name = f"{CATALOG}.{DB_SCHEMA}.{model_name}"

## model info. 
registered_model_name = full_model_name #f"{catalog_name}.{schema_name}.{model_name}"

latest_model_version = get_latest_model_version(registered_model_name)
general_model_name = f"{model_name}-{latest_model_version}" 

## endpoint names
endpoint_base_name = f"gwb_{model_name}"
endpoint_name = f"{endpoint_base_name}_mmt" #-mlflowsdk" # endpoint name

## workload types&sizes
# https://docs.databricks.com/api/workspace/servingendpoints/create#config-served_models-workload_type
# workload_type = "GPU_MEDIUM" ## deployment timeout!
workload_type = "MULTIGPU_MEDIUM"  # 4xA10G
workload_size = "Medium"

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
            "workload_type": workload_type,  #"MULTIGPU_MEDIUM",  ## defines compute: GPU_SMALL/GPU_MEDIUM/MULTIGPU_MEDIUM/GPU_LARGE
            "scale_to_zero_enabled": True,
            "tracing_enabled": True
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

# MAGIC %md
# MAGIC #### For endpoint status test/debug

# COMMAND ----------

# DBTITLE 1,check status
# from mlflow.deployments import get_deploy_client

# client = get_deploy_client("databricks")
# endpoint_status = client.get_endpoint(endpoint_name)

# COMMAND ----------

# endpoint_status

# COMMAND ----------

# endpoint_status['state']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Endpoint 

# COMMAND ----------

# model_input, example_output

# COMMAND ----------

# DBTITLE 1,try score_model with params
params={"k": 10}

# result = score_model(your_dataframe, params={"k": 10})
result = score_model(databricks_instance, endpoint_name, model_input, params)
result 

# COMMAND ----------

# DBTITLE 1,test extract predictions results_metadata
# pd.DataFrame(result["predictions"][0]["results_metadata"]) #.keys()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,format model_input + params for UI inferencing
# dataset = model_input

# ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
# if params:
#     ds_dict['params'] = params

# COMMAND ----------

# DBTITLE 1,for UI inference testing
# ds_dict