# Databricks notebook source
# MAGIC %md
# MAGIC #### Setup: `%run ./utils` 

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils 

# COMMAND ----------

CATALOG, DB_SCHEMA, MODEL_FAMILY, EXPERIMENT_NAME

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

# DBTITLE 1,Specify example model_input
## Create a DataFrame containing the embeddings

# cell_embeddings.dtype #dtype('float32')
# cell_embeddings.shape #(1, 128)

model_input = pd.DataFrame([{"embedding": cell_embeddings.tolist()[0]}])
display(model_input)

# COMMAND ----------

# DBTITLE 1,arrow warning (?)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# COMMAND ----------

# DBTITLE 1,Test model_input + params
# Call the predict method
searchNearest_output = model.predict(temp_context, model_input, params={"k": 100})
# searchNearest_output
display(searchNearest_output)

# COMMAND ----------

# DBTITLE 1,test extracting results_metadata from searchNearest_output
## Extract results_metadata from the output
# results_metadata = searchNearest_output["results_metadata"].iloc[0]
# pd.DataFrame(results_metadata)

# COMMAND ----------

# DBTITLE 1,specify params
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
params: dict[str, Any] = dict({"k": 100}) ## could take any dict and if none provided defaults to example provided


# Infer the model signature
signature = infer_signature(
    model_input=model_input, #example_input,
    model_output=example_output,
    params=params
)

# COMMAND ----------

# example_input, example_output

# COMMAND ----------

# DBTITLE 1,check signature
signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "Search_Nearest" ## 
model_name = f"SCimilarity_{MODEL_TYPE}"  

## Set the experiment
# experiment_dir = f"{user_path}/mlflow_experiments/{MODEL_FAMILY}" ## TO UPDATE
experiment_dir = f"{user_path}/mlflow_experiments/{EXPERIMENT_NAME}" ## same as MODEL_FAMILY from widget above
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
# with mlflow.start_run(run_name=f'{model_name}', experiment_id=experiment.experiment_id)
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path=f"{MODEL_TYPE}",
        python_model=model, 
        artifacts={
                    "model_path": model_path,   ## defined in ./utils          
                  },    
        input_example = example_input, # without params -- has a default value in model signature OR to add separately during inference 
        signature = signature, ## params defined in signature https://mlflow.org/docs/latest/model/signatures/#inference-params
        pip_requirements=SCimilarity_SearchNearest_requirements_path,     
        # registered_model_name=f"{CATALOG}.{SCHEMA}.{model_name}" ## to include directly wihout additonal load run_id checks   
    )

    run_id = run.info.run_id
    print("Model logged with run ID:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check `run_id` logged Model & Predictions

# COMMAND ----------

# DBTITLE 1,load MLflow Logged model + test
import mlflow
logged_model_run_uri = f'runs:/{run_id}/{MODEL_TYPE}'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model_run_uri) ## 

# COMMAND ----------

# DBTITLE 1,access input_example from loaded_model
# loaded_model.input_example, example_output

# COMMAND ----------

# DBTITLE 1,check params
params['k'], params

# COMMAND ----------

# DBTITLE 1,Test logged + loaded model prediction with params
# loaded_model.predict(loaded_model.input_example, params={"k": 100})
predictions = loaded_model.predict(loaded_model.input_example, params)
predictions

# predictions = loaded_model.predict(loaded_model.input_example) # mlflow registered default params will kick-in 
# print(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### UC Register Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,Model Info
# Register the model
model_name = f"SCimilarity_{MODEL_TYPE}" 
full_model_name = f"{CATALOG}.{DB_SCHEMA}.{model_name}"
model_uri = f"runs:/{run_id}/{MODEL_TYPE}"

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
# add_model_alias(full_model_name, "Champion")

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
# MAGIC
# MAGIC ```
# MAGIC ## workload types&sizes
# MAGIC # https://docs.databricks.com/api/workspace/servingendpoints/create#config-served_models-workload_type
# MAGIC # workload_type = "GPU_MEDIUM" ## deployment timeout!
# MAGIC workload_type = "MULTIGPU_MEDIUM"  # 4xA10G
# MAGIC workload_size = "Medium"
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,format model_input + params for UI inferencing
# dataset = model_input

# ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
# if params:
#     ds_dict['params'] = params

# COMMAND ----------

# DBTITLE 1,example input for UI inferencing
# ds_dict
