# Databricks notebook source
# MAGIC %md
# MAGIC #### Setup: `%run ./utils` 

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils_20250801 

# COMMAND ----------

# DBTITLE 1,pinning mlflow == 2.22.0
# mlflow.__version__

# COMMAND ----------

CATALOG, DB_SCHEMA, MODEL_FAMILY, MODEL_NAME, EXPERIMENT_NAME

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Define Custom PyFunc for: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,SCimilarity_SearchNearest
# import csv
# from typing import Any, Optional
# import mlflow
# import numpy as np
# import pandas as pd
# from mlflow.pyfunc.model import PythonModelContext
# from scimilarity import CellEmbedding, CellQuery
# import torch

# class SCimilarity_SearchNearest(mlflow.pyfunc.PythonModel):
#     r"""Create MLFlow Pyfunc class for SCimilarity model."""

#     def load_context(self, context: PythonModelContext):
#         r"""Intialize pre-trained SCimilarity model.

#         Parameters
#         ----------
#         context : PythonModelContext
#             Context object for MLFlow model -- here we are loading the pretrained model weights.

#         """
#         self.cq = CellQuery(context.artifacts["model_path"])

#     def predict(
#         self,
#         context: PythonModelContext,
#         model_input: pd.DataFrame, 
#         # params: Optional[dict[str, Any]],  ## move to model_input (as optional)
#     ) -> pd.DataFrame:
#         r"""Output prediction on model.

#         Parameters
#         ----------
#         context : PythonModelContext
#             Context object for MLFlow model.
#         model_input : pd.DataFrame
#             DataFrame containing embeddings.

#         Returns
#         -------
#         pd.DataFrame
#             The predicted classes.

#         """
#         embeddings = model_input.embedding[0] 
        
#         predictions = self.cq.search_nearest(embeddings, k=params["k"]) # external params dict

#         results_dict = {
#             "nn_idxs": [np_array.tolist() for np_array in predictions[0]],
#             "nn_dists": [np_array.tolist() for np_array in predictions[1]],
#             "results_metadata": predictions[2].to_dict()
#         }
#         results_df = pd.DataFrame([results_dict])
#         return results_df

# COMMAND ----------

# DBTITLE 1,update  wrt signature + limitations of sdk WorkspaceClient
import mlflow
import numpy as np
import pandas as pd
import json
from mlflow.pyfunc.model import PythonModelContext
from scimilarity import CellQuery

class SCimilarity_SearchNearest(mlflow.pyfunc.PythonModel):
    """Create MLFlow Pyfunc class for SCimilarity model."""

    def load_context(self, context: PythonModelContext):
        """Initialize pre-trained SCimilarity model."""
        self.cq = CellQuery(context.artifacts["model_path"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame
    ) -> pd.DataFrame:
        """Output prediction on model."""
        # Extract embedding
        embeddings = model_input.embedding[0]

        # Handle optional params column as JSON string
        params = {"k": 100}
        if "params" in model_input.columns:
            raw_params = model_input["params"].iloc[0]
            if raw_params is not None and not (isinstance(raw_params, float) and pd.isna(raw_params)):
                try:
                    params = json.loads(raw_params)
                except Exception:
                    params = {"k": 100}

        predictions = self.cq.search_nearest(embeddings, k=params.get("k", 100))

        results_dict = {
            "nn_idxs": [np_array.tolist() for np_array in predictions[0]],
            "nn_dists": [np_array.tolist() for np_array in predictions[1]],
            "results_metadata": predictions[2].to_dict()
        }
        results_df = pd.DataFrame([results_dict])
        return results_df

# COMMAND ----------

model_input.params[0]

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

params: Optional[dict[str, Any]] = dict({"k": 100})
params.values()

# COMMAND ----------

# DBTITLE 1,Specify example model_input
## Create a DataFrame containing the embeddings

# cell_embeddings.dtype #dtype('float32')
# cell_embeddings.shape #(1, 128)

# model_input = pd.DataFrame([{"embedding": cell_embeddings.tolist()[0]}])
model_input = pd.DataFrame([
    {
        "embedding": cell_embeddings.tolist()[0],  # list of floats
        "params": json.dumps(params) #{"k": 100}  # JSON string
    }
])

# Ensure embedding is a list of floats
model_input["embedding"] = model_input["embedding"].apply(
    lambda x: list(np.array(x, dtype=float)) if not isinstance(x, list) else x
)

display(model_input)

# COMMAND ----------

type(model_input["embedding"][0][0])

# COMMAND ----------

# DBTITLE 1,arrow warning (?)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# COMMAND ----------

# DBTITLE 1,Test model_input + params
# Call the predict method
# searchNearest_output = model.predict(temp_context, model_input, params={"k": 100})

searchNearest_output = model.predict(temp_context, model_input) #, params={"k": 100})

display(searchNearest_output)

# COMMAND ----------

pd.DataFrame(searchNearest_output["results_metadata"][0])

# COMMAND ----------

# model.predict(temp_context, model_input, params={"k": 10})
model.predict(temp_context, model_input)

# COMMAND ----------

model_input0 = pd.DataFrame([
    {
        "embedding": cell_embeddings.tolist()[0],
        # "params": {"k": 100}
    }
])
display(model_input0)

# COMMAND ----------

model.predict(temp_context, model_input0)

# COMMAND ----------

# DBTITLE 1,test extracting results_metadata from searchNearest_output
## Extract results_metadata from the output
# results_metadata = searchNearest_output["results_metadata"].iloc[0]
# pd.DataFrame(results_metadata)

# COMMAND ----------

# DBTITLE 1,specify params
## import json

# params: dict[str, Any] = dict({"k": 100})
# params.values()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Define MLflow Signature with local Model + Context

# COMMAND ----------

# DBTITLE 1,Define MLflow Signature
from mlflow.models import infer_signature
import pandas as pd

# Define a concrete example input as a Pandas DataFrame
example_input = model_input.copy() ## we will add params separately to keep it simple... but make a note on the usage input patterns 

# Ensure the example output is in a serializable format
example_output = searchNearest_output

# Create a Dict for params
# params: dict[str, Any] = dict({"k": 100}) ## could take any dict and if none provided defaults to example provided
# params: Optional[dict[str, Any]] = dict({"k": 100})

# # Infer the model signature
signature = infer_signature(
    model_input = model_input, #example_input,
    model_output = example_output,
    # params=params
)

# COMMAND ----------

# example_input, example_output

# COMMAND ----------

# DBTITLE 1,check signature
signature

## original signature with proper params key-value instead of being part of inputs
# inputs: 
#   ['embedding': Array(double) (required)]
# outputs: 
#   ['nn_idxs': Array(Array(long)) (required), 'nn_dists': Array(Array(double)) (required), 'results_metadata': string (required)]
# params: 
#   ['k': long (default: 100)]

# COMMAND ----------

# print("MLflow version:", mlflow.__version__) # no struct in mlflow 2.22.0

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import (
    Schema,      # collection of column specs
    ColSpec,     # a single column description
    DataType,    # enum of primitive types
    Array,       # wrapper for array types
    # Struct is *not* needed for the solutions below
)

embedding_col = ColSpec(
    name="embedding",
    type=Array(DataType.double),   # <-- correct way to say “array<double>”
    required=True,
)

params_col = ColSpec(
    name="params",
    type=DataType.string,          # the whole dict will be JSON‑encoded by the caller
    required=False,
)

nn_idxs_col = ColSpec(
    name="nn_idxs",
    type=Array(Array(DataType.long)),   # array<array<long>>
    required=True,
)

nn_dists_col = ColSpec(
    name="nn_dists",
    type=Array(Array(DataType.double)), # array<array<double>>
    required=True,
)

metadata_col = ColSpec(
    name="results_metadata",
    type=DataType.string,
    required=True,
)

signature = ModelSignature(
    inputs=Schema([embedding_col, params_col]),
    outputs=Schema([nn_idxs_col, nn_dists_col, metadata_col]),
)

# COMMAND ----------

signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "Search_Nearest" ## 
# model_name = f"SCimilarity_{MODEL_TYPE}"  
model_name = f"{MODEL_NAME}_{MODEL_TYPE}"  

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

# Create a requirements.txt file with the necessary dependencies & pinned versions
requirements = """
mlflow==2.22.0 
cloudpickle==2.0.0
scanpy==1.11.2
numcodecs==0.13.1
scimilarity==0.4.0
pandas==1.5.3
numpy==1.26.4 
"""

# model_name = "SCimilarity_Search_Nearest"  # to update class func
model_name = f"{MODEL_NAME}_{MODEL_TYPE}" 

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
        artifact_path=f"{MODEL_TYPE}", # artifact_path --> "name" mlflow v3.0
        python_model=model, 
        artifacts={
                    "model_path": model_path,   ## defined in ./utils          
                  },    
        input_example = example_input, # without params -- has a default value in model signature OR to add separately during inference | (model_input, params) tuple formatting not quite right
       
        signature = signature, ## optional params defined in input signature https://mlflow.org/docs/latest/model/signatures/#inference-params
        
        pip_requirements=SCimilarity_SearchNearest_requirements_path,     
        
        # registered_model_name=f"{CATALOG}.{SCHEMA}.{model_name}" ## to include directly wihout additonal load run_id checks   
    )

    run_id = run.info.run_id
    print("Model logged with run ID:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check `run_id` logged Model & Predictions

# COMMAND ----------

# run_id #= "<include to save for debugging>"
# run_id = "ccf483b373d643818dad1966a01febfe"

# COMMAND ----------

# DBTITLE 1,load MLflow Logged model + test
import mlflow
logged_model_run_uri = f'runs:/{run_id}/{MODEL_TYPE}'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model_run_uri) ## 

# COMMAND ----------

# DBTITLE 1,access input_example from loaded_model
loaded_model.input_example #, example_output

# COMMAND ----------

# DBTITLE 1,check params
# params = {"k": 10}  # Provide a default value for params
# # loaded_model.predict(loaded_model.input_example)
# loaded_model.predict(loaded_model.input_example, params=params)

loaded_model.predict(loaded_model.input_example)

# COMMAND ----------

loaded_model.predict(model_input0)
# loaded_model.predict(loaded_model.input_example, params)

# COMMAND ----------

# DBTITLE 1,Test logged + loaded model prediction with params
# loaded_model.predict(loaded_model.input_example, params={"k": 100})
# predictions = loaded_model.predict(loaded_model.input_example)

# predictions = loaded_model.predict(loaded_model.input_example, params)
# predictions

# predictions = loaded_model.predict(loaded_model.input_example) # mlflow registered default params will kick-in 
# print(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### UC Register Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,Model Info
# Register the model
model_name = f"{MODEL_NAME}_{MODEL_TYPE}" 
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
# MAGIC ## AWS workload types&sizes
# MAGIC # https://docs.databricks.com/api/workspace/servingendpoints/create#config-served_models-workload_type
# MAGIC # workload_type = "GPU_MEDIUM" ## deployment timeout!
# MAGIC workload_type = "MULTIGPU_MEDIUM"  # 4xA10G
# MAGIC workload_size = "Medium"
# MAGIC ```    
# MAGIC ---        
# MAGIC         
# MAGIC ```
# MAGIC ## AzureDB workload types&sizes
# MAGIC # https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints
# MAGIC workload_type = "GPU_Large" (A100), 
# MAGIC workload_size = "Small" 0-4 concurrency 
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,format model_input + params for UI inferencing
dataset = model_input

ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} # includes "index":[0]

ds_dict['dataframe_split'].pop('index', None)  # Remove the index key if it exists

if params:
    ds_dict['params'] = params

# COMMAND ----------

# DBTITLE 1,example input for UI inferencing
serving_input = json.dumps(ds_dict).replace("'",'"')
serving_input

# COMMAND ----------

# DBTITLE 1,search_model_versions
# mlflow.search_model_versions(filter_string="name = 'genesis_workbench.dev_mmt_core_test.SCimilarity_Search_Nearest'")

# COMMAND ----------

# DBTITLE 1,get model_uri
# # import mlflow

# ## Assumes model registered to Unity Catalog

# # Sift for model latest version 
# model_versions = mlflow.search_model_versions(filter_string="name = 'genesis_workbench.dev_mmt_core_test.SCimilarity_Search_Nearest'")
# model_version = model_versions[0].version
# print(model_version)

# model_uri = f"models:/genesis_workbench.dev_mmt_core_test.SCimilarity_Search_Nearest/{model_version}"
# print(model_uri)
