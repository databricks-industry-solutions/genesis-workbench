# Databricks notebook source
# MAGIC %md
# MAGIC #### Setup: `%run ./utils` 

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils 

# COMMAND ----------

# DBTITLE 1,pinning mlflow == 2.22.0
mlflow.__version__

# COMMAND ----------

CATALOG, DB_SCHEMA, MODEL_FAMILY, MODEL_NAME, EXPERIMENT_NAME

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

display(searchNearest_output)

# COMMAND ----------

model.predict(temp_context, model_input, params={"k": 100})

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
    model_input = model_input, #example_input,
    model_output = example_output,
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

# run_id #= "<include to save for debugging>"

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
params = {"k": 10}  # Provide a default value for params
# loaded_model.predict(loaded_model.input_example)
loaded_model.predict(loaded_model.input_example, params=params)

# COMMAND ----------

loaded_model.predict(loaded_model.input_example)
# loaded_model.predict(loaded_model.input_example, params)

# COMMAND ----------

# DBTITLE 1,Test logged + loaded model prediction with params
# loaded_model.predict(loaded_model.input_example, params={"k": 100})
# predictions = loaded_model.predict(loaded_model.input_example)

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
# dataset = model_input

# ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} # includes "index":[0]
# if params:
#     ds_dict['params'] = params

# COMMAND ----------

# DBTITLE 1,example input for UI inferencing
# json.dumps(ds_dict).replace("'",'"')

# COMMAND ----------

# DBTITLE 1,search_model_versions
# mlflow.search_model_versions(filter_string="name = 'genesis_workbench.dev_mmt_core_test.SCimilarity_Search_Nearest'")

# COMMAND ----------

# DBTITLE 1,convert serving input
serving_input_example = json.loads(mlflow.models.convert_input_example_to_serving_input((model_input, params={"k": 5}) ) )

serving_input_example

# COMMAND ----------

# DBTITLE 1,validate_serving_input by run_id
## Validate the model input with parameters
# model_uri_byrunid = f'runs:/{run_id}/Search_Nearest' #f"runs:/{run_id}/{MODEL_TYPE}"
# mlflow.models.validate_serving_input(model_uri_byrunid, json.dumps(serving_input_example).replace("'",'"'))

# COMMAND ----------

# DBTITLE 1,get model_uri
# import mlflow

## Assumes model registered to Unity Catalog

# Sift for model latest version 
model_versions = mlflow.search_model_versions(filter_string="name = 'genesis_workbench.dev_mmt_core_test.SCimilarity_Search_Nearest'")
model_version = model_versions[0].version
print(model_version)

model_uri = f"models:/genesis_workbench.dev_mmt_core_test.SCimilarity_Search_Nearest/{model_version}"
print(model_uri)

# COMMAND ----------

# DBTITLE 1,validate_serving_input
mlflow.models.validate_serving_input(
    model_uri, 
    serving_input_example
)

# COMMAND ----------

# DBTITLE 1,formatting with ""
# json.dumps(serving_input_example).replace("'",'"')

# COMMAND ----------

# DBTITLE 1,format for UI testing
# {"dataframe_split": {"columns": ["embedding"], "data": [[[0.15414905548095703, 0.008094907738268375, 0.007071635220199823, 0.11320031434297562, -0.026421433314681053, -0.017847076058387756, -0.050769757479429245, 0.0011923464480787516, 0.07798487693071365, -0.0037108196411281824, -0.17938534915447235, -0.00048449428868480027, 0.008351461961865425, 0.00436738133430481, 0.004908162634819746, -0.04016988351941109, -0.011811239644885063, 0.01156391017138958, -0.08166099339723587, 0.0016855493886396289, 0.015247324481606483, 0.01507434993982315, -0.028049129992723465, -0.005492009688168764, -0.12355209141969681, -0.0036010832991451025, 0.047126781195402145, 0.006535788998007774, -0.1449105143547058, -0.011237618513405323, 0.0017997613176703453, 0.004101550206542015, -0.009247836656868458, 0.013190013356506824, 0.04192541167140007, 0.2575031518936157, -0.0009331763722002506, -0.3935891389846802, -0.001351714483462274, 0.005353549961000681, 0.08716312795877457, -0.0016448496608063579, -0.08964385092258453, -0.0020443156827241182, 0.04125256836414337, 0.019619246944785118, -0.07807464152574539, -0.0030157300643622875, -0.03305833414196968, -0.012753852643072605, 0.000960676814429462, 0.0073832436464726925, -0.008370054885745049, -0.005742344539612532, 0.008405598811805248, 0.008876177482306957, 0.021073365584015846, 0.006765063386410475, 0.005318768788129091, 0.00022301881108433008, 0.484632283449173, 0.005790474358946085, -0.07760076224803925, 0.006920183077454567, 0.008430126123130322, 0.0013455881271511316, 0.007747098337858915, 0.4135277569293976, -0.0064574358984827995, -0.0014662531903013587, 0.0041463314555585384, -0.0016529171261936426, -0.011401004157960415, 0.003924323245882988, 0.0028761629946529865, 0.030747858807444572, 0.06868356466293335, 0.013429392129182816, 0.11059071123600006, -0.003301289165392518, 0.007821581326425076, -0.012643265537917614, 0.0006877025007270277, -0.018326250836253166, 0.017164602875709534, 0.00879678688943386, -0.29134342074394226, 0.03959014266729355, 0.011093120090663433, 0.0011855672346428037, 0.003105160780251026, 0.17362508177757263, 0.001208254136145115, -0.0011200892040506005, -0.012471795082092285, -0.012863625772297382, 0.005789272021502256, -0.030835509300231934, -0.013879021629691124, -0.010378285311162472, 0.06195586547255516, -0.11314791440963745, -0.0007368993246927857, -0.03829241916537285, -0.000894454657100141, 0.09568962454795837, -0.07871696352958679, 0.007398643530905247, -0.0059407963417470455, 0.011188282631337643, 0.0020390390418469906, -0.15420489013195038, 0.00025915916194207966, 0.011638288386166096, -0.001341450959444046, 0.005878814030438662, 0.002971096197143197, 0.002188683021813631, -0.0017645112238824368, 0.007092277053743601, -0.014363318681716919, 0.007801887113600969, -0.012165924534201622, 0.001045985845848918, 0.013607310131192207, -0.001561938552185893, 0.016956763342022896, -0.12390629202127457]]]}, "params": {"k": 5}}
