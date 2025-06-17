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

# example_input , example_output

# COMMAND ----------

signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_GeneOrder`

# COMMAND ----------

MODEL_NAME, MODEL_FAMILY, EXPERIMENT_NAME

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "Gene_Order" # 
model_name = f"{MODEL_FAMILY}_{MODEL_TYPE}" # f"SCimilarity_{MODEL_TYPE}" 

## Set the experiment
experiment_dir = f"{user_path}/mlflow_experiments/{EXPERIMENT_NAME}" ## same as MODEL_FAMILY from widget in utils
print(experiment_dir)

experiment_name = f"{experiment_dir}/{model_name}"
# experiment_name = f"{experiment_dir}" # model_name will be designated as experiment runs
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

mlflow.set_experiment(experiment_id=exp_id) #


# Save and log the model
# with mlflow.start_run(run_name=f'{model_name}', experiment_id=exp_id) as run:
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        # artifact_path=f"{MODEL_TYPE}",
        name=f"{MODEL_TYPE}",
        python_model=model, 
        artifacts={
            "geneOrder_path": geneOrder_path
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
# model_name = f"SCimilarity_{MODEL_TYPE}" 
model_name = f"{MODEL_FAMILY}_{MODEL_TYPE}"  
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
# add_model_alias(full_model_name, "Champion")

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
# MAGIC
# MAGIC ```
# MAGIC ## workload types&sizes
# MAGIC workload_type = "CPU"
# MAGIC workload_size = "Small"
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,example input for inferencing
# {
#   "dataframe_split": {
#     "columns": [
#       "input"
#     ],
#     "data": [
#       [
#         "get_gene_order"
#       ]
#     ]
#   }
# }
