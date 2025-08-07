# Databricks notebook source
# MAGIC %md
# MAGIC #### Run Initialization

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils 

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

# DBTITLE 1,check signature
signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "Search_Nearest" 
model_name= f"{MODEL_NAME}_{MODEL_TYPE}".lower()
experiment = set_mlflow_experiment(experiment_tag=EXPERIMENT_NAME, user_email=USER_EMAIL)

# COMMAND ----------

# DBTITLE 1,log SCimilarity_SearchNearest
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

# Save and log the model
with mlflow.start_run(run_name=f'{model_name}', experiment_id=experiment.experiment_id) as run:
    mlflow.pyfunc.log_model(
        artifact_path=f"{MODEL_TYPE}", 
        python_model=model, 
        artifacts={
                    "model_path": model_path,  
                  },    
        input_example = example_input,        
        signature = signature, 
        pip_requirements=[
            "mlflow==2.22.0",
            "cloudpickle==2.0.0",
            "scanpy==1.11.2",
            "numcodecs==0.13.1",
            "scimilarity==0.4.0",
            "pandas==1.5.3",
            "numpy==1.26.4"
        ],        
        registered_model_name=f"{CATALOG}.{SCHEMA}.{model_name}" 
    )

    run_id = run.info.run_id
    print("Model logged with run ID:", run_id)
