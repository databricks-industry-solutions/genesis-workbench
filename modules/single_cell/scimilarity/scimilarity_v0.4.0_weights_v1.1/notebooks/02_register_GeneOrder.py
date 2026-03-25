# Databricks notebook source
# MAGIC %md
# MAGIC #### Run Initialization
# MAGIC

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils

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
example_input = pd.DataFrame(
    {
        'input': ["get_gene_order"]  # Just to trigger getting GeneOrder list from model weights folder
    })

# Ensure the example output is in a serializable format
example_output = model.predict(example_input)

# Infer the model signature
signature = infer_signature(example_input, example_output)

# COMMAND ----------

signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow Log Custom PyFunc: `SCimilarity_GeneOrder`

# COMMAND ----------

# DBTITLE 1,create experiment_dir
MODEL_TYPE = "Gene_Order"
model_name= f"{MODEL_NAME}_{MODEL_TYPE}".lower()
experiment = set_mlflow_experiment(experiment_tag=EXPERIMENT_NAME, user_email=USER_EMAIL)

# COMMAND ----------

# DBTITLE 1,Log SCimilarity_GeneOrder
import os 
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

# Save and log the model
with mlflow.start_run(run_name=model_name, experiment_id=experiment.experiment_id) as run:
    mlflow.pyfunc.log_model(
        artifact_path=f"{MODEL_TYPE}", # artifact_path --> "name" mlflow v3.0
        python_model=model, 
        artifacts={
                   "geneOrder_path": geneOrder_path ## defined in ./utils 
                  },
        input_example=example_input,
        signature=signature,
        pip_requirements=[
            "setuptools<82"
        ],
        registered_model_name=f"{CATALOG}.{SCHEMA}.{model_name}"
    )

    run_id = run.info.run_id
    print("Model logged with run ID:", run_id)
