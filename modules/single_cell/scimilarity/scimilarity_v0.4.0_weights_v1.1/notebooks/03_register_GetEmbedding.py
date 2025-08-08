# Databricks notebook source
# MAGIC %md
# MAGIC #### Run Initialization

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Define Custom PyFunc for: `SCimilarity_GetEmbedding`

# COMMAND ----------

import csv
from typing import Any, Dict
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModelContext
from scimilarity import CellEmbedding, CellQuery
import torch

class SCimilarity_GetEmbedding(mlflow.pyfunc.PythonModel):
    r"""Create MLFlow Pyfunc class for SCimilarity embedding model."""

    def load_context(self, context: PythonModelContext):
        r"""Intialize pre-trained SCimilarity embedding model."""
        self.ce = CellEmbedding(context.artifacts["model_path"]) 
        
    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,         
    ) -> pd.DataFrame:
        r"""Output prediction on model."""

        final_results = []

        for index, row in model_input.iterrows():
            celltype_sample_json = row['celltype_sample']
            celltype_sample_obs_json = row.get('celltype_sample_obs', None)

            # Load DataFrames and preserve indices
            celltype_sample_df = pd.read_json(celltype_sample_json, orient='split')
            if celltype_sample_obs_json:
                celltype_sample_obs_df = pd.read_json(celltype_sample_obs_json, orient='split')

            embeddings_list = []

            for sample_index, sample_row in celltype_sample_df.iterrows():
                model_input_array = np.array(sample_row['celltype_subsample'], dtype=np.float64).reshape(1, -1)
                embedding = self.ce.get_embeddings(model_input_array)
                embeddings_list.append({
                    'input_index': index,
                    'celltype_sample_index': sample_index,
                    'embedding': embedding.tolist()
                })

            embedding_df = pd.DataFrame(embeddings_list)
            embedding_df.index = embedding_df['celltype_sample_index']
            embedding_df.index.name = None

            # Merge DataFrames
            # combined_df = pd.merge(celltype_sample_df, embedding_df, left_index=True, right_index=True)
            combined_df = embedding_df
            if celltype_sample_obs_json:
                combined_df = pd.merge(combined_df, celltype_sample_obs_df, left_index=True, right_index=True)

            final_results.append(combined_df)

        output_df = pd.concat(final_results).reset_index(drop=True)

        # Reorder columns
        if 'celltype_sample_obs_df' in locals():
            # columns_order = ['input_index', 'celltype_sample_index'] + list(celltype_sample_obs_df.columns) + ['celltype_subsample', 'embedding']
            columns_order = ['input_index', 'celltype_sample_index'] + list(celltype_sample_obs_df.columns) + ['embedding']
        else:
            # columns_order = ['input_index', 'celltype_sample_index', 'celltype_subsample', 'embedding']
            columns_order = ['input_index', 'celltype_sample_index', 'embedding']
        
        output_df = output_df[columns_order]

        # Convert input_index to string
        output_df['input_index'] = output_df['input_index'].astype(str)

        final_output = output_df.copy()
        for col in final_output.select_dtypes(include=['int']).columns:
            final_output[col] = final_output[col].astype(float)
        
        return final_output
        

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
model = SCimilarity_GetEmbedding()
model.load_context(temp_context)

# COMMAND ----------

# DBTITLE 1,Specify model_input
X_vals: sparse.csr_matrix = celltype_sample.X
# print(X_vals)

# X_vals is a sparse matrix
X_vals_dense = X_vals.toarray()

celltype_sample_obs_json = celltype_sample.obs.to_json(orient='split')

celltype_subsample_pdf = pd.DataFrame([{'celltype_subsample': row} for row in X_vals_dense ], 
                                      index=celltype_sample.obs.index
                                     )

celltype_subsample_pdf.iloc[0:1], celltype_subsample_pdf.iloc[1:]

model_input_tmp0 = pd.DataFrame([{
    "celltype_sample": celltype_subsample_pdf.iloc[0:1].to_json(orient='split'), 
    "celltype_sample_obs": celltype_sample.obs.iloc[0:1].to_json(orient='split')
}])

model_input_tmp1 = pd.DataFrame([{
    "celltype_sample": celltype_subsample_pdf.iloc[1:].to_json(orient='split'), 
    "celltype_sample_obs": celltype_sample.obs.iloc[1:].to_json(orient='split')
}])

# Concatenate the DataFrame with itself and reset the index
model_input = pd.concat([model_input_tmp0, model_input_tmp1], axis=0, ignore_index=True)

# Drop the 'index' column if it exists
if 'index' in model_input.columns:
    model_input.drop(columns=['index'], inplace=True)

model_input

# COMMAND ----------

# DBTITLE 1,Test model_input
# Call the predict method

embedding_output = model.predict(temp_context, model_input)
embedding_output

# COMMAND ----------

embedding_output1 = model.predict(temp_context, pd.DataFrame([model_input.iloc[0,:].to_dict()]))
embedding_output1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define MLflow Signature with local Model + Context

# COMMAND ----------

# DBTITLE 1,Define MLflow Signature
from mlflow.models import infer_signature

# Define a concrete example input as a Pandas DataFrame
example_input = model_input

# Ensure the example output is in a serializable format
example_output = embedding_output # from model.predict(temp_context, model_input)

## Initial Inference for the model signature --> return all fields are (required) --> we need to update the model signature
signature0 = infer_signature(
    model_input=example_input,
    model_output=example_output,    
)

# COMMAND ----------

optionalCols_list = list(set(example_output.columns.to_list()) - set(['original_index',
                                                                      'celltype_sample_index',
                                                                      # 'cell_subsample',
                                                                      'embedding']))
optionalCols_list

# COMMAND ----------

optionalCols_dtypes = example_output[optionalCols_list].dtypes


# COMMAND ----------

# DBTITLE 1,Manually specify/update dtypes for optionalCols_list
## Update example_input/output with optional fields

# Check and delete example_input_with_optionalCols if it exists
try:
    del example_input_with_optionalCols
except NameError:
    pass

example_input_with_optionalCols = example_input.copy()
# example_input_with_optionalCols["celltype_sample_obs"] = pd.Series([None],  dtype="string")
example_input_with_optionalCols["celltype_sample_obs"] = pd.Series([None, "value"], dtype="object")

# Check and delete example_output_with_optionalCols if it exists
try:
    del example_output_with_optionalCols
except NameError:
    pass

example_output_with_optionalCols = example_output.copy()
example_output_with_optionalCols["celltype_raw"] = pd.Series([None, "value"], dtype="string")
example_output_with_optionalCols["celltype_id"] = pd.Series([None, "value"], dtype="string")
example_output_with_optionalCols["sample"] = pd.Series([None, "value"], dtype="string")
example_output_with_optionalCols["study"] = pd.Series([None, "value"], dtype="string")
example_output_with_optionalCols["n_counts"] = pd.Series([None, 0.0], dtype="float64")
example_output_with_optionalCols["n_genes"] = pd.Series([None, 0.0], dtype="float64")
example_output_with_optionalCols["celltype_name"] = pd.Series([None, "value"], dtype="string")
example_output_with_optionalCols["doublet_score"] = pd.Series([None, 0.0], dtype="float64")
example_output_with_optionalCols["pred_dbl"] = pd.Series([None, False], dtype="boolean")
example_output_with_optionalCols["Disease"] = pd.Series([None, "value"], dtype="string")

# COMMAND ----------

# DBTITLE 1,example_input_with_optionalCols
import numpy as np
import pandas as pd

# Define a function to handle arrays properly
def handle_array(x):
    if isinstance(x, np.ndarray):
        return np.where(pd.isna(x), np.nan, x)
    else:        
        return None if pd.isna(x) or np.nan else x ## None is required for Any | more flexible for truly optional

# Apply the function element-wise using np.vectorize
example_input_with_optionalCols["celltype_sample_obs"] = example_input_with_optionalCols["celltype_sample_obs"].apply(
    handle_array #lambda x: None if pd.isna(x) else x
)

example_input_with_optionalCols

# COMMAND ----------

# DBTITLE 1,example_output_with_optionalCols
import numpy as np
import pandas as pd

# Define a function to handle arrays properly
def handle_array(x):
    if isinstance(x, np.ndarray):
        return np.where(pd.isna(x), np.nan, x)
    else:
        # return None if pd.isna(x) or np.nan else x ## None is required for Any | more flexible
        return None if pd.isna(x) else x ## this works ok

# List of optional columns
optionalCols_list = [
    "celltype_raw", "celltype_id", "sample", "study", "n_counts", "n_genes",
    "celltype_name", "doublet_score", "pred_dbl", "Disease"
]

# Apply the function element-wise to all optional columns
example_output_with_optionalCols[optionalCols_list] = example_output_with_optionalCols[optionalCols_list].applymap(handle_array)

example_output_with_optionalCols

# COMMAND ----------

signature = infer_signature(example_input_with_optionalCols, example_output_with_optionalCols)
signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_GetEmbedding`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "Get_Embedding" ##
model_name= f"{MODEL_NAME}_{MODEL_TYPE}".lower()
experiment = set_mlflow_experiment(experiment_tag=EXPERIMENT_NAME, user_email=USER_EMAIL)

# COMMAND ----------

# DBTITLE 1,log SCimilarity_GetEmbeddings
# Save and log the model
with mlflow.start_run(run_name=f'{model_name}', experiment_id=experiment.experiment_id) as run:
    mlflow.pyfunc.log_model(        
        artifact_path=f"{MODEL_TYPE}", 
        python_model=model,
        artifacts={
            "model_path": model_path,  
        },
        input_example=model_input, 
        signature=signature,
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
