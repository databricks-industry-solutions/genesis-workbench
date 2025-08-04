# Databricks notebook source
# MAGIC %md
# MAGIC #### Setup: `%run ./utils` 

# COMMAND ----------

# DBTITLE 1,install/load dependencies | # ~5mins (including initial data processing)
# MAGIC %run ./utils_20250801

# COMMAND ----------

# torch.__version__ #'2.7.1+cu126' WRT custom pyfunc 

# COMMAND ----------

CATALOG, DB_SCHEMA, MODEL_FAMILY, MODEL_NAME, EXPERIMENT_NAME

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Define Custom PyFunc for: `SCimilarity_GetEmbedding`

# COMMAND ----------

# torch.__version__ #'2.7.1+cu126'

# COMMAND ----------

# DBTITLE 1,Define SCimilarity_GetEmbedding
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

# DBTITLE 1,Define Local TempContext
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

# MAGIC %md
# MAGIC ##### Format and specify `model_input`

# COMMAND ----------

# DBTITLE 1,Initial Data Processing from utils_*
# diseasetype_ipf, celltype_myofib, celltype_sample, celltype_subsample, X_vals

# COMMAND ----------

# DBTITLE 1,!! UPDATE sparse matrix X_vals to use multiple subsamples
# print(celltype_sample) ## include multiple rows of samples for selected disease-celltype

## use all subsamples from celltype_sample (Nd array or list)
X_vals: sparse.csr_matrix = celltype_sample.X
# print(X_vals)

# X_vals is a sparse matrix
X_vals_dense = X_vals.toarray()
# print(X_vals_dense)

# COMMAND ----------

# DBTITLE 1,description: celltype_sample.obs
# celltype_sample.obs

# COMMAND ----------

# DBTITLE 1,we will want to preserve the celltype_sample.index
# celltype_sample.obs.index

# COMMAND ----------

# DBTITLE 1,cell_sample_obs_json
## derive jsons as column inputs to custom pyfunc model

celltype_sample_obs_json = celltype_sample.obs.to_json(orient='split')
# celltype_sample_obs_json

# COMMAND ----------

# DBTITLE 1,cell_subsample
## preserve the index from original data 

celltype_subsample_pdf = pd.DataFrame([{'celltype_subsample': row} for row in X_vals_dense ], 
                                      index=celltype_sample.obs.index
                                     )
# celltype_subsample_pdf

# COMMAND ----------

# DBTITLE 1,generate examples from sample data
celltype_subsample_pdf.iloc[0:1], celltype_subsample_pdf.iloc[1:]

# COMMAND ----------

# DBTITLE 1,Specify model_input
## model_input with a single row 
# model_input = pd.DataFrame([{
#     "celltype_sample": celltype_subsample_pdf.iloc[0:2].to_json(orient='split'), 
#     "celltype_sample_obs": celltype_sample.obs.iloc[0:2].to_json(orient='split')
# }])
# model_input


## model_input with multiple rows
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

# pd.read_json(model_input["celltype_sample"].iloc[0], orient='split')

# COMMAND ----------

# pd.read_json(model_input["celltype_sample_obs"].iloc[0], orient='split')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test model_input

# COMMAND ----------

# DBTITLE 1,testing optional obs col in pdDF model input
embedding_output = model.predict(temp_context, model_input)

embedding_output 

# COMMAND ----------

# embedding_output.info()

# COMMAND ----------

# model_input[['celltype_sample']]

# COMMAND ----------

# DBTITLE 1,predict for embedding_output
embedding_output = model.predict(temp_context, model_input[['celltype_sample']])

embedding_output

# COMMAND ----------

# DBTITLE 1,Test model_inputs : dtype options
embedding_output = model.predict(temp_context, model_input)
embedding_output

# COMMAND ----------

# DBTITLE 1,extract 1 row
# pd.DataFrame([model_input.iloc[0,:].to_dict()])

# COMMAND ----------

# # # Call the predict method

embedding_output1 = model.predict(temp_context, pd.DataFrame([model_input.iloc[0,:].to_dict()]))

embedding_output1

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define MLflow Signature with local Model + Context

# COMMAND ----------

# DBTITLE 1,include subsample index/extra info. | disease/type / cell type / water
model_input

# COMMAND ----------

# DBTITLE 1,FINAL output
embedding_output

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
# signature0

# COMMAND ----------

# DBTITLE 1,Specify optionalCols_list
optionalCols_list = list(set(example_output.columns.to_list()) - set(['original_index',
                                                                      'celltype_sample_index',
                                                                      # 'cell_subsample',
                                                                      'embedding']))
optionalCols_list

# COMMAND ----------

# DBTITLE 1,Check datatypes for optionalCols_list
# Get the data types for the optional columns
optionalCols_dtypes = example_output[optionalCols_list].dtypes
# display(optionalCols_dtypes)

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

# DBTITLE 1,Updated signature inference
## Infer the model signature using input/output wiht optionalCols
signature = infer_signature(example_input_with_optionalCols, example_output_with_optionalCols)
signature


# COMMAND ----------

# DBTITLE 1,ref signature outputs
# inputs: 
#   ['celltype_sample': string (required), 'celltype_sample_obs': Any (optional)]
# outputs: 
#   ['original_index': string (required), 'celltype_sample_index': string (required), 'celltype_raw': Any (optional), 'celltype_id': Any (optional), 'sample': Any (optional), 'study': Any (optional), 'n_counts': Any (optional), 'n_genes': Any (optional), 'celltype_name': Any (optional), 'doublet_score': Any (optional), 'pred_dbl': Any (optional), 'Disease': Any (optional), {{'celltype_subsample': Array(double) (required)}}, 'embedding': Array(Array(double)) (required)]
# params: 
#   None


# inputs: 
#   ['celltype_sample': string (required), 'celltype_sample_obs': Any (optional)]
# outputs: 
#   ['original_index': string (required), 'celltype_sample_index': string (required), 'celltype_raw': string (optional), 'celltype_id': string (optional), 'sample': string (optional), 'study': string (optional), 'n_counts': double (optional), 'n_genes': double (optional), 'celltype_name': string (optional), 'doublet_score': double (optional), 'pred_dbl': boolean (optional), 'Disease': string (optional), {{'cell_subsample': Array(double) (required)}}, 'embedding': Array(Array(double)) (required)]
# params: 
#   None

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_GetEmbedding`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
MODEL_TYPE = "Get_Embedding" ##
# model_name = f"SCimilarity_{MODEL_TYPE}"  
model_name = f"{MODEL_NAME}_{MODEL_TYPE}"

## Set the experiment
experiment_dir = f"{user_path}/mlflow_experiments/{EXPERIMENT_NAME}" ## same as MODEL_FAMILY from widget above
print(experiment_dir)

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

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,specify mlflow requirements.txt
import os

# Create a requirements.txt file with the necessary dependencies
requirements = """
mlflow==2.22.0
cloudpickle==2.0.0
scanpy==1.11.2
numcodecs==0.13.1
scimilarity==0.4.0
pandas==1.5.3
numpy==1.26.4 
"""

# MODEL_TYPE = "Get_Embeddings"
# model_name = f"SCimilarity_{MODEL_TYPE}"  #
model_name = f"{MODEL_NAME}_{MODEL_TYPE}" 

# Define the path to save the requirements file in the UV volumes
SCimilarity_GetEmbeddings_requirements_path = f"/Volumes/{CATALOG}/{DB_SCHEMA}/{MODEL_FAMILY}/mlflow_requirements/{model_name}/requirements.txt"
# SCimilarity_GetEmbeddings_requirements_path = f"/Volumes/mmt/genesiswb/scimilarity/mlflow_requirements/{model_name}/requirements.txt"

# Create the directory if it does not exist
os.makedirs(os.path.dirname(SCimilarity_GetEmbeddings_requirements_path), exist_ok=True)

# Write the requirements to the file
with open(SCimilarity_GetEmbeddings_requirements_path, "w") as f:
    f.write(requirements)

print(f"Requirements written to {SCimilarity_GetEmbeddings_requirements_path}")

# COMMAND ----------

# DBTITLE 1,log SCimilarity_GetEmbeddings
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
            "model_path": model_path,  ## defined in ./utils 
        },
        input_example=model_input, #example_input
        
        # signature = infer_signature(model_input, model_output), # for previous testing
        signature=signature, ## use manually updated dtypes
        
        pip_requirements=SCimilarity_GetEmbeddings_requirements_path,
        
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
loaded_model.input_example 

# COMMAND ----------

# DBTITLE 1,Test logged + loaded model prediction
loaded_model.predict(loaded_model.input_example) 
print(loaded_model)

# COMMAND ----------

# DBTITLE 1,review example_input
example_input
# example_input[["celltype_sample"]], 
# example_input[['celltype_sample']].iloc[[0]]

# COMMAND ----------

# DBTITLE 1,test with example_input etc.
loaded_model.predict(example_input[["celltype_sample"]].iloc[[0]])

# COMMAND ----------

# DBTITLE 1,test with "local" model_input
loaded_model.predict(model_input)

# COMMAND ----------

model_input.info()
loaded_model.predict(model_input).info()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### UC Register Custom PyFunc: `SCimilarity_GetEmbedding`

# COMMAND ----------

# DBTITLE 1,Model Info
## Register the model
# model_name = f"SCimilarity_{MODEL_TYPE}"  
model_name = f"{MODEL_NAME}_{MODEL_TYPE}" 
full_model_name = f"{CATALOG}.{DB_SCHEMA}.{model_name}"
model_uri = f"runs:/{run_id}/{MODEL_TYPE}"

model_name, full_model_name, model_uri

# COMMAND ----------

# DBTITLE 1,register SCimilarity_GetEmbeddings
# registered_model = 
mlflow.register_model(model_uri=model_uri, 
                      name=full_model_name,                      
                      await_registration_for=120,
                    )

# COMMAND ----------

# DBTITLE 1,extract meta_data
## Load the model
# model = mlflow.pyfunc.load_model(model_uri)

# # Get model metadata
# model_metadata = model.metadata
# print("Model Metadata:", model_metadata)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Deploy & Serve UC registered model: `SCimilarity_GetEmbeddings`
# MAGIC
# MAGIC ```
# MAGIC ## AWS workload types&sizes
# MAGIC # https://docs.databricks.com/api/workspace/servingendpoints/create#config-served_models-workload_type
# MAGIC # workload_type = "GPU_MEDIUM" ## deployment timeout!
# MAGIC workload_type = "MULTIGPU_MEDIUM"  # 4xA10G
# MAGIC workload_size = "Medium"
# MAGIC ```
# MAGIC
# MAGIC ---        
# MAGIC         
# MAGIC ```
# MAGIC ## AzureDB workload types&sizes
# MAGIC # https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints
# MAGIC workload_type = "CPU" # seems to work else "GPU_LARGE" 
# MAGIC workload_size = "Small" 0-4 concurrency 
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,example input for UI inferencing
# https://adb-830292400663869.9.azuredatabricks.net/ml/experiments/332441420736406/runs/d40fbd7bdeb044d996a3113ed65312ab/artifacts?o=830292400663869
