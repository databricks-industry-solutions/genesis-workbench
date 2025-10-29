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
import mlflow
import numpy as np
import pandas as pd
import json
from mlflow.pyfunc.model import PythonModelContext
from scimilarity import CellQuery
from typing import Any, Dict

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
        # Extract embedding - handle both list and array formats
        embeddings = model_input["embedding"].iloc[0]
        if isinstance(embeddings, str):
            embeddings = json.loads(embeddings)
        embeddings = np.array(embeddings)

        # Handle optional params column as JSON string
        params = self._parse_params(model_input)
        k = params.get("k", 100) #If "k" is not present in params, use 100 as the default value.

        # Perform search
        predictions = self.cq.search_nearest(embeddings, k=k)

        # Format results
        results_dict = {
            "nn_idxs": [np_array.tolist() for np_array in predictions[0]],
            "nn_dists": [np_array.tolist() for np_array in predictions[1]],
            "results_metadata": json.dumps(predictions[2].to_dict())  # Serialize to JSON string
        }
        
        return pd.DataFrame([results_dict])

    def _parse_params(self, model_input: pd.DataFrame) -> Dict[str, Any]:
        """Parse optional params column with robust error handling."""
        default_params = {"k": 100}
        
        # Check if params column exists
        if "params" not in model_input.columns:
            return default_params
        
        raw_params = model_input["params"].iloc[0]
        
        # Handle None, NaN, or empty string
        if raw_params is None or (isinstance(raw_params, float) and pd.isna(raw_params)) or raw_params == "":
            return default_params
        
        # Parse JSON string
        try:
            params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
            # Validate k parameter
            if "k" in params and isinstance(params["k"], (int, float)) and params["k"] > 0:
                return params
            else:
                return default_params
        except (json.JSONDecodeError, TypeError, ValueError):
            return default_params

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

# DBTITLE 1,specify params
from typing import Optional, Any

params: Optional[dict[str, Any]] = dict({"k": 10})
params.values()

# COMMAND ----------

# DBTITLE 1,Specify example model_input
## Create a DataFrame containing the embeddings

# cell_embeddings.dtype #dtype('float32')
# cell_embeddings.shape #(1, 128)

# model_input = pd.DataFrame([{"embedding": cell_embeddings.tolist()[0]}]) # previously where params was separate signature.

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

# DBTITLE 1,arrow warning (?)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# COMMAND ----------

# DBTITLE 1,Test predict with model_input + params
# Call the predict method
# searchNearest_output = model.predict(temp_context, model_input, params={"k": 100})

searchNearest_output = model.predict(temp_context, model_input) 

display(searchNearest_output)

# COMMAND ----------

# DBTITLE 1,check prediction output
import json
import pandas as pd

# JSON string
json_string = searchNearest_output["results_metadata"].iloc[0]

# Parse JSON and convert to DataFrame
data = json.loads(json_string)
df = pd.DataFrame(data)


# Display the DataFrame
display(df)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# COMMAND ----------

# DBTITLE 1,test without params
model_input0 = pd.DataFrame([
    {
        "embedding": cell_embeddings.tolist()[0],
        # "params": {"k": 100}
    }
])
display(model_input0)

# COMMAND ----------

# DBTITLE 1,predict without params input
output = model.predict(temp_context, model_input0)
output

# COMMAND ----------

# DBTITLE 1,check that where params not specified -- k defaults to 100
# JSON string
json_string0 = output["results_metadata"].iloc[0]

# Parse JSON and convert to DataFrame
data0 = json.loads(json_string0)
df0 = pd.DataFrame(data0)
display(df0)

# COMMAND ----------

# DBTITLE 1,different input testing
# # Test with params
# test_input_with_params = pd.DataFrame([{
#     "embedding": np.random.rand(256).tolist(),
#     "params": json.dumps({"k": 50})
# }])

# # Test without params column
# test_input_no_params = pd.DataFrame([{
#     "embedding": np.random.rand(256).tolist()
# }])

# # Test with None params
# test_input_none_params = pd.DataFrame([{
#     "embedding": np.random.rand(256).tolist(),
#     "params": None
# }])

# # Test with invalid params (should fallback to default)
# test_input_invalid_params = pd.DataFrame([{
#     "embedding": np.random.rand(256).tolist(),
#     "params": "invalid json"
# }])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define MLflow Signature with local Model + Context

# COMMAND ----------

# DBTITLE 1,Define MLflow Signature
# from mlflow.models import infer_signature
# import pandas as pd

# # Define a concrete example input as a Pandas DataFrame
# example_input = model_input.copy() 
# ## we will add params separately to keep it simple... but make a note on the usage input patterns 

# # Ensure the example output is in a serializable format
# example_output = searchNearest_output

# # Create a Dict for params
# # params: dict[str, Any] = dict({"k": 100}) ## could take any dict and if none provided defaults to example provided
# # params: Optional[dict[str, Any]] = dict({"k": 100})

# # # Infer the model signature
# signature = infer_signature(
#     model_input = model_input, #example_input,
#     model_output = example_output,
#     # params=params
# )

# COMMAND ----------

# DBTITLE 1,Define MLflow Signature
import numpy as np
import pandas as pd
import json
from mlflow.models import infer_signature

# Define handle_array function
def handle_array(x):
    if isinstance(x, np.ndarray):
        return np.where(pd.isna(x), np.nan, x)
    else:
        return None if pd.isna(x) else x

# Create base example_input
example_input = model_input.copy() 

# Create base example_output
example_output = searchNearest_output

# Create example_input_with_optionalCols
example_input_with_optionalCols = example_input.copy()
example_input_with_optionalCols["params"] = pd.Series(
    [None, json.dumps({"k": 100})], 
    dtype="object"
)

# Apply handle_array to optional input columns
example_input_with_optionalCols["params"] = example_input_with_optionalCols["params"].apply(handle_array)

# Create example_output_with_optionalCols (even if no optional output columns)
example_output_with_optionalCols = example_output.copy()

# Infer signature
signature = infer_signature(model_input = example_input_with_optionalCols, 
                            model_output = example_output_with_optionalCols,
                            #params ## omit -- move into input^
                           )
# print(signature)

# COMMAND ----------

# DBTITLE 1,check signature
signature

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflow LOG Custom PyFunc: `SCimilarity_SearchNearest`

# COMMAND ----------

# DBTITLE 1,Specify MODEL_TYPE & experiment_name
# MODEL_TYPE = "Search_Nearest" ## 
# # model_name = f"SCimilarity_{MODEL_TYPE}"  
# model_name = f"{MODEL_NAME}_{MODEL_TYPE}"  

# ## Set the experiment
# user_path = f"/Users/{USER_EMAIL}"
# # experiment_dir = f"{user_path}/mlflow_experiments/{MODEL_FAMILY}" ## TO UPDATE
# experiment_dir = f"{user_path}/mlflow_experiments/{EXPERIMENT_NAME}" ## same as MODEL_FAMILY from widget above
# print(experiment_dir)

# # experiment_name = f"{user_path}/mlflow_experiments/{MODEL_FAMILY}/{MODEL_TYPE}"
# experiment_name = f"{experiment_dir}/{MODEL_TYPE}"
# print(experiment_name)

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

# COMMAND ----------

# MAGIC %md
# MAGIC