# Databricks notebook source
# MAGIC %md
# MAGIC ### ESM2 Embeddings Model Registration
# MAGIC Registers Facebook's ESM-2 650M model as an MLflow PyFunc that produces
# MAGIC 1280-dimensional mean-pooled sequence embeddings for vector similarity search.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "esm2_embeddings", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "esm2_embeddings", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# requirements for genesis workbench library
%pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.2 transformers==4.41.2 accelerate==0.31.0 hf_transfer

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:","")

print(f"GWB library: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")
workload_type = dbutils.widgets.get("workload_type")

print(f"Cache dir: {cache_dir}")
cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download ESM-2 model to local cache

# COMMAND ----------

import mlflow
import torch
from transformers import AutoTokenizer, AutoModel

from typing import List
from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      get_latest_model_version,
                                      deploy_model,
                                      set_mlflow_experiment)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

MODEL_TAG = "facebook/esm2_t33_650M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(MODEL_TAG, cache_dir=cache_full_path)
model = AutoModel.from_pretrained(MODEL_TAG, cache_dir=cache_full_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ESM2 Embedding PyFunc
# MAGIC Takes amino acid sequences as input, returns 1280-dimensional mean-pooled embeddings.
# MAGIC This is the same embedding logic used in the batch embedding pipeline to ensure
# MAGIC vector space consistency between indexed and queried embeddings.

# COMMAND ----------

import transformers

class ESM2EmbeddingPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        cache_dir = context.artifacts["cache"]

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir
        )
        self.model = transformers.AutoModel.from_pretrained(
            "facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir
        )

        self.model = self.model.cuda().eval()
        torch.backends.cuda.matmul.allow_tf32 = True

    def predict(self, context, model_input: List[str], params=None) -> List[List[float]]:
        results = []
        for seq in model_input:
            tokens = self.tokenizer(
                seq, return_tensors="pt", truncation=True, max_length=1024
            ).to("cuda")
            with torch.no_grad():
                output = self.model(**tokens)
            # Mean pool over sequence positions, excluding BOS/EOS tokens
            embedding = output.last_hidden_state[0, 1:-1].mean(dim=0).cpu().tolist()
            results.append(embedding)
        return results

# COMMAND ----------

esm2_embedding_model = ESM2EmbeddingPyFunc()

# COMMAND ----------

test_input = [
    "MADVQLQESGGGSVQAGGSLRLSCVASGVTSTRPCIGWFRQAPGKEREGVAVVNFRGDSTYITDSVKGRFTISRDEDSDTVYLQMNSLKPEDTATYYCAADVNRGGFCYIEDWYFSYWGQGTQVTVSSAAAHHHHHH"
]

signature = mlflow.models.infer_signature(
    model_input=test_input,
    model_output=[[0.0] * 1280],
)

# COMMAND ----------

del model
del tokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the model
# MAGIC - Pass the HuggingFace cache directory as an artifact so the model is self-contained
# MAGIC - Specify pip requirements for the serving container

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

experiment = set_mlflow_experiment(experiment_tag=experiment_name,
                                   user_email=user_email,
                                   host=None,
                                   token=None,
                                   shared=True)

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="esm2_embeddings",
        python_model=esm2_embedding_model,
        artifacts={
            "cache": cache_full_path,
        },
        pip_requirements=[
            "--extra-index-url https://download.pytorch.org/whl/cu121",
            "mlflow==2.15.1",
            "cloudpickle==3.0.0",
            "transformers==4.41.2",
            "torch==2.3.1+cu121",
            "torchvision==0.18.1+cu121",
            "accelerate==0.31.0",
            "setuptools<82"
        ],
        input_example=test_input,
        signature=signature,
        registered_model_name=f"{catalog}.{schema}.{model_name}",
    )

# COMMAND ----------

model_uc_name=f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.PROTEIN_STUDIES,
                    model_uc_name=f"{catalog}.{schema}.{model_name}",
                    model_uc_version=model_version,
                    model_name="ESM2 Embeddings",
                    model_display_name="ESM2 Embeddings",
                    model_source_version="v1.0",
                    model_description_url="https://huggingface.co/facebook/esm2_t33_650M_UR50D")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"ESM2 Embeddings",
                deployment_description="ESM-2 650M mean-pooled sequence embeddings for vector similarity search",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout = 3600)
