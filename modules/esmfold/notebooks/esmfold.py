# Databricks notebook source

# COMMAND ----------
from src.esmfold.esmfold import ESMFoldPyFunc

# COMMAND ----------
dbutils.widgets.text("catalog", "eswanson_genai", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "esmfold", "Model Name")
dbutils.widgets.text("run_name", "eswanson_esmfold", "Run Name")
dbutils.widgets.text("endpoint_name", "eswanson_esmfold", "Endpoint Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
RUN_NAME = dbutils.widgets.get("run_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")

# COMMAND ----------

# MAGIC %md
# MAGIC #### ESMfold wrapped by mlflow Pyfunc model
# MAGIC   - postprocessing of output to PDB string can be performed as part of the model
# MAGIC   - useful for serving as users do not need to know about this
# MAGIC     - useful to consider for other models if one wants to include other processing steps:
# MAGIC       - e.g. additional relaxation of structures

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download model,tokenizer to the local disk of our compute

# COMMAND ----------
import sys

sys.path.append("../src")


# COMMAND ----------
import mlflow
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

import transformers

print(transformers.__version__)
import accelerate

print(accelerate.__version__)

import mlflow
from mlflow.models import infer_signature
import os

from typing import Any, Dict, List, Optional


# COMMAND ----------
from protein_folding.protein_folding.esmfold import ESMFoldPyFunc

# COMMAND ----------

CACHE_DIR = "/local_disk0/hf_cache/"

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=CACHE_DIR)
model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True, cache_dir=CACHE_DIR
)

# COMMAND ----------

esmfold_model = ESMFoldPyFunc()

# COMMAND ----------

test_input = [
    "MADVQLQESGGGSVQAGGSLRLSCVASGVTSTRPCIGWFRQAPGKEREGVAVVNFRGDSTYITDSVKGRFTISRDEDSDTVYLQMNSLKPEDTATYYCAADVNRGGFCYIEDWYFSYWGQGTQVTVSSAAAHHHHHH"
]
from mlflow.types.schema import ColSpec, Schema

signature = mlflow.models.signature.ModelSignature(
    inputs=Schema([ColSpec(type="string")]),
    outputs=Schema([ColSpec(type="string")]),
    params=None,
)

# COMMAND ----------

del model
del tokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register our model
# MAGIC  - Pass the directory to our local HuggingFace Cache to the artifacts of the mlflow model
# MAGIC    - this places the cache inside the logged model
# MAGIC    - This cache is then used to build the model when model starts up (saves redownloading it)
# MAGIC
# MAGIC  - Tell mlflow we also needed some pip packages for the hosted server compute - this will install only these packages on the container for serving the model

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name=RUN_NAME):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="esmfold",
        python_model=esmfold_model,
        artifacts={
            "cache": CACHE_DIR,
        },
        pip_requirements=[
            "mlflow==2.15.1",
            "cloudpickle==2.2.1",
            "transformers>4.0",
            "torch>2.0",
            "torchvision",  # required for torch (note torch+torchvision without cuda v specified defaults to 12.4 as of Jan 2024)
            "accelerate>0.31",
        ],
        input_example=test_input,
        signature=signature,
        registered_model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serve our model
# MAGIC  - This process can also be achieved in the UI directly from a UC registered model
# MAGIC  - Here we serve the model with code using the databricks-sdk
# MAGIC  - This allows model serving to be achieved with code for easier
# productionization of ML models
# MAGIC  - Serving endpoints can auto capture information about model inputs and outputs
# MAGIC    - These data are then tracked in **inference tables**
# - we turn this feature on here (again, this can also be achieved in the UI)

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServedModelInputWorkloadSize,
    ServedModelInputWorkloadType,
    AutoCaptureConfigInput,
)
from databricks.sdk import errors

w = WorkspaceClient()

endpoint_name = ENDPOINT_NAME

model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
versions = w.model_versions.list(model_name)
latest_version = max(versions, key=lambda v: v.version).version

print("version being served = ", latest_version)


served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        entity_version=latest_version,
        name=MODEL_NAME,
        workload_type="GPU_SMALL",
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]
auto_capture_config = AutoCaptureConfigInput(
    catalog_name=CATALOG,
    schema_name=SCHEMA,
    table_name_prefix=f"{MODEL_NAME}_serving",
    enabled=True,
)

try:
    # try to update the endpoint if already have one
    existing_endpoint = w.serving_endpoints.get(endpoint_name)
    # may take some time to actually do the update
    status = w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=served_entities,
        auto_capture_config=auto_capture_config,
    )
except errors.platform.ResourceDoesNotExist as e:
    # if no endpoint yet, make it, wait for it to spin up, and put model on endpoint
    status = w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=served_entities,
            auto_capture_config=auto_capture_config,
        ),
    )

print(status)

# COMMAND ----------
