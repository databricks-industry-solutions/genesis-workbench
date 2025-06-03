# Databricks notebook source
#%pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.2 torch==2.3.1 transformers==4.41.2 accelerate==0.31.0 mlflow==2.22.0
#%pip install /Volumes/genesis_workbench/dev_srijit_nair_dbx_genesis_workbench_core/libraries/genesis_workbench-0.1.0-py3-none-any.whl --force-reinstall
#dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_name", "esmfold", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "cache_dir", "Cache dir")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
USER_EMAIL = dbutils.widgets.get("user_email")
SQL_WAREHOUSE_ID = dbutils.widgets.get("sql_warehouse_id")
CACHE_DIR = dbutils.widgets.get("cache_dir")

print(f"Cache dir: {CACHE_DIR}")
cache_full_path = f"/Volumes/{CATALOG}/{SCHEMA}/{CACHE_DIR}"
print(f"Cache full path: {cache_full_path}")

# COMMAND ----------

import os

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["SQL_WAREHOUSE"]=SQL_WAREHOUSE_ID
os.environ["IS_TOKEN_AUTH"]="Y"
os.environ["DATABRICKS_TOKEN"]=databricks_token

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
from genesis_workbench.models import (ModelCategory, 
                                      import_model_from_uc,
                                      get_latest_model_version)

from genesis_workbench.workbench import AppContext

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=cache_full_path)
model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True, cache_dir=cache_full_path
)

# COMMAND ----------

class ESMFoldPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        CACHE_DIR = context.artifacts["cache"]

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/esmfold_v1", cache_dir=CACHE_DIR
        )
        self.model = transformers.EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True, cache_dir=CACHE_DIR
        )

        self.model = self.model.cuda()
        self.model.esm = self.model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True

    def _post_process(self, outputs):
        final_atom_positions = (
            transformers.models.esm.openfold_utils.feats.atom14_to_atom37(
                outputs["positions"][-1], outputs
            )
        )
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = transformers.models.esm.openfold_utils.protein.Protein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=(
                    outputs["chain_index"][i] if "chain_index" in outputs else None
                ),
            )
            pdbs.append(transformers.models.esm.openfold_utils.protein.to_pdb(pred))
        return pdbs

    def predict(self, context, model_input: List[str], params=None) -> List[str]:
        tokenized_input = self.tokenizer(
            model_input, return_tensors="pt", add_special_tokens=False, padding=True
        )["input_ids"]
        tokenized_input = tokenized_input.cuda()
        with torch.no_grad():
            output = self.model(tokenized_input)
        pdbs = self._post_process(output)
        return pdbs

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

with mlflow.start_run(run_name=f"register_{MODEL_NAME}"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="esmfold",
        python_model=esmfold_model,
        artifacts={
            "cache": cache_full_path,
        },
        pip_requirements=[
            
            "mlflow==2.15.1",
            "cloudpickle==2.2.1",
            "transformers>4.0",
            "torch>2.0",
            "torchvision", 
            "accelerate>0.31",
        ],
        input_example=test_input,
        signature=signature,
        registered_model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
    )

# COMMAND ----------

model_uc_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
model_version = get_latest_model_version(model_uc_name)
model_uri = f"models:/{model_uc_name}/{model_version}"

app_context = AppContext(
        core_catalog_name=CATALOG,
        core_schema_name=SCHEMA
    )

import_model_from_uc(app_context,user_email=USER_EMAIL,
                    model_category=ModelCategory.PROTEIN_STUDIES,
                    model_uc_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
                    model_uc_version=model_version,
                    model_name="ESMFold",
                    model_display_name="ESMFold",
                    model_source_version="v2.0",
                    model_description_url="https://github.com/facebookresearch/esm?tab=readme-ov-file#evolutionary-scale-modeling")

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import (
#     EndpointCoreConfigInput,
#     ServedEntityInput,
#     ServedModelInputWorkloadSize,
#     ServedModelInputWorkloadType,
#     AutoCaptureConfigInput,
# )
# from databricks.sdk import errors

# w = WorkspaceClient()

# endpoint_name = ENDPOINT_NAME

# model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
# versions = w.model_versions.list(model_name)
# latest_version = max(versions, key=lambda v: v.version).version

# print("version being served = ", latest_version)


# served_entities = [
#     ServedEntityInput(
#         entity_name=model_name,
#         entity_version=latest_version,
#         name=MODEL_NAME,
#         workload_type="GPU_SMALL",
#         workload_size="Small",
#         scale_to_zero_enabled=True,
#     )
# ]
# auto_capture_config = AutoCaptureConfigInput(
#     catalog_name=CATALOG,
#     schema_name=SCHEMA,
#     table_name_prefix=f"{MODEL_NAME}_serving",
#     enabled=True,
# )

# try:
#     # try to update the endpoint if already have one
#     existing_endpoint = w.serving_endpoints.get(endpoint_name)
#     # may take some time to actually do the update
#     status = w.serving_endpoints.update_config(
#         name=endpoint_name,
#         served_entities=served_entities,
#         auto_capture_config=auto_capture_config,
#     )
# except errors.platform.ResourceDoesNotExist as e:
#     # if no endpoint yet, make it, wait for it to spin up, and put model on endpoint
#     status = w.serving_endpoints.create_and_wait(
#         name=endpoint_name,
#         config=EndpointCoreConfigInput(
#             name=endpoint_name,
#             served_entities=served_entities,
#             auto_capture_config=auto_capture_config,
#         ),
#     )

# print(status)
