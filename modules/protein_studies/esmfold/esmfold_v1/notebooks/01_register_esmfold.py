# Databricks notebook source
# MAGIC %md
# MAGIC ### Installing dependencies

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_name", "esmfold", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "esm2_cache_dir", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

#requirements for genesis workbench library
%pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.2 #mlflow==2.22.0
#requirements for current library
#%pip install -r ../requirements.txt

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")

print(gwb_library_path)

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

#Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name = catalog, core_schema_name = schema, sql_warehouse_id = sql_warehouse_id, token = databricks_token)


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
                                      get_latest_model_version,
                                      deploy_model,
                                      set_mlflow_experiment)

from genesis_workbench.workbench import wait_for_job_run_completion

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=cache_full_path)
model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True, cache_dir=cache_full_path
)

# COMMAND ----------

class ESMFoldPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        cache_dir = context.artifacts["cache"]

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/esmfold_v1", cache_dir=cache_dir
        )
        self.model = transformers.EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True, cache_dir=cache_dir
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
mlflow.set_tracking_uri("databricks")

experiment = set_mlflow_experiment(experiment_tag=experiment_name, 
                                   user_email=user_email,
                                   host=None,
                                   token=None,
                                   shared=True)

with mlflow.start_run(run_name=f"{model_name}", experiment_id=experiment.experiment_id):
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
                    model_name="ESMFold",
                    model_display_name="ESMFold",
                    model_source_version="v1.0",
                    model_description_url="https://github.com/facebookresearch/esm?tab=readme-ov-file#evolutionary-scale-modeling")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"ESMFold",
                deployment_description="Initial deployment",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout = 3600)
