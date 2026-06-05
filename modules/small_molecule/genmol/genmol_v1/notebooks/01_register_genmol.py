# Databricks notebook source

# MAGIC %md
# MAGIC ### GenMol — generative small-molecule design
# MAGIC
# MAGIC GenMol (NVIDIA) is a masked-diffusion generator over SAFE fragment strings —
# MAGIC a generalist for de novo generation, scaffold decoration, fragment linking,
# MAGIC and lead optimization. It closes the small-molecule "where do candidate
# MAGIC ligands come from?" gap: **GenMol generates** candidates → DiffDock docks them
# MAGIC into the target → Chemprop/KERMT profile ADMET.
# MAGIC
# MAGIC This notebook downloads the open weights `nvidia/NV-GenMol-89M-v2`, wraps
# MAGIC generation in an MLflow PyFunc, registers it in Unity Catalog, and deploys a
# MAGIC serving endpoint via Genesis Workbench.
# MAGIC
# MAGIC **Runtime:** runs on a **classic DBR 15.4 LTS (Python 3.11)** cluster (see
# MAGIC register_genmol.yml). GenMol pins `pandas==2.1.0` / `transformers==4.52.4`,
# MAGIC which only have wheels on py3.11 — on serverless py3.12 the pandas source
# MAGIC build fails. Logging here on py3.11 also makes the serving endpoint py3.11.
# MAGIC
# MAGIC **Licensing:** weights = NVIDIA Open Model License (commercial OK; not for
# MAGIC life-critical use); GenMol code = Apache-2.0.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "genmol", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "045df48d4afed522", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "genmol", "Cache dir")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# Pinned to the commit validated on a py3.11 cluster (Sampler API + data/len.pk
# behaviour). Bump deliberately, re-validating the wrapper, not silently.
GENMOL_COMMIT = "add09fc83b7255bd09c797e527c0f4b51f5fb7c1"
GENMOL_GIT = f"git+https://github.com/NVIDIA-Digital-Bio/GenMol.git@{GENMOL_COMMIT}"
LEN_PK_URL = f"https://raw.githubusercontent.com/NVIDIA-Digital-Bio/GenMol/{GENMOL_COMMIT}/data/len.pk"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies
# MAGIC GenMol is installed **last** so its hard pins (torch 2.6.0, transformers
# MAGIC 4.52.4, pandas 2.1.0, huggingface_hub) win and the package imports cleanly —
# MAGIC installing it before mlflow/gwb let those pull an incompatible transformers/hub.

# COMMAND ----------

# MAGIC %pip install mlflow[databricks]==2.22.0 databricks-sdk==0.50.0 databricks-sql-connector==4.0.3

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if(lib.name.startswith("genesis_workbench")):
        gwb_library_path = lib.path.replace("dbfs:","")
print(gwb_library_path)

# COMMAND ----------

# MAGIC %pip install {gwb_library_path}

# COMMAND ----------

# GenMol last — pulls its own pinned torch/transformers/pandas/safe-mol/etc.
# MAGIC %pip install {GENMOL_GIT}

# COMMAND ----------

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

GENMOL_COMMIT = "add09fc83b7255bd09c797e527c0f4b51f5fb7c1"
GENMOL_GIT = f"git+https://github.com/NVIDIA-Digital-Bio/GenMol.git@{GENMOL_COMMIT}"
LEN_PK_URL = f"https://raw.githubusercontent.com/NVIDIA-Digital-Bio/GenMol/{GENMOL_COMMIT}/data/len.pk"

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print("Cache full path:", cache_full_path)

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# Initialize Genesis Workbench
from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the open checkpoint + the length-prior (`data/len.pk`)
# MAGIC GenMol's `de_novo_generation` reads `data/len.pk` from a path computed as
# MAGIC `dirname×3(sampler.py)` — present in the source repo but NOT in the pip wheel.
# MAGIC We download it and ship it as a model artifact; the PyFunc places it back at
# MAGIC that exact path in `load_context`.

# COMMAND ----------

import os, urllib.request

weights_dir = os.path.join(cache_full_path, "weights")
os.makedirs(weights_dir, exist_ok=True)

from huggingface_hub import snapshot_download
snapshot_download(repo_id="nvidia/NV-GenMol-89M-v2", local_dir=weights_dir)
print("weights:", os.listdir(weights_dir))

checkpoint_path = os.path.join(weights_dir, "model_v2.ckpt")
assert os.path.exists(checkpoint_path), f"checkpoint missing: {os.listdir(weights_dir)}"

len_pk_path = os.path.join(cache_full_path, "len.pk")
urllib.request.urlretrieve(LEN_PK_URL, len_pk_path)
print("checkpoint:", checkpoint_path, "| len.pk:", len_pk_path, os.path.getsize(len_pk_path), "bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate the PyFunc model code (Models-from-Code)
# MAGIC The PyFunc lives in `genmol_model.py` alongside this notebook. We log it by
# MAGIC file path (`python_model=<path>`) rather than as a pickled instance, because
# MAGIC mlflow.pyfunc cannot reliably cloudpickle a notebook __main__ class.

# COMMAND ----------

import os
import mlflow

_nb_dir = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
model_code_path = f"/Workspace{_nb_dir}/genmol_model.py"
assert os.path.exists(model_code_path), f"model code not found: {model_code_path}"
print("model code:", model_code_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Smoke-test the wrapper before logging

# COMMAND ----------

import importlib.util
import pandas as pd

_spec = importlib.util.spec_from_file_location("genmol_model", model_code_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

class _Ctx:
    artifacts = {"checkpoint": checkpoint_path, "len_pk": len_pk_path}

_gen = _mod.GenMolGenerator()
_gen.load_context(_Ctx())
_smoke = _gen.predict(None, pd.DataFrame({"fragment": [""]}), params={"num_molecules": 4})
print("smoke-test generated molecules:")
print(_smoke)
assert not _smoke.empty, "GenMol produced no molecules in smoke test"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Signature

# COMMAND ----------

from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

input_schema = Schema([ColSpec(type="string", name="fragment")])
output_schema = Schema([
    ColSpec(type="string", name="seed"),
    ColSpec(type="string", name="smiles"),
    ColSpec(type="double", name="score"),
])
param_schema = ParamSchema([
    ParamSpec("num_molecules", "integer", 20),
    ParamSpec("temperature", "double", 1.0),
    ParamSpec("randomness", "double", 1.0),
    ParamSpec("scoring", "string", "qed"),
    ParamSpec("unique", "boolean", True),
])
signature = mlflow.models.signature.ModelSignature(
    inputs=input_schema, outputs=output_schema, params=param_schema)

input_example = pd.DataFrame({"fragment": ["", "c1ccccc1C(=O)N"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register in Unity Catalog, import into Genesis Workbench, deploy

# COMMAND ----------

from genesis_workbench.models import (ModelCategory,
                                      import_model_from_uc,
                                      get_latest_model_version,
                                      deploy_model,
                                      set_mlflow_experiment)
from genesis_workbench.workbench import wait_for_job_run_completion

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
        artifact_path="genmol",
        python_model=model_code_path,
        artifacts={
            "checkpoint": checkpoint_path,
            "len_pk": len_pk_path,
        },
        # The serving endpoint inherits this env. GenMol's git install pulls its
        # pinned torch/transformers/pandas/safe-mol; rdkit is for our scoring.
        # NOTE: a VCS (git+) requirement needs git in the serving build image
        # (available on Databricks Model Serving). If a future image lacks it,
        # switch to logging the built genmol wheel as an artifact instead.
        pip_requirements=[
            GENMOL_GIT,
            "rdkit==2025.3.6",
        ],
        input_example=input_example,
        signature=signature,
        registered_model_name=f"{catalog}.{schema}.{model_name}",
    )

# COMMAND ----------

model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(user_email=user_email,
                    model_category=ModelCategory.SMALL_MOLECULE,
                    model_uc_name=model_uc_name,
                    model_uc_version=model_version,
                    model_name="GenMol",
                    model_display_name="GenMol Molecule Generator",
                    model_source_version="NV-GenMol-89M-v2",
                    model_description_url="https://huggingface.co/nvidia/NV-GenMol-89M-v2")

# COMMAND ----------

run_id = deploy_model(user_email=user_email,
                gwb_model_id=gwb_model_id,
                deployment_name=f"GenMol",
                deployment_description="GenMol masked-diffusion generator for de novo / fragment-based small-molecule design",
                input_adapter_str="none",
                output_adapter_str="none",
                sample_input_data_dict_as_json="none",
                sample_params_as_json="none",
                workload_type=workload_type,
                workload_size="Small")

# COMMAND ----------

result = wait_for_job_run_completion(run_id, timeout=3600)
