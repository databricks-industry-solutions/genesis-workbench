# Databricks notebook source
# MAGIC %md
# MAGIC ### DeepSTABp Thermostability (Tm) Predictor
# MAGIC
# MAGIC Registers the [DeepSTABp](https://github.com/CSBiology/deepStabP) protein melting
# MAGIC temperature predictor into MLflow / Unity Catalog and deploys a GPU_SMALL serving
# MAGIC endpoint via Genesis Workbench.
# MAGIC
# MAGIC **License:** MIT (verified at upstream repo + LICENSE file).
# MAGIC **Backbone:** ProtT5-XL (`Rostlab/prot_t5_xl_uniref50`, ~3 GB) +
# MAGIC `deepSTAPpMLP` head (4-layer MLP, ~80 MB Lightning checkpoint from upstream).
# MAGIC **Output:** `predicted_tm_celsius` — melting temperature regression in °C.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The `deepSTAPpMLP` class is vendored inline below from
# MAGIC `src/Api/app/predictor.py` of the upstream MIT-licensed repo, with attribution.
# MAGIC The MLP checkpoint is downloaded once at registration time via raw GitHub URL —
# MAGIC ProtT5 backbone via HuggingFace `from_pretrained`. No manual upload step.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "deepstabp_v1", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "deepstabp", "Cache dir (UC volume)")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")
dbutils.widgets.text("hf_backbone_id", "Rostlab/prot_t5_xl_uniref50", "ProtT5 backbone HF id")
dbutils.widgets.text(
    "upstream_ckpt_url",
    "https://github.com/CSBiology/deepStabP/raw/main/src/Api/trained_model/b25_sampled_10k_tuned_2_d01/checkpoints/epoch%3D1-step%3D2316.ckpt",
    "Upstream MLP checkpoint URL",
)

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies (exact pins)

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC     transformers==4.46.3 \
# MAGIC     safetensors==0.4.5 \
# MAGIC     huggingface-hub==0.26.2 \
# MAGIC     pytorch-lightning==2.5.5 \
# MAGIC     sentencepiece==0.2.0 \
# MAGIC     biopython==1.84 \
# MAGIC     numpy==1.26.4 \
# MAGIC     pandas==1.5.3 \
# MAGIC     mlflow==2.22.0 \
# MAGIC     cloudpickle==2.0.0 \
# MAGIC     databricks-sdk==0.50.0 \
# MAGIC     databricks-sql-connector==4.0.2

# COMMAND ----------

import os, sys, shutil, json, urllib.request
import numpy as np
import pandas as pd
import mlflow

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")
workload_type = dbutils.widgets.get("workload_type")
hf_backbone_id = dbutils.widgets.get("hf_backbone_id")
upstream_ckpt_url = dbutils.widgets.get("upstream_ckpt_url")

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache volume:  {cache_full_path}")
print(f"HF backbone:   {hf_backbone_id}")
print(f"MLP ckpt URL:  {upstream_ckpt_url}")

# COMMAND ----------

gwb_library_path = None
libraries = dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")
for lib in libraries:
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(f"Genesis Workbench library wheel: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, shutil, json, urllib.request
import numpy as np
import pandas as pd
import mlflow

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")
workload_type = dbutils.widgets.get("workload_type")
hf_backbone_id = dbutils.widgets.get("hf_backbone_id")
upstream_ckpt_url = dbutils.widgets.get("upstream_ckpt_url")
cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(
    core_catalog_name=catalog,
    core_schema_name=schema,
    sql_warehouse_id=sql_warehouse_id,
    token=databricks_token,
)

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download artifacts: ProtT5 backbone + DeepSTABp MLP checkpoint

# COMMAND ----------

ARTIFACTS_DIR = "/tmp/deepstabp_artifacts"
PROTT5_DIR = os.path.join(ARTIFACTS_DIR, "prot_t5_xl_uniref50")
CKPT_PATH = os.path.join(ARTIFACTS_DIR, "deepstabp_mlp.ckpt")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PROTT5_DIR, exist_ok=True)

from huggingface_hub import snapshot_download

# ProtT5-XL (~3 GB) — auto-download via HF
snapshot_download(
    repo_id=hf_backbone_id,
    local_dir=PROTT5_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json", "*.safetensors", "pytorch_model.bin",
        "tokenizer.json", "tokenizer_config.json", "spiece.model",
        "special_tokens_map.json", "added_tokens.json",
    ],
)
print("ProtT5 files:")
for f in sorted(os.listdir(PROTT5_DIR)):
    sz_mb = os.path.getsize(os.path.join(PROTT5_DIR, f)) / (1024 * 1024)
    print(f"  {f}  ({sz_mb:.1f} MB)")

# MLP checkpoint (~80 MB) — fetch raw from upstream GitHub
if not os.path.exists(CKPT_PATH) or os.path.getsize(CKPT_PATH) == 0:
    print(f"Downloading MLP checkpoint from {upstream_ckpt_url}")
    urllib.request.urlretrieve(upstream_ckpt_url, CKPT_PATH)
sz_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
print(f"MLP checkpoint: {sz_mb:.1f} MB at {CKPT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate the PyFunc wrapper module
# MAGIC
# MAGIC `DeepSTABpModel` lives in `deepstabp_wrapper.py` next to this notebook.
# MAGIC Code-based logging (passing the wrapper's absolute path) avoids
# MAGIC cloudpickling a class defined in the notebook's `__main__` scope.

# COMMAND ----------

_notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
WRAPPER_PATH = "/Workspace" + os.path.dirname(_notebook_path) + "/deepstabp_wrapper.py"
print(f"Wrapper module: {WRAPPER_PATH}")
assert os.path.exists(WRAPPER_PATH), f"Wrapper file missing at {WRAPPER_PATH}"

# Constants duplicated from the wrapper for the MLflow log_model call below.
MODEL_MAX_SEQ_LEN = 1024
MODEL_CLASS_NAME = "DeepSTABpModel"

# COMMAND ----------

conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "python=3.11",
        "pip",
        {
            "pip": [
                "torch==2.7.1",
                "transformers==4.46.3",
                "safetensors==0.4.5",
                "huggingface-hub==0.26.2",
                "pytorch-lightning==2.5.5",
                "sentencepiece==0.2.0",
                "biopython==1.84",
                "numpy==1.26.4",
                "pandas==1.5.3",
                "mlflow==2.22.0",
                "cloudpickle==2.0.0",
            ]
        },
    ],
    "name": "deepstabp_env",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the PyFunc model into Unity Catalog

# COMMAND ----------

from genesis_workbench.models import set_mlflow_experiment

set_mlflow_experiment(
    experiment_tag=experiment_name,
    user_email=user_email,
    host=None,
    token=None,
    shared=True,
)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

uc_model_name = f"{catalog}.{schema}.{model_name}"

example_input = pd.DataFrame({
    "sequence": ["MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADMEDVCGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQIRLQAEAFQARLKSWFEPLVEDMQRQWAGLVEKVQAAVGTSAAPVPSDNH"],
    "growth_temp": [37.0],
    "mt_mode": ["Cell"],
})

from mlflow.models import infer_signature

example_output = pd.DataFrame({
    "sequence": ["MKVL..."],
    "predicted_tm_celsius": [60.0],
})
example_signature = infer_signature(example_input, example_output)

with mlflow.start_run(run_name=f"register-{model_name}") as run:
    mlflow.log_params({
        "model_class": MODEL_CLASS_NAME,
        "upstream_repo": "https://github.com/CSBiology/deepStabP",
        "license": "MIT",
        "hf_backbone_id": hf_backbone_id,
        "backbone": "ProtT5-XL UniRef50",
        "max_seq_len": MODEL_MAX_SEQ_LEN,
        "ckpt_url": upstream_ckpt_url,
    })

    logged = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WRAPPER_PATH,
        artifacts={"artifacts_dir": ARTIFACTS_DIR},
        conda_env=conda_env,
        signature=example_signature,
        input_example=example_input,
        registered_model_name=uc_model_name,
    )

    print(f"Registered as {uc_model_name}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Model URI: {logged.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import into Genesis Workbench and deploy serving endpoint

# COMMAND ----------

from genesis_workbench.models import (
    ModelCategory,
    import_model_from_uc,
    get_latest_model_version,
    deploy_model,
)
from genesis_workbench.workbench import wait_for_job_run_completion

model_version = get_latest_model_version(uc_model_name)
print(f"UC model version: {model_version}")

gwb_model_id = import_model_from_uc(
    user_email=user_email,
    model_category=ModelCategory.SMALL_MOLECULE,
    model_uc_name=uc_model_name,
    model_uc_version=model_version,
    model_name="DeepSTABp Tm",
    model_display_name="DeepSTABp Thermostability (Tm) Predictor",
    model_source_version="v1.0 (CSBiology deepStabP, MIT)",
    model_description_url="https://github.com/CSBiology/deepStabP",
)

deploy_run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="DeepSTABp Thermostability Predictor",
    deployment_description="DeepSTABp ProtT5-XL + MLP head melting-temperature regression (MIT). Returns predicted Tm in °C; takes optional growth_temp (default 37) and mt_mode ('Cell' or 'Lysate', default 'Cell') columns.",
    input_adapter_str="none",
    output_adapter_str="none",
    sample_input_data_dict_as_json="none",
    sample_params_as_json="none",
    workload_type=workload_type,
    workload_size="Small",
)
print(f"Deploy run ID: {deploy_run_id}")

# COMMAND ----------

result = wait_for_job_run_completion(deploy_run_id, timeout=3600)
print(f"Deployment finished: {result}")
