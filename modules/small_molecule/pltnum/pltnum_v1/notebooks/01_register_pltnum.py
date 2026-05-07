# Databricks notebook source
# MAGIC %md
# MAGIC ### PLTNUM-ESM2 Half-Life Relative Stability Predictor
# MAGIC
# MAGIC Registers the [PLTNUM](https://github.com/sagawatatsuya/PLTNUM) ESM-2 half-life
# MAGIC predictor (HuggingFace checkpoint `sagawa/PLTNUM-ESM2-NIH3T3`) into MLflow / Unity
# MAGIC Catalog and deploys a GPU_SMALL serving endpoint via Genesis Workbench.
# MAGIC
# MAGIC **License:** MIT (verified at upstream repo + HuggingFace model card).
# MAGIC **Backbone:** ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) + 1280→1 linear head.
# MAGIC **Output:** `predicted_stability` — sigmoid prob in [0, 1]. **Relative ranker, not hours.** The
# MAGIC enzyme-optimization loop turns this into a real half-life signal by anchoring against
# MAGIC user-supplied reference enzymes with known half-life values; see the loop's reward
# MAGIC composer for the soft-prior calibration.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The `PLTNUM_PreTrainedModel` class is vendored inline below from `scripts/models.py`
# MAGIC of the upstream MIT-licensed repo, with attribution. No `git clone` step at
# MAGIC registration time. Weights download from HuggingFace via `from_pretrained` —
# MAGIC fully automated, no manual upload step.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "pltnum_v1", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "pltnum", "Cache dir (UC volume)")
dbutils.widgets.text("workload_type", "GPU_SMALL", "Workload Type for endpoints")
dbutils.widgets.text("hf_model_id", "sagawa/PLTNUM-ESM2-NIH3T3", "HuggingFace model id")

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
# MAGIC     numpy==1.26.4 \
# MAGIC     pandas==1.5.3 \
# MAGIC     mlflow==2.22.0 \
# MAGIC     cloudpickle==2.0.0 \
# MAGIC     databricks-sdk==0.50.0 \
# MAGIC     databricks-sql-connector==4.0.2

# COMMAND ----------

import os, sys, shutil, json
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
hf_model_id = dbutils.widgets.get("hf_model_id")

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache volume: {cache_full_path}")
print(f"HF model id:  {hf_model_id}")

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

import os, sys, shutil, json
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
hf_model_id = dbutils.widgets.get("hf_model_id")
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
# MAGIC ### Download HuggingFace checkpoint into a local artifacts dir

# COMMAND ----------

from huggingface_hub import snapshot_download
import shutil

ARTIFACTS_DIR = "/tmp/pltnum_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Download the full checkpoint with retry on transient corruption. This repo
# only ships pytorch_model.bin (~2.6 GB); the HF Hub consistency check
# occasionally fails mid-download when the cluster's network hiccups, so we
# wipe and retry up to 3 times before giving up.
def _fetch_with_retry(repo_id, local_dir, max_attempts=3):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[attempt {attempt}/{max_attempts}] snapshot_download({repo_id}) → {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                max_workers=4,
                force_download=(attempt > 1),  # retry: wipe + re-download
            )
            return
        except OSError as e:
            last_err = e
            print(f"[attempt {attempt}] failed: {e}")
            for entry in os.listdir(local_dir):
                p = os.path.join(local_dir, entry)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    raise RuntimeError(f"HF download failed after {max_attempts} attempts: {last_err}")

_fetch_with_retry(hf_model_id, ARTIFACTS_DIR)
print("Downloaded files:")
for f in sorted(os.listdir(ARTIFACTS_DIR)):
    sz_mb = os.path.getsize(os.path.join(ARTIFACTS_DIR, f)) / (1024 * 1024)
    print(f"  {f}  ({sz_mb:.1f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate the PyFunc wrapper module
# MAGIC
# MAGIC `PLTNUMHalfLifeModel` lives in `pltnum_wrapper.py` next to this notebook.
# MAGIC Code-based logging (passing the wrapper's absolute path to `python_model=`)
# MAGIC avoids cloudpickling a class defined in the notebook's `__main__` scope.

# COMMAND ----------

_notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
WRAPPER_PATH = "/Workspace" + os.path.dirname(_notebook_path) + "/pltnum_wrapper.py"
print(f"Wrapper module: {WRAPPER_PATH}")
assert os.path.exists(WRAPPER_PATH), f"Wrapper file missing at {WRAPPER_PATH}"

# Constants duplicated from the wrapper for the MLflow log_model call below.
MODEL_MAX_SEQ_LEN = 1024
MODEL_CLASS_NAME = "PLTNUMHalfLifeModel"

# COMMAND ----------

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
                "numpy==1.26.4",
                "pandas==1.5.3",
                "mlflow==2.22.0",
                "cloudpickle==2.0.0",
            ]
        },
    ],
    "name": "pltnum_env",
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
    "sequence": ["MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADMEDVCGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQIRLQAEAFQARLKSWFEPLVEDMQRQWAGLVEKVQAAVGTSAAPVPSDNH"]
})

# UC requires an explicit signature with code-based logging.
from mlflow.models import infer_signature

example_output = pd.DataFrame({
    "sequence": ["MKVL..."],
    "predicted_stability": [0.5],
})
example_signature = infer_signature(example_input, example_output)

with mlflow.start_run(run_name=f"register-{model_name}") as run:
    mlflow.log_params({
        "model_class": MODEL_CLASS_NAME,
        "upstream_repo": "https://github.com/sagawatatsuya/PLTNUM",
        "license": "MIT",
        "hf_model_id": hf_model_id,
        "backbone": "ESM-2 650M (facebook/esm2_t33_650M_UR50D)",
        "max_seq_len": MODEL_MAX_SEQ_LEN,
    })

    logged = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WRAPPER_PATH,
        artifacts={"weights_dir": ARTIFACTS_DIR},
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
    model_name="PLTNUM Half-Life Stability",
    model_display_name="PLTNUM-ESM2 Half-Life Relative Stability Ranker",
    model_source_version="v1.0 (HF sagawa/PLTNUM-ESM2-NIH3T3)",
    model_description_url="https://github.com/sagawatatsuya/PLTNUM",
)

deploy_run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="PLTNUM-ESM2 Half-Life Relative Stability Ranker",
    deployment_description="PLTNUM ESM-2 relative-stability ranker (MIT). Returns sigmoid probability in [0,1] — higher means predicted longer in-vivo half-life. Anchor against a known reference enzyme to derive a usable threshold (see enzyme optimization loop).",
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
