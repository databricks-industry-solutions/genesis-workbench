# Databricks notebook source
# MAGIC %md
# MAGIC ### NetSolP-1.0 Solubility Predictor
# MAGIC
# MAGIC Registers the [NetSolP-1.0](https://github.com/tvinet/NetSolP-1.0) protein solubility predictor
# MAGIC into MLflow / Unity Catalog and deploys a CPU serving endpoint via Genesis Workbench.
# MAGIC
# MAGIC **License:** BSD-3-Clause (verified at upstream repo).
# MAGIC **Backbone:** ESM-12 (5-fold ensemble, split 0 only — 85 MB), ONNX Runtime inference (no PyTorch on the serving endpoint).
# MAGIC **Output:** `predicted_solubility` — probability in [0, 1] that the protein is soluble in *E. coli*.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Weight distribution
# MAGIC
# MAGIC The two NetSolP files (`Solubility_ESM12_0_quantized.onnx`, `ESM12_alphabet.pkl`) are
# MAGIC bundled with this submodule under `weights/` and redistributed under their
# MAGIC original BSD-3-Clause license (see `weights/LICENSE.NETSOLP`). They are
# MAGIC uploaded to the Databricks workspace by `databricks bundle deploy` and read
# MAGIC from the workspace path passed via the `weights_path` job parameter
# MAGIC (default: `/Workspace${current_working_directory}/files/weights`).
# MAGIC
# MAGIC If `weights/` is empty on a fresh clone, populate it with
# MAGIC `bash modules/small_molecule/netsolp/netsolp_v1/weights/extract_weights.sh
# MAGIC <path-to-netsolp-1.0.ALL.tar.gz>` and commit before running this job.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "netsolp_v1", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "netsolp", "Cache dir (UC volume)")
dbutils.widgets.text("workload_type", "CPU", "Workload Type for endpoints")
dbutils.widgets.text("weights_path", "", "Bundled weights path (set by DAB)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies (exact pins)

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC     onnxruntime==1.20.1 \
# MAGIC     fair-esm==2.0.0 \
# MAGIC     torch==2.7.1 \
# MAGIC     numpy==1.26.4 \
# MAGIC     pandas==1.5.3 \
# MAGIC     mlflow==2.22.0 \
# MAGIC     cloudpickle==2.0.0 \
# MAGIC     databricks-sdk==0.50.0 \
# MAGIC     databricks-sql-connector==4.0.2

# COMMAND ----------

import os, sys, shutil, tempfile, json, pickle
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
weights_path = dbutils.widgets.get("weights_path")

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache volume:   {cache_full_path}")
print(f"Bundled weights: {weights_path}")

# COMMAND ----------

# Resolve and install the pre-built genesis_workbench library wheel from the libraries volume.
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

import os, sys, shutil, tempfile, json, pickle
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
weights_path = dbutils.widgets.get("weights_path")
cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(
    core_catalog_name=catalog,
    core_schema_name=schema,
    sql_warehouse_id=sql_warehouse_id,
    token=databricks_token,
)

# COMMAND ----------

# Ensure the UC volume exists (used at registration time as a runtime cache).
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")
print(f"Volume {catalog}.{schema}.{cache_dir} ready at {cache_full_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-flight: verify bundled weights are present
# MAGIC
# MAGIC The two files are committed under `weights/` in the submodule and uploaded
# MAGIC to `${weights_path}` by `databricks bundle deploy`. If they're missing on a
# MAGIC fresh clone, run `weights/extract_weights.sh` before deploying — see
# MAGIC `weights/README.md` in this submodule.

# COMMAND ----------

REQUIRED_FILES = {
    "onnx": "Solubility_ESM12_0_quantized.onnx",
    "alphabet": "ESM12_alphabet.pkl",
}
# We ship a single split of the upstream 5-fold ESM-12 ensemble (split 0). The
# 5-split full ensemble is more accurate but adds ~340 MB more to the repo;
# the distilled ESM-1b model is 652 MB (over GitHub's 100 MB hard limit).

missing = []
for tag, filename in REQUIRED_FILES.items():
    path = os.path.join(weights_path, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        missing.append(f"  - {filename}  (expected at {path})")

if missing:
    msg = (
        "\n\nNetSolP bundled weights not found. Files missing:\n"
        + "\n".join(missing)
        + "\n\nThe weights are shipped under "
        "modules/small_molecule/netsolp/netsolp_v1/weights/ in the GWB repo. "
        "On a fresh clone, populate them once with:\n"
        "  bash modules/small_molecule/netsolp/netsolp_v1/weights/extract_weights.sh "
        "<path-to-netsolp-1.0.ALL.tar.gz>\n"
        "then commit before re-running this job."
    )
    print(msg)
    dbutils.notebook.exit(msg)

print(f"All required NetSolP weight files are present at {weights_path}")

# COMMAND ----------

# Stage the bundled weights into a local artifacts dir so mlflow.log_model can
# capture them. The bundled path lives under /Workspace/... which mlflow can't
# always traverse efficiently for log_model artifacts.
ARTIFACTS_DIR = "/tmp/netsolp_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

for filename in REQUIRED_FILES.values():
    src = os.path.join(weights_path, filename)
    dst = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  staged {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  {filename} already staged")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate the PyFunc wrapper module
# MAGIC
# MAGIC The `NetSolPSolubilityModel` class lives in `netsolp_wrapper.py` next to
# MAGIC this notebook. We use MLflow's code-based logging (passing the wrapper's
# MAGIC absolute path to `python_model=`) instead of cloudpickling a class
# MAGIC defined in the notebook's `__main__` scope — that surfaced a brittle
# MAGIC `IndexError` inside `cloudpickle.dump` for this class.

# COMMAND ----------

_notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
WRAPPER_PATH = "/Workspace" + os.path.dirname(_notebook_path) + "/netsolp_wrapper.py"
print(f"Wrapper module: {WRAPPER_PATH}")
assert os.path.exists(WRAPPER_PATH), f"Wrapper file missing at {WRAPPER_PATH}"

# Constants duplicated from the wrapper for the MLflow log_model call below.
# Keep in sync if the wrapper's class attributes change.
MODEL_MAX_SEQ_LEN = 1022
MODEL_BACKBONE = "ESM-12 (5-fold ensemble split 0)"
MODEL_CLASS_NAME = "NetSolPSolubilityModel"

# COMMAND ----------

# Conda env that the serving endpoint will use. Exact pins per the GWB dependency rule.
conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "python=3.11",
        "pip",
        {
            "pip": [
                "onnxruntime==1.20.1",
                "fair-esm==2.0.0",
                # fair-esm 2.0.0's setup.py doesn't pull torch transitively
                # (verified empirically — MLflow's loader-test subprocess
                # ImportErrors on torch at log_model time without this pin).
                "torch==2.7.1",
                "numpy==1.26.4",
                "pandas==1.5.3",
                "mlflow==2.22.0",
                "cloudpickle==2.0.0",
            ]
        },
    ],
    "name": "netsolp_env",
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

# UC requires an explicit model signature; with code-based logging MLflow
# can't infer it from input_example alone. Build it from a synthetic output
# that matches the wrapper's predict() return schema.
from mlflow.models import infer_signature

example_output = pd.DataFrame({
    "sequence": ["MKVL..."],
    "predicted_solubility": [0.5],
})
example_signature = infer_signature(example_input, example_output)

with mlflow.start_run(run_name=f"register-{model_name}") as run:
    mlflow.log_params({
        "model_class": MODEL_CLASS_NAME,
        "upstream_repo": "https://github.com/tvinet/NetSolP-1.0",
        "license": "BSD-3-Clause",
        "backbone": MODEL_BACKBONE,
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
    model_name="NetSolP Solubility",
    model_display_name="NetSolP-1.0 Solubility Predictor",
    model_source_version="v1.0 (DTU NetSolP-1.0)",
    model_description_url="https://github.com/tvinet/NetSolP-1.0",
)

deploy_run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="NetSolP-1.0 Solubility Predictor",
    deployment_description="NetSolP-1.0 distilled ONNX solubility predictor (BSD-3-Clause). Returns probability that the input protein is soluble in E. coli.",
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
