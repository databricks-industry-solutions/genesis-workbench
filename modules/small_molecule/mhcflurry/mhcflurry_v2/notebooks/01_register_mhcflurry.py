# Databricks notebook source
# MAGIC %md
# MAGIC ### MHCflurry 2.x Immunogenicity Predictor
# MAGIC
# MAGIC Registers [MHCflurry 2.x](https://github.com/openvax/mhcflurry) (Apache-2.0,
# MAGIC PyTorch backend, ~150 MB models) into MLflow / Unity Catalog and deploys a CPU
# MAGIC serving endpoint via Genesis Workbench.
# MAGIC
# MAGIC **License:** Apache-2.0 (verified at upstream repo).
# MAGIC **Predictor:** `Class1PresentationPredictor` — pan-allele MHC-I peptide presentation.
# MAGIC **Output:**
# MAGIC - `predicted_immuno_burden` — count of "strong-presentation" 9-mer epitopes
# MAGIC   (`presentation_score >= 0.5`) per residue across the user-supplied HLA panel.
# MAGIC   Lower = more developable.
# MAGIC - `max_presentation_score` — worst-case single epitope across the panel.
# MAGIC
# MAGIC The wrapper scans the input protein with a sliding 9-mer window and scores every
# MAGIC peptide × allele combination, then aggregates. Default allele panel covers ~95 %
# MAGIC of the global population (Sette-style 6-allele HLA-A/B mix); override via the
# MAGIC `alleles` input column.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "mhcflurry_v2", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "mhcflurry", "Cache dir (UC volume)")
dbutils.widgets.text("workload_type", "CPU", "Workload Type for endpoints")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies (exact pins)

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC     mhcflurry==2.2.1 \
# MAGIC     torch==2.7.1 \
# MAGIC     numpy==1.26.4 \
# MAGIC     pandas==2.2.3 \
# MAGIC     scikit-learn==1.5.2 \
# MAGIC     biopython==1.84 \
# MAGIC     mlflow==2.22.0 \
# MAGIC     cloudpickle==2.0.0 \
# MAGIC     databricks-sdk==0.50.0 \
# MAGIC     databricks-sql-connector==4.0.2

# COMMAND ----------

import os, sys, shutil, json, subprocess
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

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache volume: {cache_full_path}")

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

import os, sys, shutil, json, subprocess
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
# MAGIC ### Download MHCflurry models bundle

# COMMAND ----------

ARTIFACTS_DIR = "/tmp/mhcflurry_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Point mhcflurry at our artifact dir for the download.
os.environ["MHCFLURRY_DATA_DIR"] = ARTIFACTS_DIR

print("Running 'mhcflurry-downloads fetch'...")
result = subprocess.run(
    ["mhcflurry-downloads", "fetch", "models_class1_presentation"],
    capture_output=True, text=True, timeout=900,
)
print("STDOUT:", result.stdout[-2000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"mhcflurry-downloads fetch failed: {result.returncode}")

# Print final layout for sanity.
for root, _, files in os.walk(ARTIFACTS_DIR):
    rel = os.path.relpath(root, ARTIFACTS_DIR)
    if rel == ".":
        rel = ""
    for f in files[:5]:
        sz_mb = os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
        print(f"  {os.path.join(rel, f)}  ({sz_mb:.1f} MB)")
    if len(files) > 5:
        print(f"  ... ({len(files) - 5} more files in {rel or '.'})")

# COMMAND ----------

# Smoke-test the predictor in this notebook before packaging.
from mhcflurry import Class1PresentationPredictor

_smoke = Class1PresentationPredictor.load()
_smoke_out = _smoke.predict(
    peptides=["SIINFEKL", "GILGFVFTL"],
    alleles=["HLA-A*02:01"],
)
print("Smoke test output (head):")
print(_smoke_out.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Locate the PyFunc wrapper module
# MAGIC
# MAGIC `MHCFlurryImmunoBurdenModel` lives in `mhcflurry_wrapper.py` next to this
# MAGIC notebook. Code-based logging avoids cloudpickling a class defined in the
# MAGIC notebook's `__main__` scope.

# COMMAND ----------

_notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
WRAPPER_PATH = "/Workspace" + os.path.dirname(_notebook_path) + "/mhcflurry_wrapper.py"
print(f"Wrapper module: {WRAPPER_PATH}")
assert os.path.exists(WRAPPER_PATH), f"Wrapper file missing at {WRAPPER_PATH}"

# Constants duplicated from the wrapper for the MLflow log_model call below.
MODEL_CLASS_NAME = "MHCFlurryImmunoBurdenModel"
MODEL_PEPTIDE_LEN = 9
MODEL_STRONG_THRESHOLD = 0.5
MODEL_DEFAULT_PANEL = (
    "HLA-A*01:01,HLA-A*02:01,HLA-A*03:01,"
    "HLA-B*07:02,HLA-B*08:01,HLA-B*44:02"
)

# COMMAND ----------

conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "python=3.11",
        "pip",
        {
            "pip": [
                "mhcflurry==2.2.1",
                "torch==2.7.1",
                "numpy==1.26.4",
                "pandas==2.2.3",
                "scikit-learn==1.5.2",
                "biopython==1.84",
                "mlflow==2.22.0",
                "cloudpickle==2.0.0",
            ]
        },
    ],
    "name": "mhcflurry_env",
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
    "alleles": [MODEL_DEFAULT_PANEL],
})

from mlflow.models import infer_signature

example_output = pd.DataFrame({
    "sequence": ["MKVL..."],
    "predicted_immuno_burden": [0.005],
    "max_presentation_score": [0.4],
})
example_signature = infer_signature(example_input, example_output)

with mlflow.start_run(run_name=f"register-{model_name}") as run:
    mlflow.log_params({
        "model_class": MODEL_CLASS_NAME,
        "upstream_repo": "https://github.com/openvax/mhcflurry",
        "license": "Apache-2.0",
        "mhcflurry_version": "2.2.1",
        "predictor_class": "Class1PresentationPredictor",
        "default_panel": MODEL_DEFAULT_PANEL,
        "peptide_len": MODEL_PEPTIDE_LEN,
        "strong_threshold": MODEL_STRONG_THRESHOLD,
    })

    logged = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WRAPPER_PATH,
        artifacts={"mhcflurry_data": ARTIFACTS_DIR},
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
    model_name="MHCflurry Immunogenicity",
    model_display_name="MHCflurry 2.x MHC-I Immunogenic Burden",
    model_source_version="v2.2.1 (openvax mhcflurry, Apache-2.0)",
    model_description_url="https://github.com/openvax/mhcflurry",
)

deploy_run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="MHCflurry MHC-I Immunogenic Burden",
    deployment_description="MHCflurry 2.x peptide-MHC class I presentation predictor (Apache-2.0). Scans each input protein with a 9-mer sliding window across a default 6-allele HLA panel (~95% global coverage); returns per-residue strong-presenter density as predicted_immuno_burden.",
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
