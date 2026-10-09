# Databricks notebook source
# MAGIC %md
# MAGIC ### HLAIIPred — MHC class II (CD4 / anti-drug-antibody) immunogenicity predictor
# MAGIC
# MAGIC Registers [HLAIIPred](https://github.com/pfizer-opensource/HLAIIPred) (Pfizer, **Apache-2.0**, PyTorch,
# MAGIC ~9 MB weights) into MLflow / Unity Catalog and deploys a **CPU** serving endpoint via Genesis Workbench.
# MAGIC
# MAGIC This is the MHC-II counterpart to the MHC-I MHCflurry endpoint — the *right* immunogenicity signal for
# MAGIC **antibodies** (ADA risk is MHC-II / CD4-driven). The code (`hlapred`) ships via `code_paths`; the model
# MAGIC weights (`models/epT_{0,1}.pt`, `config.yaml`) + `mhcII/` pseudosequences ship via `artifacts`, so serving
# MAGIC needs no git/PyPI access. The wrapper slides a 15-mer window over the input chain, scores every window ×
# MAGIC allele across the two released folds, and returns `predicted_immuno_burden` (strong presenters per residue;
# MAGIC lower = more de-immunized). Default panel = 8 common DRB1 alleles.
# MAGIC
# MAGIC > HLAIIPred outputs MHC-II **presentation** (a strong ADA *proxy/screen*), not calibrated clinical
# MAGIC > immunogenicity.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "hlaiipred_v1", "Model Name")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "hlaiipred", "Cache dir (UC volume)")
dbutils.widgets.text("workload_type", "CPU", "Workload Type for endpoints")
dbutils.widgets.text("hlaiipred_git_url", "https://github.com/pfizer-opensource/HLAIIPred.git", "HLAIIPred git URL (public)")
dbutils.widgets.text("hlaiipred_git_ref", "main", "HLAIIPred git ref (pin a commit for reproducibility)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies (exact pins) — HLAIIPred is small torch on CPU

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC     torch==2.7.1 \
# MAGIC     scipy==1.13.1 \
# MAGIC     numpy==1.26.4 \
# MAGIC     pandas==2.2.3 \
# MAGIC     tqdm==4.66.5 \
# MAGIC     pyyaml==6.0.2 \
# MAGIC     biopython==1.84 \
# MAGIC     mlflow==2.22.0 \
# MAGIC     cloudpickle==2.0.0 \
# MAGIC     databricks-sdk==0.50.0 \
# MAGIC     databricks-sql-connector==4.0.2

# COMMAND ----------

gwb_library_path = None
for lib in dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries"):
    if lib.name.startswith("genesis_workbench"):
        gwb_library_path = lib.path.replace("dbfs:", "")
print(f"Genesis Workbench library wheel: {gwb_library_path}")

# COMMAND ----------

# MAGIC %pip install {gwb_library_path} --force-reinstall
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, subprocess
import numpy as np
import pandas as pd
import mlflow

g = dbutils.widgets.get
catalog, schema, model_name = g("catalog"), g("schema"), g("model_name")
experiment_name, user_email, sql_warehouse_id = g("experiment_name"), g("user_email"), g("sql_warehouse_id")
cache_dir, workload_type = g("cache_dir"), g("workload_type")
git_url, git_ref = g("hlaiipred_git_url"), g("hlaiipred_git_ref")

from genesis_workbench.workbench import initialize
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
initialize(core_catalog_name=catalog, core_schema_name=schema, sql_warehouse_id=sql_warehouse_id, token=databricks_token)
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")

# COMMAND ----------

# DBTITLE 1,Clone HLAIIPred (public, Apache-2.0) — ships hlapred via code_paths + weights via artifacts
REPO_DIR = "/tmp/hlaiipred_repo"
if os.path.exists(REPO_DIR):
    subprocess.run(["rm", "-rf", REPO_DIR], check=True)
subprocess.run(["git", "clone", "--depth", "1", "--branch", git_ref, git_url, REPO_DIR], check=True)
resolved_sha = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "HEAD"], text=True).strip()
print(f"HLAIIPred @ {git_ref} ({resolved_sha})")

HLAPRED_PKG = os.path.join(REPO_DIR, "hlapred")      # code_paths: the importable package
MODELS_DIR = os.path.join(REPO_DIR, "models")        # artifacts: epT_0.pt / epT_1.pt / config.yaml
MHCII_DIR = os.path.join(REPO_DIR, "mhcII")          # artifacts: allele pseudosequences
for p in (HLAPRED_PKG, MODELS_DIR, MHCII_DIR):
    assert os.path.isdir(p), f"expected {p} in the HLAIIPred repo — layout changed? pin a known-good ref"
sys.path.insert(0, REPO_DIR)  # make hlapred importable for the smoke test below

# COMMAND ----------

# DBTITLE 1,Smoke-test the predictor before packaging (CPU, both folds)
import torch
from hlapred.predict import HLAIIPredict

_peptides = ["IKKWEKQVSQKKKQKN", "KANVKIFKSQGAA"]      # from the repo's example
_alleles = [["DRB1*01:01", "DRB1*04:01"] for _ in _peptides]
for _fold in (0, 1):
    _p = HLAIIPredict(MODELS_DIR, _fold, torch.device("cpu"), MHCII_DIR)
    _inp = _p.prepare_input(_peptides, _alleles)
    _yp, _sc = _p.predict(_inp, batch_size=32, sigmoid=True)
    print(f"fold {_fold} presentation scores:", np.asarray(_yp, dtype=float).reshape(-1))

# COMMAND ----------

# DBTITLE 1,Locate the PyFunc wrapper (code-based logging) + build the conda env
_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
WRAPPER_PATH = "/Workspace" + os.path.dirname(_notebook_path) + "/hlaiipred_wrapper.py"
assert os.path.exists(WRAPPER_PATH), f"Wrapper file missing at {WRAPPER_PATH}"
print(f"Wrapper module: {WRAPPER_PATH}")

# conda_env lists ONLY PyPI deps — hlapred itself ships via code_paths (not on PyPI), weights via artifacts.
conda_env = {
    "channels": ["defaults", "conda-forge"],
    "dependencies": [
        "python=3.11",
        "pip",
        {"pip": [
            "torch==2.7.1", "scipy==1.13.1", "numpy==1.26.4", "pandas==2.2.3",
            "tqdm==4.66.5", "pyyaml==6.0.2", "biopython==1.84",
            "mlflow==2.22.0", "cloudpickle==2.0.0",
        ]},
    ],
    "name": "hlaiipred_env",
}

# COMMAND ----------

# DBTITLE 1,Register the PyFunc into Unity Catalog
from genesis_workbench.models import set_mlflow_experiment
set_mlflow_experiment(experiment_tag=experiment_name, user_email=user_email, host=None, token=None, shared=True)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

uc_model_name = f"{catalog}.{schema}.{model_name}"
DEFAULT_PANEL = "DRB1*01:01,DRB1*03:01,DRB1*04:01,DRB1*07:01,DRB1*08:01,DRB1*11:01,DRB1*13:01,DRB1*15:01"

from mlflow.models import infer_signature
example_input = pd.DataFrame({
    "sequence": ["QVQLVESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAR... "],
    "alleles": [DEFAULT_PANEL],
})
example_output = pd.DataFrame({
    "sequence": ["QVQL..."], "predicted_immuno_burden": [0.02], "max_presentation_score": [0.6],
})
example_signature = infer_signature(example_input, example_output)

with mlflow.start_run(run_name=f"register-{model_name}") as run:
    mlflow.log_params({
        "model_class": "HLAIIPredImmunoBurdenModel",
        "upstream_repo": git_url, "upstream_ref": git_ref, "upstream_sha": resolved_sha,
        "license": "Apache-2.0", "predictor": "HLAIIPred (MHC-II presentation)",
        "default_panel": DEFAULT_PANEL, "peptide_len": 15, "strong_threshold": 0.5,
    })
    logged = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WRAPPER_PATH,
        code_paths=[HLAPRED_PKG],
        artifacts={"models": MODELS_DIR, "mhcII": MHCII_DIR},
        conda_env=conda_env,
        signature=example_signature,
        input_example=example_input,
        registered_model_name=uc_model_name,
    )
    print(f"Registered {uc_model_name}; run {run.info.run_id}; uri {logged.model_uri}")

# COMMAND ----------

# DBTITLE 1,Import into GWB + deploy the CPU serving endpoint (standard path auto-grants the app SP)
from genesis_workbench.models import (ModelCategory, import_model_from_uc, get_latest_model_version, deploy_model)
from genesis_workbench.workbench import wait_for_job_run_completion

model_version = get_latest_model_version(uc_model_name)
gwb_model_id = import_model_from_uc(
    user_email=user_email,
    model_category=ModelCategory.SMALL_MOLECULE,
    model_uc_name=uc_model_name,
    model_uc_version=model_version,
    model_name="HLAIIPred Immunogenicity",
    model_display_name="HLAIIPred MHC-II (CD4/ADA) Immunogenic Burden",
    model_source_version=f"{git_ref} ({resolved_sha[:8]}, pfizer-opensource, Apache-2.0)",
    model_description_url="https://github.com/pfizer-opensource/HLAIIPred",
)
deploy_run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="HLAIIPred MHC-II Immunogenic Burden",
    deployment_description="HLAIIPred (Pfizer, Apache-2.0) MHC class II presentation predictor. Slides a 15-mer window over the input protein across an 8-allele DRB1 panel, averages the two released folds, and returns per-residue strong-presenter density as predicted_immuno_burden (MHC-II / CD4 / anti-drug-antibody proxy; lower = more de-immunized).",
    input_adapter_str="none", output_adapter_str="none",
    sample_input_data_dict_as_json="none", sample_params_as_json="none",
    workload_type=workload_type, workload_size="Small",
)
print(f"Deploy run ID: {deploy_run_id}")
print(wait_for_job_run_completion(deploy_run_id, timeout=3600))
