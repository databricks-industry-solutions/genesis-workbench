# Databricks notebook source
# MAGIC %md
# MAGIC # Register <MODEL NAME> — <one-line purpose>
# MAGIC
# MAGIC **What this notebook does.** Download the <MODEL> weights, wrap the model in an MLflow PyFunc,
# MAGIC and register it in Unity Catalog so Genesis Workbench can serve it. Input `<…>` → output `<…>`.
# MAGIC
# MAGIC **Compute.** <serverless GPU GPU_1xA10 | classic T4 | CPU — and why: VRAM, CUDA, a `%sh` step, etc.>
# MAGIC
# MAGIC **Pipeline.** install deps → parameters → download weights → define PyFunc → dry-load test →
# MAGIC signature + input_example → log & register in UC → (hand off to `02_import_model_gwb.py` for deploy).
# MAGIC
# MAGIC > **Status:** <confirmed end-to-end on … | first-draft — refine the … step>. Keep this honest so a
# MAGIC > reader knows what has actually been verified.
# MAGIC
# MAGIC > **How to read this notebook:** every step is one cell with a short markdown header above it saying
# MAGIC > *what* it does and *why*. Code comments are kept to the surprising bits only. This is the required
# MAGIC > shape for every GWB registration notebook — see the Notebook authoring standard in
# MAGIC > `claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Install dependencies
# MAGIC Every pip dep is **exactly pinned** (`pkg==X.Y.Z`) — unpinned installs have silently broken deploys
# MAGIC (see the dependency-hygiene hard rule). The serverless-GPU AI runtime already ships `<torch x.y / cuda>`
# MAGIC and `mlflow`, so we do **not** reinstall torch; we add only `<the model's own package + extras>`.
# MAGIC `mlflow`/`databricks-sdk` are (re)installed **last** so nothing downgrades them below what registration
# MAGIC needs.

# COMMAND ----------

# MAGIC %pip install -q <model_pkg>==<X.Y.Z> <extra>==<X.Y.Z>
# MAGIC %pip install -q "mlflow==<X.Y.Z>" "databricks-sdk==<X.Y.Z>"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Parameters
# MAGIC All inputs are notebook widgets so the registration job (and a human re-run) can override them. The
# MAGIC catalog/schema/cache_dir triple resolves the UC Volume the weights are cached in.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "<model_name>", "Model Name")
dbutils.widgets.text("cache_dir", "<model_name>", "Cache dir (UC Volume)")
dbutils.widgets.text("experiment_name", "dbx_genesis_workbench_modules", "MLflow experiment")
dbutils.widgets.text("sql_warehouse_id", "", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "user@databricks.com", "User Id/Email")
# TODO: add model-specific widgets (HF repo/revision, checkpoint filenames, token scope/key, workload_type).

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
cache_dir = dbutils.widgets.get("cache_dir")
experiment_name = dbutils.widgets.get("experiment_name")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
user_email = dbutils.widgets.get("user_email")

volume_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"catalog={catalog} schema={schema} model={model_name} volume_root={volume_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Download the model weights (skip if already cached)
# MAGIC Weights are cached in the UC Volume so a re-deploy doesn't re-download multi-GB files. The
# MAGIC `os.path.exists(...) and getsize > 0` guard makes this cell idempotent.

# COMMAND ----------

import os

weights_dir = f"{volume_root}/models"
os.makedirs(weights_dir, exist_ok=True)
weights_file = os.path.join(weights_dir, "<model.ckpt>")

if os.path.exists(weights_file) and os.path.getsize(weights_file) > 0:
    print(f"Weights already present at {weights_file} — skipping download")
else:
    print(f"Downloading weights → {weights_file}")
    # TODO: download from HuggingFace / NGC / Zenodo. Prefer a fast LOCAL temp for very large files
    #       and copy into the Volume, rather than writing multi-GB through the /Volumes FUSE mount.
    ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Define the MLflow PyFunc wrapper
# MAGIC `load_context` loads the weights **once** when the serving container starts (never per request);
# MAGIC `predict` turns each request row into model input and returns the serving contract below. Keeping the
# MAGIC class small and the heavy lifting in named helpers is what makes this readable at the endpoint later.
# MAGIC
# MAGIC **Serving contract** — input columns `[<…>]` → output columns `[<…>]`.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext


class <ModelName>Model(PythonModel):
    """<One-paragraph description: what the model does and how predict maps request → output.>"""

    def load_context(self, context):
        # Load weights ONCE here, not in predict (avoids re-reading GB per request).
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ...  # load from context.artifacts["weights"]
        self.model.float()  # force float32 — avoids float16/float32 dtype mismatches at serving
        self.model.to(self.device).eval()

    def predict(self, context, model_input, params=None):
        import pandas as pd

        results = []
        for _, row in model_input.iterrows():
            # TODO: map the request row to model input, run the model, collect the output row.
            results.append({...})
        return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Dry-load test
# MAGIC Load the wrapper and run one tiny prediction **before** logging — catches load/predict bugs here
# MAGIC (seconds) instead of after a long UC registration + endpoint build.

# COMMAND ----------

import pandas as pd

artifacts = {"weights": weights_file}
_model = <ModelName>Model()
_model.load_context(PythonModelContext(artifacts=artifacts, model_config={}))

example_input = pd.DataFrame([{...}])  # one small, representative row
print(_model.predict(None, example_input).head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Signature + input example
# MAGIC Keep the input example **small** (one short sequence / a handful of cells) — a large example slows
# MAGIC model logging and bloats the artifact. Use `long`/`double` column types so a pandas int/float example
# MAGIC survives MLflow schema enforcement.

# COMMAND ----------

from mlflow.models.signature import infer_signature

signature = infer_signature(example_input, _model.predict(None, example_input))
signature

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Log + register in Unity Catalog
# MAGIC Standard path: log the PyFunc and register it under `{catalog}.{schema}.{model_name}`. `02_import_model_gwb.py`
# MAGIC then imports it into the GWB registry and deploys the endpoint.
# MAGIC
# MAGIC *(For a torch > 2.7.1 / CUDA-13 model that standard GPU serving can't rebuild, use the Express variant in
# MAGIC §7-Express instead of this cell — not both. See the Express section of the development skill.)*

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user_email}/{experiment_name}")
uc_model_name = f"{catalog}.{schema}.{model_name}"

with mlflow.start_run(run_name=f"{model_name}_register"):
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model=<ModelName>Model(),
        artifacts=artifacts,
        signature=signature,
        input_example=example_input,
        pip_requirements="../requirements.txt",  # exact pins; must match the install cell
        registered_model_name=uc_model_name,
    )
print(f"registered {uc_model_name}: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7-Express · (Alternative) Express registration for torch > 2.7.1 / CUDA-13 models
# MAGIC Use this **instead of** §7 when the model's stack is newer than standard GPU serving's torch-2.7.1 /
# MAGIC CUDA-11.8 base (it force-reinstalls torch and cannot run a CUDA-13 stack). Express packages **this**
# MAGIC serverless-GPU env at registration and restores it verbatim at serving. Must run FROM a serverless-GPU
# MAGIC kernel, with `mlflow>=3.12` + `databricks-sdk>=0.150`.
# MAGIC
# MAGIC `install_dependencies=False` snapshots the installed site-packages verbatim (needed when the env has
# MAGIC git/binary/local-tagged wheels that no index can re-resolve). Do **not** set
# MAGIC `MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC="false"` — on MLflow 3.12+ the model logs under a
# MAGIC LoggedModel and the presigned repo can't vend credentials for it; leave the default (SDK repo).

# COMMAND ----------

# import torch, mlflow
# from mlflow.utils.env_pack import EnvPackConfig
# assert torch.cuda.is_available(), "Express must register from serverless GPU"
# mlflow.set_registry_uri("databricks-uc")
# with mlflow.start_run(run_name=f"{model_name}_register"):
#     model_info = mlflow.pyfunc.log_model(                       # log WITHOUT registered_model_name
#         name=model_name, python_model=<ModelName>Model(), artifacts=artifacts,
#         code_paths=code_paths, signature=signature, input_example=example_input,
#         pip_requirements=["mlflow", "cloudpickle", "pandas", "numpy"])  # env_pack defines the serving env
# model_version = mlflow.register_model(                          # register SEPARATELY with env_pack
#     model_uri=model_info.model_uri, name=uc_model_name,
#     env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False)).version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Summary
# MAGIC Print what was registered and the next step so the job log reads as a clear record.

# COMMAND ----------

print(f"✅ Registered {uc_model_name}")
print("Next: run 02_import_model_gwb.py to import into the GWB registry and deploy the serving endpoint.")
