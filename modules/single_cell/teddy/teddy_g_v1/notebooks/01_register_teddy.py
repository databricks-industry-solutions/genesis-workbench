# Databricks notebook source
# MAGIC %md
# MAGIC ### TEDDY (Merck) — Register encoder for cell embeddings
# MAGIC
# MAGIC The publicly released TEDDY-G checkpoints (70M / 160M / 400M) are
# MAGIC encoder-only foundation models — `n_cls=0`, no classification heads. This
# MAGIC notebook wraps the encoder in an MLflow PyFunc that returns a per-cell
# MAGIC embedding (mean-pooled `last_hidden_state`). The annotation workflow uses
# MAGIC the embedding to query a separately-built Vector Search index of labeled
# MAGIC reference cells and majority-votes the labels.
# MAGIC
# MAGIC Source: https://huggingface.co/Merck/TEDDY (Apache 2.0)

# COMMAND ----------

# DBTITLE 1,[gwb] pip install pinned requirements
# MAGIC %cat ../requirements.txt
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_yyang_genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "teddy", "Model Name")
dbutils.widgets.text("experiment_name", "teddy_genesis_workbench_modules", "Experiment Name")
dbutils.widgets.text("sql_warehouse_id", "w123", "SQL Warehouse Id")
dbutils.widgets.text("user_email", "user@databricks.com", "User Id/Email")
dbutils.widgets.text("cache_dir", "teddy", "Cache dir")
dbutils.widgets.text("teddy_hf_repo", "Merck/TEDDY", "HF repo")
dbutils.widgets.text("teddy_hf_revision", "main", "HF revision (commit/tag)")
dbutils.widgets.text("teddy_model_size", "70M", "TEDDY-G variant (70M / 160M / 400M)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
user_email = dbutils.widgets.get("user_email")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
cache_dir = dbutils.widgets.get("cache_dir")
hf_repo = dbutils.widgets.get("teddy_hf_repo")
hf_revision = dbutils.widgets.get("teddy_hf_revision")
model_size = dbutils.widgets.get("teddy_model_size")

cache_full_path = f"/Volumes/{catalog}/{schema}/{cache_dir}"
print(f"Cache full path: {cache_full_path}")
print(f"HF repo: {hf_repo} @ {hf_revision}, variant: TEDDY-G {model_size}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cache_dir}")
print(f"Volume {cache_full_path} ready")

# COMMAND ----------

# DBTITLE 1,Download TEDDY snapshot (weights + source code) from HuggingFace
import os
from huggingface_hub import snapshot_download

snapshot_dir = f"{cache_full_path}/snapshots/{hf_revision}"
os.makedirs(snapshot_dir, exist_ok=True)

sentinel = os.path.join(snapshot_dir, ".snapshot_complete")
if os.path.exists(sentinel):
    print(f"Snapshot already present at {snapshot_dir}, skipping download")
else:
    print(f"Downloading {hf_repo}@{hf_revision} -> {snapshot_dir}")
    snapshot_download(
        repo_id=hf_repo,
        revision=hf_revision,
        local_dir=snapshot_dir,
        local_dir_use_symlinks=False,
    )
    with open(sentinel, "w") as f:
        f.write("ok")

import sys
if snapshot_dir not in sys.path:
    sys.path.insert(0, snapshot_dir)

teddy_pkg_dir = os.path.join(snapshot_dir, "teddy")
assert os.path.isdir(teddy_pkg_dir), (
    f"Expected `teddy/` package at {teddy_pkg_dir} — HF repo layout may have changed. "
    f"Inspect snapshot: {os.listdir(snapshot_dir)}"
)
print(f"teddy package: {teddy_pkg_dir}")

# COMMAND ----------

# DBTITLE 1,Resolve the variant checkpoint path
model_dir = os.path.join(teddy_pkg_dir, "models", "teddy_g", model_size)
assert os.path.isdir(model_dir), (
    f"Expected TEDDY-G {model_size} checkpoint at {model_dir}. "
    f"Available under teddy_g: {os.listdir(os.path.join(teddy_pkg_dir, 'models', 'teddy_g'))}"
)
print(f"Checkpoint dir: {model_dir}")

# COMMAND ----------

# DBTITLE 1,Stage code_paths content + import the wrapper class
# Top-level imports needed for the cells below (the inline class previously
# carried these; now they live in teddy_wrapper.py instead).
import json
import numpy as np
import pandas as pd
import mlflow.pyfunc

# Build a clean dir that becomes our MLflow code_paths bundle:
#   /tmp/teddy_code/teddy/         — package source (no weights, no LFS)
#   /tmp/teddy_code/teddy_wrapper.py — the PyFunc class file (loaded by python_model=path)
# This is the trimmed artifact (~30 MB) the serving image will receive.
import shutil
clean_code_dir = "/tmp/teddy_code"
if os.path.exists(clean_code_dir):
    shutil.rmtree(clean_code_dir)
shutil.copytree(
    teddy_pkg_dir,
    f"{clean_code_dir}/teddy",
    ignore=shutil.ignore_patterns(
        "*.safetensors", "*.bin", "*.ckpt", "*.pt",
        "__pycache__", ".DS_Store",
    ),
)

# teddy_wrapper.py is committed alongside this notebook in the bundle; copy it
# next to teddy/ so MLflow's code_paths bundle ships both.
_nb_dir = "/Workspace" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 1)[0]
wrapper_src = f"{_nb_dir}/teddy_wrapper.py"
wrapper_dest = f"{clean_code_dir}/teddy_wrapper.py"
shutil.copy2(wrapper_src, wrapper_dest)
print(f"Staged code_paths bundle at {clean_code_dir}")

# Import the class IN THIS NOTEBOOK so we can run the dry-load test below.
# (At serving / mlflow.pyfunc.load_model time, MLflow imports the wrapper file
# via importlib.util.spec_from_file_location — see python_model=wrapper_dest in
# the log_model call further down.)
if clean_code_dir not in sys.path:
    sys.path.insert(0, clean_code_dir)
from teddy_wrapper import TEDDYEmbedder

# COMMAND ----------

# DBTITLE 1,Synthetic input_example for signature inference
import scanpy as sc

n_cells, n_genes = 5, 100
rng = np.random.default_rng(seed=42)
expr = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)

# Use real ENSG IDs from vocab.txt so the tokenizer can encode them (synthetic
# placeholder names get rejected by GeneTokenizer rather than falling back to <unk>).
vocab_path = os.path.join(model_dir, "vocab.txt")
with open(vocab_path) as f:
    vocab_tokens = [line.strip() for line in f if line.strip()]
# Skip any special tokens at the head of the file (they start with '<'), then take the first n_genes regular genes.
real_genes = [t for t in vocab_tokens if not t.startswith("<")][:n_genes]
assert len(real_genes) == n_genes, (
    f"vocab.txt has only {len(real_genes)} regular tokens — expected at least {n_genes}"
)
gene_names = real_genes

obs_df = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
var_df = pd.DataFrame({"index": gene_names})

input_data = pd.DataFrame({
    "adata_sparsematrix": [expr.tolist()],
    "adata_obs": [obs_df.to_json(orient="split")],
    "adata_var": [var_df.to_json(orient="split")],
})

default_params = {
    "max_seq_len": 2048,
    "pooling": "mean",
}

# COMMAND ----------

# DBTITLE 1,Dry-load test before MLflow logging
from mlflow.pyfunc import PythonModelContext

# Two artifacts:
#  - model_dir: the 70M checkpoint dir (~280 MB) — weights + tokenizer files
#  - teddy_pkg_parent: the parent dir that contains the `teddy/` package source.
#    Bundling teddy/ as an artifact (not code_paths) means it lands at a
#    well-known location accessible via context.artifacts at load_context time,
#    which sidesteps MLflow's code_paths-not-on-sys.path bug for the code-model
#    loader.
artifacts = {
    "model_dir": model_dir,
    "teddy_pkg_parent": clean_code_dir,  # contains teddy/ (weights stripped)
}
ctx = PythonModelContext(artifacts=artifacts, model_config={})

model = TEDDYEmbedder(model_size=model_size)
model.load_context(ctx)

print("=== config (key fields) ===")
for k in ("d_model", "nlayers", "nheads", "ntoken", "max_position_embeddings",
          "add_cls", "cls_token_id", "pad_token", "mask_token"):
    print(f"  {k}: {getattr(model.config, k, None)}")

print("=== tokenizer methods (first 20) ===")
print(f"  type: {type(model.tokenizer).__name__}")
print(f"  methods: {[m for m in dir(model.tokenizer) if not m.startswith('_') and callable(getattr(model.tokenizer, m, None))][:20]}")

output_example = model.predict(ctx, input_data, default_params)
print(f"Output: {len(output_example)} embeddings, dim={len(output_example[0]['embedding'])}")

# COMMAND ----------

# DBTITLE 1,Infer signature
from mlflow.models import infer_signature

signature = infer_signature(
    model_input=input_data,
    model_output=output_example,
    params=default_params,
)
signature

# COMMAND ----------

# DBTITLE 1,Log + register the model in UC
import mlflow
from databricks.sdk import WorkspaceClient

def set_mlflow_experiment(experiment_tag):
    w = WorkspaceClient()
    base = "Shared/dbx_genesis_workbench_models"
    w.workspace.mkdirs(f"/Workspace/{base}")
    path = f"/{base}/{experiment_tag}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    return mlflow.set_experiment(path)

experiment = set_mlflow_experiment(experiment_name)
registered_model_name = f"{catalog}.{schema}.{model_name}"

with mlflow.start_run(run_name=f"{model_name}_{model_size}_embedder", experiment_id=experiment.experiment_id) as run:
    mlflow.pyfunc.log_model(
        artifact_path="teddy",
        # File-path API (MLflow 2.20+): MLflow imports this file at load time via
        # importlib.util.spec_from_file_location — NO cloudpickle of the class.
        python_model=wrapper_dest,
        # teddy/ ships inside artifacts["teddy_pkg_parent"], not via code_paths,
        # because the code-model loader doesn't add code_paths to sys.path before
        # calling load_context. Artifacts give us a stable path we can sys.path
        # ourselves at load time.
        artifacts=artifacts,
        pip_requirements="../requirements.txt",
        signature=signature,
        input_example=(input_data, default_params),
        registered_model_name=registered_model_name,
    )

print(f"Registered {registered_model_name}")
