# Databricks notebook source
# MAGIC %md
# MAGIC ### TEDDY — Load + Predict Smoke Test (notebook, not endpoint)
# MAGIC
# MAGIC Loads the registered TEDDY model via `mlflow.pyfunc.load_model` and runs predict
# MAGIC on a tiny synthetic input. If this works but serving doesn't, the issue is
# MAGIC infra-only (MLflow's code_paths-vs-cloudpickle ordering in the serving runtime)
# MAGIC and the batch-workflow path is viable.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("model_name", "teddy", "Model Name")
dbutils.widgets.text("model_version", "3", "Model Version to test")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

# MAGIC %pip install mlflow==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

model_uri = f"models:/{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}.{dbutils.widgets.get('model_name')}/{dbutils.widgets.get('model_version')}"
print(f"Loading {model_uri}")

# MLflow downloads the artifact, creates a conda env from the model's own
# requirements.txt (NOT our local one), adds code_paths to sys.path,
# then cloudpickle.loads the wrapper. Whatever happens here is the same
# path the serving runtime would hit at model-load time.
model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded: {type(model).__name__}")
print(f"Inner: {type(model._model_impl).__name__}")

# COMMAND ----------

# DBTITLE 1,Build synthetic input using real ENSG IDs from the bundled artifact
import os, json, glob
import numpy as np
import pandas as pd

# The model artifact bundles model_dir which includes vocab.txt. We re-resolve
# it from inside the loaded model so we don't have to hardcode the snapshot path.
artifacts_dir = model._model_impl.context.artifacts["model_dir"]
print(f"model_dir artifact: {artifacts_dir}")
vocab_path = os.path.join(artifacts_dir, "vocab.txt")
with open(vocab_path) as f:
    vocab_tokens = [line.strip() for line in f if line.strip()]
real_genes = [t for t in vocab_tokens if not t.startswith("<")][:100]
print(f"Loaded {len(real_genes)} ENSG ids from vocab.txt")

# COMMAND ----------

n_cells, n_genes = 5, 100
rng = np.random.default_rng(seed=42)
expr = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)

obs_df = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
var_df = pd.DataFrame({"index": real_genes})

input_df = pd.DataFrame({
    "adata_sparsematrix": [expr.tolist()],
    "adata_obs": [obs_df.to_json(orient="split")],
    "adata_var": [var_df.to_json(orient="split")],
})

# COMMAND ----------

# DBTITLE 1,Predict
print("Calling predict...")
result = model.predict(input_df, params={"max_seq_len": 256, "pooling": "mean"})
print(f"OK — got {len(result)} predictions")
print(f"First embedding length: {len(result[0]['embedding'])}")
print(f"First 8 dims: {result[0]['embedding'][:8]}")
