# Databricks notebook source
# MAGIC %md
# MAGIC # BioNeMo ESM2 — Inference (containerless HF + Transformer Engine)
# MAGIC
# MAGIC Runs sequence-level regression / classification inference with an ESM-2 model, re-architected
# MAGIC off the NVIDIA BioNeMo container (`infer_esm2` CLI) onto a **serverless-GPU `notebook_task`**
# MAGIC (`GPU_1xA10`). Loads ESM-2 from Hugging Face on Transformer Engine (see the fine-tune notebook)
# MAGIC and either a fine-tuned head (from `bionemo_weights` → UC Volume) or a fresh head for a base
# MAGIC model, then writes a predictions CSV to the Volume. Batch-workflow Layer 4: attaches to the
# MAGIC dispatcher's MLflow run and advances `job_status`.

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

# NOTE: widget names are the app contract — genesis_workbench.bionemo.start_esm2_inference passes
# these exact job-parameter keys via jobs.run_now. Do not rename. (model_volume is an added
# default so the HF cache / weights Volume can be located; the app need not pass it.)
dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("model_volume", "bionemo", "Volume where weights are stored")
dbutils.widgets.text("is_base_model", "false", "Use Base Model?")
dbutils.widgets.text("esm_variant", "650M", "ESM Variant")
dbutils.widgets.text("task_type", "regression", "Task type: Regression or Classification")
dbutils.widgets.text("finetune_run_id", "3", "Finetune Run Id")
dbutils.widgets.text("data_location", "", "Inference data location")
dbutils.widgets.text("sequence_column_name", "sequence", "Column name containing the sequence")
dbutils.widgets.text("result_location", "", "Result Location in UC Volume")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
dbutils.widgets.text("experiment_name", "gwb_bionemo_esm2_inference", "MLflow experiment name")
dbutils.widgets.text("run_name", "esm2_inference", "MLflow run name")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (from the app dispatcher)")

# COMMAND ----------

# MAGIC %pip install -q "transformers==4.55.0" "accelerate>=0.34" "mlflow[databricks]==2.22.0" "databricks-sdk==0.50.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Read widgets AFTER restartPython (the restart clears everything defined before it).
import os
g = dbutils.widgets.get
catalog = g("core_catalog")
schema = g("core_schema")
sql_warehouse_id = g("sql_warehouse_id")
model_volume = g("model_volume")
is_base_model = g("is_base_model") == "true"
esm_variant = g("esm_variant")
task_type = g("task_type")
finetune_run_id = g("finetune_run_id")
data_location = g("data_location")
sequence_column_name = g("sequence_column_name")
result_location = g("result_location")
user_email = g("user_email")
experiment_name = g("experiment_name")
run_name = g("run_name") or "esm2_inference"
mlflow_run_id = g("mlflow_run_id") or None

_vol_root = f"/Volumes/{catalog}/{schema}/{model_volume}"
os.environ["HF_HOME"] = f"{_vol_root}/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Transformer Engine (prebuilt wheel from the libraries Volume; same as finetune)

# COMMAND ----------

import sys, glob, subprocess


def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


te_status = "stock-hf-no-te"
try:
    vol_wheels = glob.glob(f"/Volumes/{catalog}/{schema}/libraries/transformer_engine*.whl")
    if vol_wheels:
        _pip("--no-deps", *vol_wheels)  # install all TE wheels; never source-build inline
        te_status = "volume-wheel:" + ",".join(os.path.basename(w) for w in vol_wheels)
    else:
        print("[TE] no transformer_engine wheel in the libraries Volume "
              "(run core/build_transformer_engine) — running stock HF ESM-2 without TE")
except Exception as e:
    te_status = f"install-failed ({type(e).__name__}: {str(e)[:160]}); stock HF"

print("Transformer Engine:", te_status)

# COMMAND ----------

import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

_HF_IDS = {
    "650M": ("nvidia/esm2_t33_650M_UR50D", "facebook/esm2_t33_650M_UR50D"),
    "3B":   ("nvidia/esm2_t36_3B_UR50D",   "facebook/esm2_t36_3B_UR50D"),
}
_dtype = torch.bfloat16


def _local_or_hub(repo_id: str) -> str:
    # Prefer a pre-staged snapshot on the Volume (serverless can't reach the HF LFS CDN); full
    # org--name (HF cache convention) so facebook/ and nvidia/ don't collide.
    local = f"{_vol_root}/hf_models/{repo_id.replace('/', '--')}"
    return local if os.path.isdir(local) else repo_id


def load_te_esm2_encoder(variant: str):
    nvidia_id, facebook_id = _HF_IDS.get(variant, _HF_IDS["650M"])
    tokenizer = AutoTokenizer.from_pretrained(_local_or_hub(facebook_id))
    try:
        enc = AutoModel.from_pretrained(_local_or_hub(nvidia_id), trust_remote_code=True, torch_dtype=_dtype)
        return tokenizer, enc, enc.config.hidden_size, f"nvidia-te:{nvidia_id}"
    except Exception as e:
        print(f"[TE] nvidia remote-code failed ({type(e).__name__}: {str(e)[:160]}); "
              f"trying facebook + convert_esm_hf_to_te")
    enc = AutoModel.from_pretrained(_local_or_hub(facebook_id), torch_dtype=_dtype)
    try:
        from esm.convert import convert_esm_hf_to_te
        enc = convert_esm_hf_to_te(enc)
        return tokenizer, enc, enc.config.hidden_size, "convert_esm_hf_to_te"
    except Exception as e:
        print(f"[TE] convert unavailable ({type(e).__name__}: {str(e)[:160]}); stock HF ESM-2")
        return tokenizer, enc, enc.config.hidden_size, "stock-hf-no-te"


class ESM2SeqHead(nn.Module):
    """Must match the fine-tune notebook's architecture so a saved state_dict loads cleanly."""

    def __init__(self, encoder, hidden_size, mlp_hidden, mlp_target, dropout, task_type):
        super().__init__()
        self.encoder = encoder
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_target))

    @torch.no_grad()
    def predict_logits(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled.float())

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow run (status + search)

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

if mlflow_run_id:
    _run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    from databricks.sdk import WorkspaceClient
    _exp_folder = f"/Users/{user_email}/mlflow_experiments"
    WorkspaceClient().workspace.mkdirs(f"/Workspace{_exp_folder}")
    exp = mlflow.set_experiment(f"{_exp_folder}/{experiment_name}")
    MlflowClient().set_experiment_tag(exp.experiment_id, "used_by_genesis_workbench", "yes")
    _run_ctx = mlflow.start_run(run_name=run_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resolve weights, run inference, write results

# COMMAND ----------

work_dir = "/tmp/bionemo_infer"
shutil.rmtree(work_dir, ignore_errors=True)
ft_weights_dir = f"{work_dir}/ft_weights"
os.makedirs(work_dir, exist_ok=True)

active_run_id = None
try:
    with _run_ctx as run:
        active_run_id = run.info.run_id
        for k, v in {"origin": "genesis_workbench", "feature": "bionemo_esm_inference",
                     "created_by": user_email, "result_location": result_location,
                     "job_status": "running"}.items():
            mlflow.set_tag(k, v)
        mlflow.log_param("esm_variant", esm_variant)
        mlflow.log_param("is_base_model", str(is_base_model))

        tokenizer, encoder, hidden_size, te_how = load_te_esm2_encoder(esm_variant)
        mlflow.set_tag("te_backend", te_how)
        print(f"encoder hidden_size={hidden_size}  TE={te_how}")

        if is_base_model:
            # No fine-tuned head available — attach a fresh (untrained) head, matching the
            # container's base-model path (predictions come from an untrained head).
            saved_cfg = {"mlp_hidden_size": 256, "mlp_target_size": 1, "mlp_ft_dropout": 0.0,
                         "task_type": task_type, "label_classes": None}
            model = ESM2SeqHead(encoder, hidden_size, saved_cfg["mlp_hidden_size"],
                                saved_cfg["mlp_target_size"], saved_cfg["mlp_ft_dropout"], task_type)
        else:
            # Look up the fine-tuned weights dir from bionemo_weights (native spark.sql), copy it
            # local, and reconstruct the exact module the fine-tune notebook saved.
            row = spark.sql(f"SELECT weights_volume_location FROM {catalog}.{schema}.bionemo_weights "
                            f"WHERE ft_id = {int(finetune_run_id)}").collect()
            if not row:
                raise RuntimeError(f"finetune run {finetune_run_id} not found in bionemo_weights")
            weights_volume_location = row[0][0]
            shutil.rmtree(ft_weights_dir, ignore_errors=True)
            shutil.copytree(weights_volume_location, ft_weights_dir)
            ckpt = torch.load(f"{ft_weights_dir}/gwb_esm2_model.pt", map_location="cpu", weights_only=False)
            saved_cfg = ckpt["config"]
            tokenizer = AutoTokenizer.from_pretrained(ft_weights_dir)
            model = ESM2SeqHead(encoder, hidden_size, saved_cfg["mlp_hidden_size"],
                                saved_cfg["mlp_target_size"], saved_cfg["mlp_ft_dropout"],
                                saved_cfg["task_type"])
            model.load_state_dict(ckpt["state_dict"])
            task_type = saved_cfg["task_type"]
            print(f"loaded fine-tuned weights from {weights_volume_location}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        # --- data ---
        infer_pdf = pd.read_csv(data_location)
        seqs = infer_pdf[sequence_column_name].astype(str).tolist()

        # --- batched inference ---
        preds = []
        bs = 8
        for i in range(0, len(seqs), bs):
            enc = tokenizer(seqs[i:i + bs], return_tensors="pt", padding=True,
                            truncation=True, max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model.predict_logits(enc["input_ids"], enc["attention_mask"]).float().cpu()
            if task_type == "classification":
                idx = logits.argmax(-1).tolist()
                classes = saved_cfg.get("label_classes")
                preds.extend([classes[j] for j in idx] if classes else idx)
            else:
                preds.extend(logits.squeeze(-1).tolist())

        results_df = infer_pdf.copy()
        results_df["predictions"] = preds

        # --- write results CSV to the Volume ---
        os.makedirs(result_location, exist_ok=True)
        results_file = f"{result_location}/results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"results -> {results_file} ({len(results_df)} rows)")

        mlflow.log_metric("num_sequences", int(len(results_df)))
        mlflow.set_tag("results_file", results_file)
        mlflow.set_tag("job_status", "complete")
except Exception as exc:
    if active_run_id:
        try:
            MlflowClient().set_tag(active_run_id, "job_status", "failed")
            MlflowClient().set_tag(active_run_id, "failure_reason", str(exc)[:500])
        except Exception:
            pass
    raise

# COMMAND ----------

print("DONE")
