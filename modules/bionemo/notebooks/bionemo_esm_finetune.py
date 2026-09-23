# Databricks notebook source
# MAGIC %md
# MAGIC # BioNeMo ESM2 — Fine-tune (containerless HF + Transformer Engine)
# MAGIC
# MAGIC Sequence-level regression / classification fine-tuning of ESM-2, re-architected off the
# MAGIC NVIDIA BioNeMo **container** (`finetune_esm2` CLI) onto a **serverless-GPU `notebook_task`**
# MAGIC (`hardware_accelerator: GPU_1xA10`) — the classic A10 GPU cluster can't provision at the
# MAGIC workshop's EC2 GPU quota=0, and the AI-Runtime custom-container pool is capacity-intermittent.
# MAGIC
# MAGIC Approach (per NVIDIA's ESM-2 recipe, https://docs.nvidia.com/bionemo-framework/latest/main/recipes/models/esm2/esm2/):
# MAGIC read ESM-2 **from Hugging Face** and run it through **Transformer Engine (TE)**:
# MAGIC   * primary — NVIDIA's TE checkpoint `nvidia/esm2_*` (`trust_remote_code=True`), whose remote
# MAGIC     code is already built on TE layers;
# MAGIC   * fallback — stock `facebook/esm2_*` converted with `esm.convert.convert_esm_hf_to_te`;
# MAGIC   * last resort — stock HF ESM-2 (no TE) so the job still completes on the working GPU pool.
# MAGIC
# MAGIC A configurable MLP head (the module's existing `mlp_*` params) sits on the mean-pooled encoder
# MAGIC output; the encoder is frozen (mirrors the container's `--encoder-frozen`) and only the head
# MAGIC trains, via HF `Trainer`. Batch-workflow Layer 4: attaches to the dispatcher's MLflow run,
# MAGIC advances `job_status`, writes the HF weights dir to the UC Volume, and inserts a
# MAGIC `bionemo_weights` row (native `spark.sql`). FP8 needs Hopper+, so A10 runs bf16 (no fp8).
# MAGIC
# MAGIC ⚠️ HF egress: serverless can't reach the HF LFS CDN. The ESM-2 weights load from a UC-Volume
# MAGIC HF cache (`{model_volume}/hf_cache`); if a run can't download, pre-stage the snapshot to
# MAGIC `{model_volume}/hf_models/<repo-name>` (a `from_pretrained`-loadable dir) — the loader uses it.

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------

# NOTE: widget names are the app contract — genesis_workbench.bionemo.start_esm2_finetuning
# passes these exact job-parameter keys via jobs.run_now. Do not rename.
dbutils.widgets.text("esm_variant", "650M", "ESM Variant")
dbutils.widgets.text("train_data_location", "", "Training data location")
dbutils.widgets.text("validation_data_location", "", "Validation data location")
dbutils.widgets.text("should_use_lora", "false", "Should use LORA")
dbutils.widgets.text("finetune_label", "esm_650m_ft_xyz", "A label using which these finetune weights are saved")
dbutils.widgets.text("core_catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("core_schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("sql_warehouse_id", "8f210e00850a2c16", "SQL Warehouse Id")
dbutils.widgets.text("model_volume", "bionemo", "Volume where weights are stored")

dbutils.widgets.text("task_type", "regression", "Task type")
dbutils.widgets.text("mlp_ft_dropout", "0.25", "Dropout")
dbutils.widgets.text("mlp_hidden_size", "256", "Hidden size")
dbutils.widgets.text("mlp_target_size", "1", "Target size")
dbutils.widgets.text("experiment_name", "sequence_level_regression", "Experiment name")
dbutils.widgets.text("num_steps", "50", "Num steps")
dbutils.widgets.text("lr", "5e-3", "Learning rate")
dbutils.widgets.text("lr_multiplier", "1e2", "Learning rate multiplier")
dbutils.widgets.text("micro_batch_size", "2", "Micro batch size")
dbutils.widgets.text("precision", "bf16-mixed", "Precision")
dbutils.widgets.text("user_email", "a@b.com", "User Email")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (from the app dispatcher)")

# COMMAND ----------

# Containerless deps. transformers/accelerate drive the HF Trainer; mlflow[databricks] for
# tracking. Transformer Engine is installed separately below (it may need a source build).
# MAGIC %pip install -q "transformers==4.55.0" "accelerate>=0.34" "mlflow[databricks]==2.22.0" "databricks-sdk==0.50.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Read widgets AFTER restartPython (the restart clears everything defined before it).
g = dbutils.widgets.get
catalog = g("core_catalog")
schema = g("core_schema")
esm_variant = g("esm_variant")
train_data_location = g("train_data_location")
validation_data_location = g("validation_data_location")
should_use_lora = g("should_use_lora") == "true"
finetune_label = g("finetune_label")
task_type = g("task_type")
mlp_ft_dropout = float(g("mlp_ft_dropout"))
mlp_hidden_size = int(g("mlp_hidden_size"))
mlp_target_size = int(g("mlp_target_size"))
experiment_name = g("experiment_name")
num_steps = int(g("num_steps"))
lr = float(g("lr"))
lr_multiplier = float(g("lr_multiplier"))
micro_batch_size = int(g("micro_batch_size"))
precision = g("precision")
user_email = g("user_email")
mlflow_run_id = g("mlflow_run_id") or None
model_volume = g("model_volume")
sql_warehouse_id = g("sql_warehouse_id")

# UC-Volume HF cache so downloads persist / can be pre-staged (serverless HF-egress workaround).
import os
_vol_root = f"/Volumes/{catalog}/{schema}/{model_volume}"
os.environ["HF_HOME"] = f"{_vol_root}/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Transformer Engine (prebuilt wheel from the libraries Volume)
# MAGIC TE is built ONCE on serverless GPU by `core/build_transformer_engine` (during the core
# MAGIC deploy) and cached to the `libraries` Volume — this notebook only INSTALLS that wheel, and
# MAGIC NEVER source-builds inline (a from-source build compiles CUDA kernels for ~40 min per run).
# MAGIC If the wheel isn't present yet, run stock HF ESM-2 (correct, just no TE acceleration).

# COMMAND ----------

import sys, glob, subprocess


def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


te_status = "stock-hf-no-te"
try:
    vol_wheels = glob.glob(f"/Volumes/{catalog}/{schema}/libraries/transformer_engine*.whl")
    if vol_wheels:
        _pip("--no-deps", *vol_wheels)  # install all TE wheels (transformer_engine + _torch)
        te_status = "volume-wheel:" + ",".join(os.path.basename(w) for w in vol_wheels)
    else:
        print("[TE] no transformer_engine wheel in the libraries Volume "
              "(run core/build_transformer_engine) — running stock HF ESM-2 without TE")
except Exception as e:
    te_status = f"install-failed ({type(e).__name__}: {str(e)[:160]}); stock HF"

print("Transformer Engine:", te_status)

# COMMAND ----------

import time, shutil, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments)

# ESM-2 variant -> Hugging Face repo ids. NVIDIA's `nvidia/esm2_*` ship TE remote code;
# `facebook/esm2_*` are the stock reference weights (same tokenizer/vocab as NVIDIA's).
_HF_IDS = {
    "650M": ("nvidia/esm2_t33_650M_UR50D", "facebook/esm2_t33_650M_UR50D"),
    "3B":   ("nvidia/esm2_t36_3B_UR50D",   "facebook/esm2_t36_3B_UR50D"),
}
use_bf16 = precision.startswith("bf16")
_dtype = torch.bfloat16 if use_bf16 else torch.float32


def _local_or_hub(repo_id: str) -> str:
    """Prefer a pre-staged snapshot on the Volume (serverless HF-egress workaround), else the hub id."""
    local = f"{_vol_root}/hf_models/{repo_id.split('/')[-1]}"
    return local if os.path.isdir(local) else repo_id


def load_te_esm2_encoder(variant: str):
    """Read ESM-2 from HF and get it running on Transformer Engine. Returns (tokenizer, encoder,
    hidden_size, how). `how` records which TE path engaged (for MLflow provenance)."""
    nvidia_id, facebook_id = _HF_IDS.get(variant, _HF_IDS["650M"])
    tokenizer = AutoTokenizer.from_pretrained(_local_or_hub(facebook_id))
    # 1) NVIDIA TE checkpoint — remote code is built on Transformer Engine layers.
    try:
        enc = AutoModel.from_pretrained(_local_or_hub(nvidia_id), trust_remote_code=True, torch_dtype=_dtype)
        return tokenizer, enc, enc.config.hidden_size, f"nvidia-te:{nvidia_id}"
    except Exception as e:
        print(f"[TE] nvidia remote-code path failed ({type(e).__name__}: {str(e)[:160]}); "
              f"falling back to facebook + convert_esm_hf_to_te")
    # 2) stock facebook weights, converted to TE with the BioNeMo recipe utility.
    enc = AutoModel.from_pretrained(_local_or_hub(facebook_id), torch_dtype=_dtype)
    try:
        from esm.convert import convert_esm_hf_to_te
        enc = convert_esm_hf_to_te(enc)
        return tokenizer, enc, enc.config.hidden_size, "convert_esm_hf_to_te"
    except Exception as e:
        print(f"[TE] convert_esm_hf_to_te unavailable ({type(e).__name__}: {str(e)[:160]}); "
              f"running stock HF ESM-2 (no TE)")
        return tokenizer, enc, enc.config.hidden_size, "stock-hf-no-te"


class ESM2SeqHead(nn.Module):
    """Frozen ESM-2 encoder + mean-pool + configurable MLP head (mirrors the container's
    ESM2FineTuneSeqConfig: mlp_hidden_size / mlp_target_size / mlp_ft_dropout, encoder-frozen)."""

    def __init__(self, encoder, hidden_size, mlp_hidden, mlp_target, dropout, task_type):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():  # --encoder-frozen: only the head trains
            p.requires_grad = False
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_target))

    def _pool(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = self.head(self._pool(input_ids, attention_mask).float())
        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss = F.mse_loss(logits.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(logits, labels.long())
        return {"loss": loss, "logits": logits}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data (UC Volume CSVs use `sequence` / `target`)

# COMMAND ----------

work_dir = "/tmp/bionemo_ft"
shutil.rmtree(work_dir, ignore_errors=True)
os.makedirs(work_dir, exist_ok=True)
ft_weights_dir = f"{work_dir}/ft_weights"
os.makedirs(ft_weights_dir, exist_ok=True)

ft_weights_volume_location = f"{_vol_root}/esm2/{esm_variant}/{finetune_label}"

train_pdf = pd.read_csv(train_data_location)[["sequence", "target"]].dropna()
val_pdf = pd.read_csv(validation_data_location)[["sequence", "target"]].dropna()
print(f"train={len(train_pdf)}  val={len(val_pdf)}  task={task_type}")

# For classification, map string/int targets to contiguous class ids; head width = #classes
# (honor mlp_target_size when the caller sized it, else infer from the data).
label_classes = None
if task_type == "classification":
    label_classes = sorted(pd.concat([train_pdf["target"], val_pdf["target"]]).unique().tolist())
    cls_index = {c: i for i, c in enumerate(label_classes)}
    train_pdf = train_pdf.assign(target=train_pdf["target"].map(cls_index))
    val_pdf = val_pdf.assign(target=val_pdf["target"].map(cls_index))
    num_target = max(mlp_target_size, len(label_classes))
else:
    num_target = mlp_target_size

# COMMAND ----------

tokenizer, encoder, hidden_size, te_how = load_te_esm2_encoder(esm_variant)
print(f"encoder hidden_size={hidden_size}  TE={te_how}")

model = ESM2SeqHead(encoder, hidden_size, mlp_hidden_size, num_target, mlp_ft_dropout, task_type)

_label_dtype = torch.long if task_type == "classification" else torch.float


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, pdf):
        self.enc = tokenizer(pdf["sequence"].tolist(), truncation=True, max_length=1024)
        self.labels = pdf["target"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def collate(features):
    batch = tokenizer.pad({"input_ids": [f["input_ids"] for f in features],
                           "attention_mask": [f["attention_mask"] for f in features]},
                          return_tensors="pt")
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=_label_dtype)
    return batch


train_ds, val_ds = SeqDataset(train_pdf), SeqDataset(val_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow run (status + search)

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

is_finetune_success = False
active_run_id = None

# Attach to the dispatcher's pre-created run, else create one (direct Jobs-UI fallback).
if mlflow_run_id:
    _run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    from databricks.sdk import WorkspaceClient
    _exp_folder = f"/Users/{user_email}/mlflow_experiments"
    WorkspaceClient().workspace.mkdirs(f"/Workspace{_exp_folder}")
    exp = mlflow.set_experiment(f"{_exp_folder}/{experiment_name}")
    MlflowClient().set_experiment_tag(exp.experiment_id, "used_by_genesis_workbench", "yes")
    _run_ctx = mlflow.start_run(run_name=finetune_label)

# COMMAND ----------

try:
    with _run_ctx as run:
        active_run_id = run.info.run_id
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        for k, v in {"origin": "genesis_workbench", "feature": "bionemo_esm_finetune",
                     "created_by": user_email, "result_location": ft_weights_volume_location,
                     "job_status": "training", "te_backend": te_how}.items():
            mlflow.set_tag(k, v)
        mlflow.log_params({
            "esm_variant": esm_variant, "finetune_label": finetune_label, "task_type": task_type,
            "should_use_lora": should_use_lora, "mlp_ft_dropout": mlp_ft_dropout,
            "mlp_hidden_size": mlp_hidden_size, "mlp_target_size": num_target,
            "num_steps": num_steps, "lr": lr, "lr_multiplier": lr_multiplier,
            "micro_batch_size": micro_batch_size, "precision": precision,
        })

        # Encoder is frozen, so only the head trains; the container scaled the head LR by
        # lr_multiplier (scale-lr-layer=regression_head/classification_head) — fold it into the LR.
        args = TrainingArguments(
            output_dir=f"{work_dir}/trainer",
            max_steps=num_steps,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            learning_rate=lr * lr_multiplier,
            bf16=use_bf16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=max(1, min(10, num_steps)),
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                          eval_dataset=val_ds, data_collator=collate)
        trainer.train()

        # Final validation metrics (regression: MSE/MAE; classification: accuracy).
        preds = trainer.predict(val_ds)
        logits = torch.tensor(np.asarray(preds.predictions)).float()
        y = torch.tensor(np.asarray(preds.label_ids))
        if task_type == "regression":
            yhat = logits.squeeze(-1)
            mlflow.log_metric("val_mse", float(F.mse_loss(yhat, y.float())))
            mlflow.log_metric("val_mae", float((yhat - y.float()).abs().mean()))
        else:
            mlflow.log_metric("val_accuracy", float((logits.argmax(-1) == y).float().mean()))

        # --- save the HF weights dir (state_dict + reconstruction config + tokenizer) ---
        model_pt = f"{ft_weights_dir}/gwb_esm2_model.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "esm_variant": esm_variant, "hidden_size": hidden_size,
                "mlp_hidden_size": mlp_hidden_size, "mlp_target_size": num_target,
                "mlp_ft_dropout": mlp_ft_dropout, "task_type": task_type,
                "te_backend": te_how, "label_classes": label_classes, "precision": precision,
            },
        }, model_pt)
        tokenizer.save_pretrained(ft_weights_dir)

        # --- copy weights to the UC Volume ---
        try:
            dbutils.fs.rm(ft_weights_volume_location, True)
        except Exception:
            pass
        dbutils.fs.mkdirs(ft_weights_volume_location)
        shutil.copytree(ft_weights_dir, ft_weights_volume_location, dirs_exist_ok=True)
        print(f"weights -> {ft_weights_volume_location}")
        is_finetune_success = True

        # --- record in bionemo_weights (native spark.sql on serverless) ---
        ft_id = time.time_ns()
        spark.sql(f"""
            INSERT INTO {catalog}.{schema}.bionemo_weights VALUES (
                {ft_id}, '{finetune_label}', 'esm2', '{esm_variant}', '{experiment_name}',
                '{active_run_id}', '{ft_weights_volume_location}', '{user_email}',
                CURRENT_TIMESTAMP(), true, NULL
            )
        """)
        mlflow.set_tag("ft_id", str(ft_id))
        mlflow.set_tag("job_status", "complete")
        print(f"bionemo_weights row inserted, ft_id = {ft_id}")
except Exception as exc:
    if active_run_id:
        try:
            MlflowClient().set_tag(active_run_id, "job_status", "failed")
            MlflowClient().set_tag(active_run_id, "failure_reason", str(exc)[:500])
        except Exception:
            pass
    raise

# COMMAND ----------

print("DONE" if is_finetune_success else "No deployments made")
