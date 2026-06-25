# Databricks notebook source
# MAGIC %md
# MAGIC # KERMT — Fine-tune (orchestrator)
# MAGIC Fine-tunes KERMT (GROVERbase) on a SMILES+target dataset and records the result.
# MAGIC Batch-workflow Layer 4: attaches to the dispatcher's pre-created MLflow run, advances
# MAGIC `job_status` (submitted→training→complete/failed), and inserts a `kermt_weights` row.
# MAGIC
# MAGIC Runs on a classic single-node A10 cluster (15.4 gpu-ml). Installs KERMT pip-only and uses
# MAGIC the RDKit featurization path (no cuik_molmaker) — validated by the spike.

# COMMAND ----------

dbutils.widgets.text("catalog", "srijit_nair_ci_demo_catalog", "Catalog")
dbutils.widgets.text("schema", "genesis_workbench", "Schema")
dbutils.widgets.text("cache_dir", "kermt", "KERMT UC volume name")
dbutils.widgets.text("kermt_src_path", "", "Workspace path to vendored kermt_src (bundle files)")
dbutils.widgets.text("user_email", "srijit.nair@databricks.com", "User Id/Email")

dbutils.widgets.text("train_data_location", "", "Train CSV (UC volume path; smiles + target cols)")
dbutils.widgets.text("validation_data_location", "", "Validation CSV")
dbutils.widgets.text("test_data_location", "", "Test CSV")
dbutils.widgets.text("target_names", "toxicity", "Comma-separated target/task column names")
dbutils.widgets.text("dataset_type", "classification", "regression | classification")

dbutils.widgets.text("finetune_label", "kermt_ft_xyz", "Unique label for these fine-tuned weights")
dbutils.widgets.text("epochs", "20", "Fine-tune epochs")
dbutils.widgets.text("batch_size", "16", "Batch size")
dbutils.widgets.text("ffn_hidden_size", "700", "FFN hidden size")

dbutils.widgets.text("experiment_name", "gwb_kermt_finetune", "MLflow experiment (short tag)")
dbutils.widgets.text("mlflow_run_name", "", "MLflow run name")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (dispatcher); empty = create new")

# COMMAND ----------

# MAGIC %pip install -q rdkit==2025.3.6 descriptastorus==2.8.0 optuna==4.1.0 scikit-learn==1.5.2 tensorboard==2.18.0 mlflow[databricks]==2.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json, time, glob, shutil, subprocess, traceback

g = dbutils.widgets.get
catalog, schema, cache_dir = g("catalog"), g("schema"), g("cache_dir")
kermt_src_path = g("kermt_src_path")
user_email = g("user_email")
train_loc, val_loc, test_loc = g("train_data_location"), g("validation_data_location"), g("test_data_location")
target_names = [t.strip() for t in g("target_names").split(",") if t.strip()]
dataset_type = g("dataset_type")
finetune_label = g("finetune_label")
epochs, batch_size, ffn_hidden_size = int(g("epochs")), int(g("batch_size")), int(g("ffn_hidden_size"))
experiment_name = g("experiment_name")
mlflow_run_name = g("mlflow_run_name") or finetune_label
mlflow_run_id = g("mlflow_run_id") or None

vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"
metric = "auc" if dataset_type == "classification" else "mae"

# Stage the vendored (patched) KERMT source to a writable local dir.
WS_SRC = ("/Workspace" + kermt_src_path) if kermt_src_path else None
LOCAL_SRC = "/local_disk0/kermt_src"
shutil.rmtree(LOCAL_SRC, ignore_errors=True)
assert WS_SRC and os.path.exists(WS_SRC), f"kermt_src not found at {WS_SRC}"
shutil.copytree(WS_SRC, LOCAL_SRC)
print("KERMT source staged to", LOCAL_SRC)

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Attach to the dispatcher's pre-created run, else create one (direct Jobs-UI fallback).
if mlflow_run_id:
    _run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    # Canonical GWB deploy-time experiment location: the Shared base path used by
    # genesis_workbench.models.set_mlflow_experiment(..., shared=True) and the register
    # notebooks. mkdirs the parent first so a fresh user/workspace doesn't fail inside
    # create_experiment (BAD_REQUEST: "For input string: None").
    from databricks.sdk import WorkspaceClient
    _exp_base = "Shared/dbx_genesis_workbench_models"
    WorkspaceClient().workspace.mkdirs(f"/Workspace/{_exp_base}")
    exp_path = f"/{_exp_base}/{experiment_name}"
    exp = mlflow.set_experiment(exp_path)
    MlflowClient().set_experiment_tag(exp.experiment_id, "used_by_genesis_workbench", "yes")
    _run_ctx = mlflow.start_run(run_name=mlflow_run_name)

active_run_id = None
try:
    with _run_ctx as run:
        active_run_id = run.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "kermt_finetune")
        mlflow.set_tag("created_by", user_email)
        mlflow.set_tag("job_status", "started")
        mlflow.log_params({
            "model_type": "kermt", "dataset_type": dataset_type,
            "target_names": ",".join(target_names), "epochs": epochs,
            "batch_size": batch_size, "ffn_hidden_size": ffn_hidden_size,
            "finetune_label": finetune_label,
        })

        # --- stage data + pretrained checkpoint locally ---
        data_dir = "/local_disk0/kermt_data"; os.makedirs(data_dir, exist_ok=True)
        local_train = f"{data_dir}/train.csv"; local_val = f"{data_dir}/val.csv"; local_test = f"{data_dir}/test.csv"
        shutil.copy(train_loc, local_train); shutil.copy(val_loc, local_val); shutil.copy(test_loc, local_test)
        ckpt_local = f"{LOCAL_SRC}/kermt_contrastive_v2.0.pt"
        shutil.copy(f"{vol_root}/pretrained/kermt_contrastive_v2.0.pt", ckpt_local)

        mlflow.set_tag("job_status", "training")
        save_dir = "/local_disk0/kermt_run/finetune"

        env = os.environ.copy()
        env["PYTHONPATH"] = LOCAL_SRC
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cmd = ["python", "main.py", "finetune",
               "--data_path", local_train, "--separate_val_path", local_val, "--separate_test_path", local_test,
               "--save_dir", save_dir, "--checkpoint_path", "kermt_contrastive_v2.0.pt",
               "--dataset_type", dataset_type, "--split_type", "scaffold_balanced",
               "--ensemble_size", "1", "--num_folds", "1", "--no_features_scaling",
               "--ffn_hidden_size", str(ffn_hidden_size), "--ffn_num_layers", "3", "--bond_drop_rate", "0.1",
               "--epochs", str(epochs), "--warmup_epochs", str(min(2.0, max(0.5, epochs / 2.0))),
               "--metric", metric, "--self_attention", "--dist_coff", "0.15",
               "--max_lr", "1e-4", "--final_lr", "2e-5", "--dropout", "0.0",
               "--features_generator", "rdkit_2d_normalized_onthefly",
               "--batch_size", str(batch_size), "--seed", "50"]
        print("$", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=LOCAL_SRC, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        out = proc.stdout
        print(out[-4000:])
        # Always persist the full KERMT log to the volume for debugging (FUSE POSIX).
        log_path = f"{vol_root}/finetune_logs/{finetune_label}.log"
        os.makedirs(f"{vol_root}/finetune_logs", exist_ok=True)
        with open(log_path, "w") as _f:
            _f.write(out)
        print("full KERMT log ->", log_path)
        if proc.returncode != 0:
            raise RuntimeError(f"KERMT finetune failed (rc={proc.returncode}) — see {log_path}")

        # Parse the test metric from KERMT's stdout (e.g. "Overall test <metric> = X +/- Y").
        test_score = None
        for line in out.splitlines():
            low = line.lower()
            if "test" in low and metric in low and "=" in line:
                try:
                    test_score = float(line.split("=")[1].split("+/-")[0].strip().split()[0])
                except Exception:
                    pass
        if test_score is not None:
            mlflow.log_metric(f"test_{metric}", test_score)
        print(f"test_{metric} = {test_score}")

        # --- copy the fine-tuned checkpoint dir to the UC volume ---
        weights_loc = f"{vol_root}/finetuned/{finetune_label}"
        # remove stale, then copy the full save_dir (model + args + scaler) for predict/serving
        try:
            dbutils.fs.rm(weights_loc, True)
        except Exception:
            pass
        shutil.copytree(save_dir, weights_loc)
        print("weights ->", weights_loc)
        # Carry the test split with the weights so the deploy job can fit probability
        # calibration on a labeled holdout specific to this model.
        try:
            shutil.copy(local_test, f"{weights_loc}/calibration.csv")
            print("calibration holdout ->", f"{weights_loc}/calibration.csv")
        except Exception as _e:
            print("could not stage calibration.csv:", _e)

        # --- record in kermt_weights ---
        ft_id = time.time_ns()
        spark.sql(f"""
            INSERT INTO {catalog}.{schema}.kermt_weights VALUES (
                {ft_id}, '{finetune_label}', 'kermt', '{dataset_type}', '{",".join(target_names)}',
                '{experiment_name}', '{active_run_id}', '{weights_loc}', '{user_email}',
                CURRENT_TIMESTAMP(), true, NULL
            )
        """)
        mlflow.set_tag("ft_id", str(ft_id))
        mlflow.set_tag("weights_volume_location", weights_loc)
        mlflow.set_tag("result_location", weights_loc)
        mlflow.set_tag("job_status", "complete")
        print("kermt_weights row inserted, ft_id =", ft_id)
except Exception as exc:
    if active_run_id:
        try:
            MlflowClient().set_tag(active_run_id, "job_status", "failed")
            MlflowClient().set_tag(active_run_id, "failure_reason", str(exc)[:500])
        except Exception:
            pass
    raise
