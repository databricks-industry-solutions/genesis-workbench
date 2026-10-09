# Databricks notebook source
# MAGIC %md
# MAGIC # RFD4-Proteína — Fine-tune (PEFT/LoRA on the released flow checkpoint)
# MAGIC
# MAGIC KERMT-v2 finetune pattern, RFD4 version: dispatcher-triggered GPU job that attaches to a
# MAGIC pre-created MLflow run, runs `rfd4-train` (PEFT adapter on top of the base flow ckpt), copies the
# MAGIC adapter to the cache Volume, and records it in `rfd4_weights` for `03_rfd4_register_serving` to deploy.
# MAGIC Runs on an **H100 / AI Runtime v6 (CUDA 13)** node — same stack as register.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "rfd4_proteina", "Cache dir (UC volume)")
dbutils.widgets.text("rfd4_git_url", "github.com/NVIDIA-BioNeMo/RFD4-Proteina-Open-Model-Private.git", "RFD4 git host/path")
dbutils.widgets.text("rfd4_git_ref", "84d83b98442e5946a6a626d0d99a19773207e894", "RFD4 git ref (commit/branch/tag)")
dbutils.widgets.text("rfd4_git_token_scope", "dbx_genesis_workbench", "Secret scope for the GitHub PAT")
dbutils.widgets.text("rfd4_git_token_key", "rfd4_github_token", "Secret key for the GitHub PAT")
dbutils.widgets.text("user_email", "a@b.com", "User Id/Email")
dbutils.widgets.text("pretrain_ckpt_path", "", "Base flow checkpoint to fine-tune from")
dbutils.widgets.text("train_data_location", "", "Training data dir (parquet manifest + CIFs)")
dbutils.widgets.text("experiment_preset", "tutorial/finetune_test_dataset", "rfd4-train experiment preset")
dbutils.widgets.text("finetune_label", "rfd4_proteina_ft", "Unique label for this fine-tune")
dbutils.widgets.text("max_epochs", "5", "Max epochs")
dbutils.widgets.text("steps_per_epoch", "100", "Steps per epoch")
dbutils.widgets.text("experiment_name", "gwb_rfd4_finetune", "MLflow experiment")
dbutils.widgets.text("mlflow_run_name", "", "MLflow run name")
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (from dispatcher)")

g = dbutils.widgets.get
catalog, schema, cache_dir = g("catalog"), g("schema"), g("cache_dir")
user_email = g("user_email")
git_url, git_ref = g("rfd4_git_url"), g("rfd4_git_ref")
git_token_scope, git_token_key = g("rfd4_git_token_scope"), g("rfd4_git_token_key")
pretrain_ckpt_path, train_data_location = g("pretrain_ckpt_path"), g("train_data_location")
experiment_preset, finetune_label = g("experiment_preset"), g("finetune_label")
max_epochs, steps_per_epoch = g("max_epochs"), g("steps_per_epoch")
experiment_name, mlflow_run_name, mlflow_run_id = g("experiment_name"), g("mlflow_run_name"), g("mlflow_run_id")
vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# COMMAND ----------

# DBTITLE 1,Install the rfproteina CUDA-13 stack (same as register) + ensure rfd4_weights exists
import os, shutil, subprocess, sys
# Install rfproteina from the private GitHub repo (nothing vendored; customer needs nothing local),
# authenticating with a GitHub PAT from the GWB secret scope. `rfd4-train` lands on PATH as a console
# entry point. Same CUDA-13 stack as register (torch 2.14.1/cu132 + TMol source build + cuEquivariance).
try:
    gh_token = dbutils.secrets.get(git_token_scope, git_token_key)
except Exception:
    gh_token = None  # OK once the repo is public — fall back to an unauthenticated clone
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
# Install into THIS kernel's env (sys.executable → /databricks/python, writable; diffdock pattern).
# uv --system hits the read-only /usr; --python sys.executable upgrades deps in-place. rfd4-train lands
# on the kernel's bin (on PATH).
cred = f"x-access-token:{gh_token}@" if gh_token else ""
spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{cred}{git_url}@{git_ref}"
_r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                     "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", spec],
                    capture_output=True, text=True)
if _r.returncode != 0:
    _err = (_r.stderr or "")
    if gh_token:
        _err = _err.replace(gh_token, "<redacted>")
    hint = "" if gh_token else f" (no token at {git_token_scope}/{git_token_key}; set a PAT if the repo is still private)"
    raise RuntimeError(f"uv install failed rc={_r.returncode}{hint}; stderr tail:\n{_err[-3000:]}")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.rfd4_weights (
        ft_id BIGINT, ft_label STRING, model_type STRING, experiment_name STRING,
        run_id STRING, adapter_volume_location STRING, pretrain_ckpt STRING,
        created_by STRING, created_datetime TIMESTAMP, is_active BOOLEAN, deactivated_timestamp TIMESTAMP
    )
""")

# COMMAND ----------

# DBTITLE 1,Attach to MLflow run + run rfd4-train (PEFT adapter)
import time, mlflow
mlflow.set_registry_uri("databricks-uc")
if mlflow_run_id:
    run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    mlflow.set_experiment(f"/Users/{user_email}/{experiment_name}")
    run_ctx = mlflow.start_run(run_name=(mlflow_run_name or finetune_label))
active_run_id = run_ctx.info.run_id
mlflow.set_tags({"origin": "genesis_workbench", "feature": "rfd4_finetune",
                 "created_by": user_email, "job_status": "started"})
mlflow.log_params({"model_type": "rfd4_proteina", "experiment_preset": experiment_preset,
                   "finetune_label": finetune_label, "max_epochs": max_epochs})

save_dir = "/tmp/rfd4_ft_out"
os.makedirs(save_dir, exist_ok=True)
try:
    # ITERATE: rfd4-train arg surface. The repo's finetune tutorial uses `experiment=<preset>` +
    # `pretrain_ckpt_path=<ckpt>` + `+single=true`; dataset + epochs/steps are overlaid here. The
    # dataset manifest/CIFs come from train_data_location; adapter lands in save_dir.
    cmd = ["rfd4-train", f"experiment={experiment_preset}",
           f"pretrain_ckpt_path={pretrain_ckpt_path}",
           f"opt.max_epochs={max_epochs}", f"opt.steps_per_epoch={steps_per_epoch}",
           f"paths.output_dir={save_dir}", "+single=true", "+nolog=true"]
    if train_data_location:
        cmd.append(f"datasets.train.data_dir={train_data_location}")
    env = {**os.environ, "DATA_PATH": vol_root}  # rfproteina is in the kernel env; rfd4-train is on PATH
    print("running:", " ".join(cmd))
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (open(f"{vol_root}/finetune_logs/{finetune_label}.log", "w")
     if os.path.isdir(f"{vol_root}/finetune_logs") or not os.makedirs(f"{vol_root}/finetune_logs", exist_ok=True)
     else None)
    with open(f"{vol_root}/finetune_logs/{finetune_label}.log", "w") as lf:
        lf.write(res.stdout + "\n==STDERR==\n" + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"rfd4-train failed (rc={res.returncode}); see log at {vol_root}/finetune_logs/{finetune_label}.log")

    # Copy adapter/checkpoint to the cache Volume + record in rfd4_weights.
    adapter_loc = f"{vol_root}/finetuned/{finetune_label}"
    if os.path.isdir(adapter_loc):
        shutil.rmtree(adapter_loc)
    shutil.copytree(save_dir, adapter_loc)
    ft_id = time.time_ns()
    spark.sql(f"""
        INSERT INTO {catalog}.{schema}.rfd4_weights VALUES (
            {ft_id}, '{finetune_label}', 'rfd4_proteina', '{experiment_name}',
            '{active_run_id}', '{adapter_loc}', '{pretrain_ckpt_path}',
            '{user_email}', CURRENT_TIMESTAMP(), true, NULL)
    """)
    mlflow.set_tags({"ft_id": str(ft_id), "adapter_location": adapter_loc, "job_status": "complete"})
    print(f"fine-tune complete: ft_id={ft_id} adapter={adapter_loc}")
except Exception as e:
    mlflow.set_tags({"job_status": "failed", "failure_reason": str(e)[:500]})
    raise
finally:
    mlflow.end_run()
