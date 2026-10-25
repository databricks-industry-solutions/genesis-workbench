# Databricks notebook source
# MAGIC %md
# MAGIC # RFD4-Proteína — Fine-tune (PEFT/LoRA on the released flow checkpoint)
# MAGIC
# MAGIC KERMT-v2 finetune pattern, RFD4 version: dispatcher-triggered GPU job that attaches to a
# MAGIC pre-created MLflow run, runs `rfd4-train` (PEFT adapter on top of the base flow ckpt), copies the
# MAGIC adapter to the cache Volume, and records it in `rfd4_weights` for `03_rfd4_register_serving` to deploy.
# MAGIC Runs on an **H100 / AI Runtime v6 (CUDA 13)** node — same stack as register.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Parameters
# MAGIC All inputs are notebook widgets so the fine-tune orchestrator job (and a human re-run) can override
# MAGIC them. `pretrain_ckpt_path` (a native ckpt) and the HF bundle are mutually exclusive — leave it blank to
# MAGIC fine-tune from the public HF base weights. `mlflow_run_id` is the dispatcher's pre-created run to attach to.

# COMMAND ----------

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("cache_dir", "rfd4_proteina", "Cache dir (UC volume)")
dbutils.widgets.text("rfd4_git_url", "github.com/RosettaCommons/RFD4-Proteina.git", "RFD4 git host/path (public repo)")
dbutils.widgets.text("rfd4_git_ref", "v0.0.1", "RFD4 git ref (release tag/commit)")
dbutils.widgets.text("rfd4_hf_repo", "nvidia/RFD4-Proteina", "HF repo for base weights (matched flow+AE bundle)")
dbutils.widgets.text("rfd4_hf_revision", "a06e2509c7033b010fc43614e9fd840303588502", "HF Hub commit for the base weights")
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

get_widget = dbutils.widgets.get
catalog, schema, cache_dir = get_widget("catalog"), get_widget("schema"), get_widget("cache_dir")
user_email = get_widget("user_email")
git_url, git_ref = get_widget("rfd4_git_url"), get_widget("rfd4_git_ref")
hf_repo, hf_revision = get_widget("rfd4_hf_repo"), get_widget("rfd4_hf_revision")
pretrain_ckpt_path, train_data_location = get_widget("pretrain_ckpt_path"), get_widget("train_data_location")
experiment_preset, finetune_label = get_widget("experiment_preset"), get_widget("finetune_label")
max_epochs, steps_per_epoch = get_widget("max_epochs"), get_widget("steps_per_epoch")
experiment_name, mlflow_run_name, mlflow_run_id = get_widget("experiment_name"), get_widget("mlflow_run_name"), get_widget("mlflow_run_id")
volume_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Install the rfproteina CUDA-13 stack (same as register) + ensure `rfd4_weights` exists
# MAGIC Install `rfproteina` from the **public GitHub repo** (nothing vendored; customer needs nothing local);
# MAGIC `rfd4-train` lands on PATH as a console entry point. Same CUDA-13 stack as register
# MAGIC (torch 2.14.1/cu132 + TMol source build + cuEquivariance), installed into THIS kernel env via
# MAGIC `--python sys.executable` (the diffdock pattern; `uv --system` hits the read-only `/usr`).
# MAGIC
# MAGIC **typing_extensions fix:** the `rfproteina` install pulls a newer `anyio` that needs `from
# MAGIC typing_extensions import sentinel` (added in typing_extensions 4.14). `rfproteina`'s own resolution can
# MAGIC leave an older/capped `te`, so we FORCE the latest into this kernel env (ephemeral, first on sys.path)
# MAGIC with `--reinstall-package` so it wins over any cap and the base image's old copy. Without this, `import
# MAGIC mlflow` (and `rfd4-train`) ImportError at startup. The verbose log + self-check show whether the upgrade
# MAGIC actually took. Finally, ensure the `rfd4_weights` table exists so the training result can be recorded.
# MAGIC
# MAGIC Note: this cell intentionally does **not** `restartPython()` — the next cell manipulates `sys.path`
# MAGIC in-process so the upgraded `typing_extensions` resolves first.

# COMMAND ----------

import os, shutil, subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
rfd4_package_spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{git_url}@{git_ref}"
install_result = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                     "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", rfd4_package_spec],
                    capture_output=True, text=True)
if install_result.returncode != 0:
    install_stderr = (install_result.stderr or "")
    raise RuntimeError(f"uv install failed rc={install_result.returncode}; stderr tail:\n{install_stderr[-3000:]}")

# Force typing_extensions>=4.14 into this kernel env (see the markdown above for why).
te_install_result = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                      "--reinstall-package", "typing_extensions", "typing_extensions>=4.14"],
                     capture_output=True, text=True)
print("typing_extensions upgrade rc:", te_install_result.returncode, "|", (te_install_result.stderr or te_install_result.stdout or "")[-300:])
sentinel_check = subprocess.run([sys.executable, "-c",
                       "from typing_extensions import sentinel; import importlib.metadata as m;"
                       "print('sentinel OK, te='+m.version('typing_extensions'))"],
                      capture_output=True, text=True)
print("sentinel check:", (sentinel_check.stdout.strip() or sentinel_check.stderr.strip()[-250:]))

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.rfd4_weights (
        ft_id BIGINT, ft_label STRING, model_type STRING, experiment_name STRING,
        run_id STRING, adapter_volume_location STRING, pretrain_ckpt STRING,
        created_by STRING, created_datetime TIMESTAMP, is_active BOOLEAN, deactivated_timestamp TIMESTAMP
    )
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Attach to the MLflow run + run `rfd4-train` (PEFT adapter)
# MAGIC First fix `sys.path`: this notebook process would otherwise resolve `typing_extensions` to the base
# MAGIC image's OLD copy before the ephemeral env's upgraded one, so `import mlflow` → `anyio` → `from
# MAGIC typing_extensions import sentinel` ImportErrors. Put the ephemeral site-packages first and drop any
# MAGIC cached `te` so the import re-resolves to the new one. (`rfd4-train` runs as its own subprocess, whose
# MAGIC fresh interpreter already resolves the ephemeral env first, so it needs no change.)
# MAGIC
# MAGIC Then attach to the dispatcher's pre-created MLflow run (or start one), and run the canonical v0.0.1
# MAGIC fine-tune (`docs/contributing/developer-guide.md §"Run the finetune"`): a NEW run from the public
# MAGIC portable bundle via `hf_ckpt_path` on the self-contained tutorial dataset (staged to
# MAGIC `${DATA_PATH}/test_dataset` by `register_rfd4_jobs`). Command corrections vs the old drifted command:
# MAGIC `out_dir=<dir>` is the real run-dir key (there is NO `paths.output_dir`; ckpts land in
# MAGIC `out_dir/checkpoints`); `log.log_wandb=false` disables W&B (no creds) — do NOT use `+nolog`, which gates
# MAGIC checkpoint writing off (`train.py:484`); `hf_ckpt_path`/`pretrain_ckpt_path` are mutually exclusive
# MAGIC (`train.py:234`). `+single=true` only drops the SLURM plugin (`train.py:417`) and leaves
# MAGIC `opt.dist_strategy=ddp`, whose launcher hangs on TCPStore rendezvous on a 1-GPU node — so set
# MAGIC `opt.dist_strategy=auto` to make Lightning pick `SingleDeviceStrategy` (what the docs say `+single` gives).
# MAGIC On success, copy the adapter to the cache Volume and record it in `rfd4_weights`.

# COMMAND ----------

ephemeral_site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib",
                       f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
if os.path.isdir(ephemeral_site_packages):
    sys.path.insert(0, ephemeral_site_packages)
sys.modules.pop("typing_extensions", None)
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
    if pretrain_ckpt_path.strip():
        base_args = [f"pretrain_ckpt_path={pretrain_ckpt_path}", "hf_ckpt_path=null"]
    else:
        base_args = [f"hf_ckpt_path={hf_repo}", f"hf_ckpt_revision={hf_revision}", "pretrain_ckpt_path=null"]
    cmd = ["rfd4-train", f"experiment={experiment_preset}", *base_args,
           f"opt.max_epochs={max_epochs}", f"opt.steps_per_epoch={steps_per_epoch}",
           f"out_dir={save_dir}", "+single=true", "opt.dist_strategy=auto", "log.log_wandb=false"]
    env = {**os.environ, "DATA_PATH": volume_root}  # rfproteina is in the kernel env; rfd4-train is on PATH
    print("running:", " ".join(cmd))
    train_result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Ensure the finetune_logs dir exists (makedirs side-effect), then write the train log.
    (open(f"{volume_root}/finetune_logs/{finetune_label}.log", "w")
     if os.path.isdir(f"{volume_root}/finetune_logs") or not os.makedirs(f"{volume_root}/finetune_logs", exist_ok=True)
     else None)
    with open(f"{volume_root}/finetune_logs/{finetune_label}.log", "w") as log_file:
        log_file.write(train_result.stdout + "\n==STDERR==\n" + train_result.stderr)
    if train_result.returncode != 0:
        raise RuntimeError(f"rfd4-train failed (rc={train_result.returncode}); see log at {volume_root}/finetune_logs/{finetune_label}.log")

    # Copy adapter/checkpoint to the cache Volume + record in rfd4_weights.
    adapter_loc = f"{volume_root}/finetuned/{finetune_label}"
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
except Exception as error:
    mlflow.set_tags({"job_status": "failed", "failure_reason": str(error)[:500]})
    raise
finally:
    mlflow.end_run()
