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

g = dbutils.widgets.get
catalog, schema, cache_dir = g("catalog"), g("schema"), g("cache_dir")
user_email = g("user_email")
git_url, git_ref = g("rfd4_git_url"), g("rfd4_git_ref")
hf_repo, hf_revision = g("rfd4_hf_repo"), g("rfd4_hf_revision")
pretrain_ckpt_path, train_data_location = g("pretrain_ckpt_path"), g("train_data_location")
experiment_preset, finetune_label = g("experiment_preset"), g("finetune_label")
max_epochs, steps_per_epoch = g("max_epochs"), g("steps_per_epoch")
experiment_name, mlflow_run_name, mlflow_run_id = g("experiment_name"), g("mlflow_run_name"), g("mlflow_run_id")
vol_root = f"/Volumes/{catalog}/{schema}/{cache_dir}"

# COMMAND ----------

# DBTITLE 1,Install the rfproteina CUDA-13 stack (same as register) + ensure rfd4_weights exists
import os, shutil, subprocess, sys
# Install rfproteina from the PUBLIC GitHub repo (nothing vendored; customer needs nothing local).
# `rfd4-train` lands on PATH as a console entry point. Same CUDA-13 stack as register
# (torch 2.14.1/cu132 + TMol source build + cuEquivariance).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
open("/tmp/bc.txt", "w").write("torch==2.14.1\n")
# Install into THIS kernel's env (sys.executable → /databricks/python, writable; diffdock pattern).
# uv --system hits the read-only /usr; --python sys.executable upgrades deps in-place. rfd4-train lands
# on the kernel's bin (on PATH). The public repo needs no credential injection.
spec = f"rfd4-proteina[metrics-cuda13] @ git+https://{git_url}@{git_ref}"
_r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                     "--build-constraint", "/tmp/bc.txt", "--torch-backend", "cu132", spec],
                    capture_output=True, text=True)
if _r.returncode != 0:
    _err = (_r.stderr or "")
    raise RuntimeError(f"uv install failed rc={_r.returncode}; stderr tail:\n{_err[-3000:]}")

# The rfproteina install pulls a newer `anyio` that needs `from typing_extensions import sentinel`
# (added in typing_extensions 4.14). rfproteina's own resolution can leave an older/capped te, so FORCE
# the latest into this kernel env (ephemeral, first on sys.path) with --reinstall-package so it wins over
# any cap and the base image's old copy. Without this, `import mlflow` (and rfd4-train) ImportError at
# startup. Verbose + self-check so the fine-tune log shows whether the upgrade actually took.
_te = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--break-system-packages",
                      "--reinstall-package", "typing_extensions", "typing_extensions>=4.14"],
                     capture_output=True, text=True)
print("typing_extensions upgrade rc:", _te.returncode, "|", (_te.stderr or _te.stdout or "")[-300:])
_chk = subprocess.run([sys.executable, "-c",
                       "from typing_extensions import sentinel; import importlib.metadata as m;"
                       "print('sentinel OK, te='+m.version('typing_extensions'))"],
                      capture_output=True, text=True)
print("sentinel check:", (_chk.stdout.strip() or _chk.stderr.strip()[-250:]))

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.rfd4_weights (
        ft_id BIGINT, ft_label STRING, model_type STRING, experiment_name STRING,
        run_id STRING, adapter_volume_location STRING, pretrain_ckpt STRING,
        created_by STRING, created_datetime TIMESTAMP, is_active BOOLEAN, deactivated_timestamp TIMESTAMP
    )
""")

# COMMAND ----------

# DBTITLE 1,Attach to MLflow run + run rfd4-train (PEFT adapter)
# This notebook process resolves `typing_extensions` to the base image's OLD copy (/databricks/python)
# before the ephemeral env's upgraded one, so `import mlflow` -> anyio -> `from typing_extensions import
# sentinel` ImportErrors. Put the ephemeral site-packages (where we force-installed te>=4.14) first and
# drop any cached te so the import re-resolves to the new one. (rfd4-train runs as its own subprocess,
# whose fresh interpreter already resolves the ephemeral env first, so it needs no change.)
_eph_sp = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib",
                       f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
if os.path.isdir(_eph_sp):
    sys.path.insert(0, _eph_sp)
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
    # Canonical v0.0.1 fine-tune (docs/contributing/developer-guide.md §"Run the finetune"): start a NEW
    # run from the PUBLIC portable bundle via hf_ckpt_path on the self-contained tutorial dataset (staged
    # to ${DATA_PATH}/test_dataset by register_rfd4_jobs). Corrections vs the old drifted command:
    #   • out_dir=<dir>       — the REAL run-dir key (there is NO paths.output_dir); ckpts land in out_dir/checkpoints
    #   • log.log_wandb=false — disable W&B (no creds needed). Do NOT use +nolog: train.py gates checkpoint
    #                           writing on `not nolog` (train.py:484), so +nolog=true silently disabled ckpts.
    #   • hf_ckpt_path / pretrain_ckpt_path are MUTUALLY EXCLUSIVE (train.py:234) — a user-supplied native
    #     ckpt wins, otherwise the HF bundle (pretrain_ckpt_path=null).
    if pretrain_ckpt_path.strip():
        base_args = [f"pretrain_ckpt_path={pretrain_ckpt_path}", "hf_ckpt_path=null"]
    else:
        base_args = [f"hf_ckpt_path={hf_repo}", f"hf_ckpt_revision={hf_revision}", "pretrain_ckpt_path=null"]
    # Single serverless H100: force a single-device strategy. `+single=true` only drops the SLURM
    # plugin (train.py:417) — the trainer otherwise keeps opt.dist_strategy=ddp (default), whose subprocess
    # launcher hangs on TCPStore rendezvous on a 1-GPU serverless node (30-min timeout -> fail at
    # setup_distributed, before any training step). opt.dist_strategy=auto makes Lightning pick
    # SingleDeviceStrategy (no DDP/rendezvous) — which is what the docs say `+single` gives ("no SLURM or DDP").
    cmd = ["rfd4-train", f"experiment={experiment_preset}", *base_args,
           f"opt.max_epochs={max_epochs}", f"opt.steps_per_epoch={steps_per_epoch}",
           f"out_dir={save_dir}", "+single=true", "opt.dist_strategy=auto", "log.log_wandb=false"]
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
