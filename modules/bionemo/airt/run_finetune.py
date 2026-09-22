#!/usr/bin/env python3
"""BioNeMo ESM2 fine-tune for the serverless-GPU ai_runtime_task (bionemo_esm_finetune).

Converted from notebooks/bionemo_esm_finetune.py to run INSIDE the NGC BioNeMo container on
serverless GPU_1xA10 (the classic a10 GPU cluster can't provision at EC2 GPU quota=0). Runs the
`finetune_esm2` CLI (patched — see command.sh), then logs to MLflow and records the fine-tune in
the bionemo_weights table.

Container-vs-notebook adaptations:
  * inputs come from env (command.sh maps the run-now job_parameters) + a shared UC-Volume config
    fallback, not dbutils.widgets;
  * node-local scratch is /tmp (no /local_disk0/ /workdir on serverless);
  * /Volumes I/O uses os/shutil (readable+writable in-container), not dbutils.fs;
  * the bionemo_weights INSERT uses the SQL warehouse via databricks-sql-connector (no Spark in
    the container), not spark.sql;
  * MLflow + SQL need Databricks auth in the container — DATABRICKS_HOST/DATABRICKS_TOKEN from env
    (see command.sh / the job). UNCONFIRMED whether the ai_runtime_task provides these; validate
    on the first GPU run (blocked on A10 capacity). If unavailable, the MLflow+DB steps degrade to
    warnings and the fine-tune + weight-copy still complete.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def load_shared_config() -> None:
    cfg = _env("PB_CONFIG")
    if cfg and os.path.exists(cfg):
        try:
            for k, v in json.load(open(cfg)).items():
                os.environ.setdefault(k, str(v))
            print(f"[config] merged {cfg}", flush=True)
        except Exception as e:
            print(f"[config] WARN {e}", flush=True)


load_shared_config()

CATALOG = _env("PB_CATALOG", "genesis_workbench")
SCHEMA = _env("PB_SCHEMA", "genesis_workbench")
SQL_WAREHOUSE_ID = _env("PB_SQL_WAREHOUSE_ID")
MODEL_VOLUME = _env("PB_MODEL_VOLUME", "bionemo")
ESM_VARIANT = _env("PB_ESM_VARIANT", "650M")
TRAIN_DATA = _env("PB_TRAIN_DATA")
VAL_DATA = _env("PB_VAL_DATA")
FINETUNE_LABEL = _env("PB_FINETUNE_LABEL", "esm_ft")
TASK_TYPE = _env("PB_TASK_TYPE", "regression")
MLP_DROPOUT = _env("PB_MLP_DROPOUT", "0.25")
MLP_HIDDEN = _env("PB_MLP_HIDDEN", "256")
MLP_TARGET = _env("PB_MLP_TARGET", "1")
EXPERIMENT_NAME = _env("PB_EXPERIMENT_NAME", "sequence_level_regression")
NUM_STEPS = _env("PB_NUM_STEPS", "50")
LR = _env("PB_LR", "5e-3")
LR_MULT = _env("PB_LR_MULTIPLIER", "1e2")
MICRO_BATCH = _env("PB_MICRO_BATCH_SIZE", "2")
PRECISION = _env("PB_PRECISION", "bf16-mixed")
USER_EMAIL = _env("PB_USER_EMAIL", "")
MLFLOW_RUN_ID = _env("PB_MLFLOW_RUN_ID") or None
SCALE_LR_LAYER = "regression_head" if TASK_TYPE == "regression" else "classification_head"

WORK = _env("PB_WORK_DIR", "/tmp/bionemo_ft")
FT_WEIGHTS_DIR = f"{WORK}/ft_weights"
TRAIN_CSV = f"{WORK}/data/train/train.csv"
VAL_CSV = f"{WORK}/data/val/val.csv"
VOL_WEIGHTS = f"/Volumes/{CATALOG}/{SCHEMA}/{MODEL_VOLUME}/esm2/{ESM_VARIANT}/{FINETUNE_LABEL}"

DB_HOST = _env("DATABRICKS_HOST")
DB_TOKEN = _env("DATABRICKS_TOKEN")


def run(cmd: str) -> int:
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"[exit {rc} in {time.time()-t0:.0f}s]", flush=True)
    return rc


def _mlflow():
    """MLflow handle configured for the workspace, or None if unavailable in-container."""
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        return mlflow
    except Exception as e:
        print(f"[mlflow] unavailable: {e}", flush=True)
        return None


def prepare_data():
    import pandas as pd
    for d in (f"{WORK}/data/train", f"{WORK}/data/val", FT_WEIGHTS_DIR):
        os.makedirs(d, exist_ok=True)
    os.makedirs(VOL_WEIGHTS, exist_ok=True)
    # Volume CSVs use sequence/target; the CLI expects sequences/labels.
    pd.read_csv(TRAIN_DATA)[["sequence", "target"]].rename(
        columns={"sequence": "sequences", "target": "labels"}).to_csv(TRAIN_CSV, index=False)
    pd.read_csv(VAL_DATA)[["sequence", "target"]].rename(
        columns={"sequence": "sequences", "target": "labels"}).to_csv(VAL_CSV, index=False)


def download_checkpoint() -> str:
    from bionemo.core.data.load import load
    p = load(f"esm2/{ESM_VARIANT.lower()}:2.0")
    print(f"[ckpt] {p}", flush=True)
    return str(p)


def log_tensorboard_metrics(mlflow, run_id):
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(
            f"{FT_WEIGHTS_DIR}/{EXPERIMENT_NAME}/dev",
            size_guidance={event_accumulator.SCALARS: 0}).Reload()
        for k in ea.scalars.Keys():
            for v in ea.Scalars(k):
                mlflow.log_metric(k, v.value, step=v.step + 1)
    except Exception as e:
        print(f"[mlflow] tensorboard metric export skipped: {e}", flush=True)


def copy_weights_to_volume() -> bool:
    ckpt_dir = f"{FT_WEIGHTS_DIR}/{EXPERIMENT_NAME}/dev/checkpoints"
    if not os.path.isdir(ckpt_dir):
        print(f"[weights] no checkpoints dir at {ckpt_dir}", flush=True)
        return False
    for name in os.listdir(ckpt_dir):
        src = f"{ckpt_dir}/{name}"
        if os.path.isdir(src) and name.endswith("-last"):
            print(f"[weights] copying {src} -> {VOL_WEIGHTS}", flush=True)
            shutil.copytree(src, VOL_WEIGHTS, dirs_exist_ok=True)
            return True
    return False


def record_bionemo_weights(run_id: str):
    """INSERT the fine-tune row via the SQL warehouse (no Spark in the container)."""
    if not (SQL_WAREHOUSE_ID and DB_HOST and DB_TOKEN):
        print("[db] SQL warehouse creds unavailable in-container; skipping bionemo_weights insert "
              "(record it from the dispatcher/a follow-up task instead)", flush=True)
        return
    try:
        from databricks import sql
        host = DB_HOST.replace("https://", "").rstrip("/")
        with sql.connect(server_hostname=host,
                         http_path=f"/sql/1.0/warehouses/{SQL_WAREHOUSE_ID}",
                         access_token=DB_TOKEN) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {CATALOG}.{SCHEMA}.bionemo_weights(
                        ft_id, ft_label, model_type, variant, experiment_name, run_id,
                        weights_volume_location, created_by, created_datetime, is_active,
                        deactivated_timestamp)
                    VALUES ({time.time_ns()}, '{FINETUNE_LABEL}', 'esm2', '{ESM_VARIANT}',
                        '{EXPERIMENT_NAME}', '{run_id}', '{VOL_WEIGHTS}', '{USER_EMAIL}',
                        CURRENT_TIMESTAMP(), true, NULL)
                """)
        print("[db] bionemo_weights row inserted", flush=True)
    except Exception as e:
        print(f"[db] WARN bionemo_weights insert failed: {e}", flush=True)


def main():
    run("nvidia-smi -L")
    prepare_data()
    ckpt = download_checkpoint()

    mlflow = _mlflow()
    run_id = MLFLOW_RUN_ID or ""
    if mlflow:
        try:
            if MLFLOW_RUN_ID:
                r = mlflow.start_run(run_id=MLFLOW_RUN_ID)
            else:
                # Replicate genesis_workbench.set_mlflow_experiment: per-user experiment + tag.
                exp_path = f"/Users/{USER_EMAIL}/mlflow_experiments/{EXPERIMENT_NAME}"
                mlflow.set_experiment(exp_path)
                mlflow.set_experiment_tag("used_by_genesis_workbench", "yes")
                r = mlflow.start_run(run_name=FINETUNE_LABEL)
            run_id = r.info.run_id
            for k, v in {"origin": "genesis_workbench", "feature": "bionemo_esm_finetune",
                         "created_by": USER_EMAIL, "result_location": VOL_WEIGHTS,
                         "job_status": "training"}.items():
                mlflow.set_tag(k, v)
            mlflow.log_params({"esm_variant": ESM_VARIANT, "finetune_label": FINETUNE_LABEL,
                               "task_type": TASK_TYPE, "num_steps": NUM_STEPS, "lr": LR,
                               "micro_batch_size": MICRO_BATCH, "precision": PRECISION})
        except Exception as e:
            print(f"[mlflow] setup failed: {e}", flush=True)
            mlflow = None

    rc = run(
        f"finetune_esm2 --restore-from-checkpoint-path {ckpt} "
        f"--train-data-path {TRAIN_CSV} --valid-data-path {VAL_CSV} "
        f"--config-class ESM2FineTuneSeqConfig --dataset-class InMemorySingleValueDataset "
        f"--task-type {TASK_TYPE} --mlp-ft-dropout {MLP_DROPOUT} --mlp-hidden-size {MLP_HIDDEN} "
        f"--mlp-target-size {MLP_TARGET} --experiment-name {EXPERIMENT_NAME} "
        f"--num-steps {NUM_STEPS} --num-gpus 1 --val-check-interval 10 --log-every-n-steps 10 "
        f"--encoder-frozen --lr {LR} --lr-multiplier {LR_MULT} --scale-lr-layer {SCALE_LR_LAYER} "
        f"--result-dir {FT_WEIGHTS_DIR} --micro-batch-size {MICRO_BATCH} --precision {PRECISION} "
        f"--create-tensorboard-logger")
    if rc != 0:
        if mlflow:
            try:
                mlflow.set_tag("job_status", "failed"); mlflow.end_run(status="FAILED")
            except Exception:
                pass
        sys.exit(rc)

    if mlflow:
        log_tensorboard_metrics(mlflow, run_id)
        try:
            mlflow.end_run()
        except Exception:
            pass

    ok = copy_weights_to_volume()
    if ok:
        record_bionemo_weights(run_id)
        if mlflow:
            try:
                from mlflow.tracking import MlflowClient
                MlflowClient().set_tag(run_id, "job_status", "complete")
            except Exception:
                pass
        print(f"\n=== DONE === weights: {VOL_WEIGHTS}", flush=True)
    else:
        print("=== fine-tune produced no -last checkpoint ===", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
