#!/usr/bin/env python3
"""BioNeMo ESM2 inference for the serverless-GPU ai_runtime_task (bionemo_esm_inference).

Converted from notebooks/bionemo_esm_inference.py to run INSIDE the NGC BioNeMo container on
serverless GPU_1xA10. Runs the `infer_esm2` CLI on base or fine-tuned ESM2 weights and writes a
predictions CSV to the UC Volume. Same container-vs-notebook adaptations as run_finetune.py:
env inputs (+ shared-Volume config), /tmp scratch, os/shutil for /Volumes, the bionemo_weights
lookup via the SQL warehouse (not spark.sql), MLflow guarded (DATABRICKS_HOST/TOKEN from env).
"""
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
IS_BASE_MODEL = _env("PB_IS_BASE_MODEL", "true").lower() in ("1", "true", "yes")
ESM_VARIANT = _env("PB_ESM_VARIANT", "650M")
TASK_TYPE = _env("PB_TASK_TYPE", "regression")
FINETUNE_RUN_ID = _env("PB_FINETUNE_RUN_ID")
DATA_LOCATION = _env("PB_DATA_LOCATION")
SEQUENCE_COLUMN = _env("PB_SEQUENCE_COLUMN", "sequence")
RESULT_LOCATION = _env("PB_RESULT_LOCATION")
USER_EMAIL = _env("PB_USER_EMAIL", "")
EXPERIMENT_NAME = _env("PB_EXPERIMENT_NAME", "gwb_bionemo_esm2_inference")
RUN_NAME = _env("PB_RUN_NAME", "esm2_inference")
MLFLOW_RUN_ID = _env("PB_MLFLOW_RUN_ID") or None

WORK = _env("PB_WORK_DIR", "/tmp/bionemo_infer")
DATA_DIR = f"{WORK}/data"
RESULTS_DIR = f"{WORK}/results"
FT_WEIGHTS_DIR = f"{WORK}/ft_weights"
DATA_CSV = f"{DATA_DIR}/data.csv"

DB_HOST = _env("DATABRICKS_HOST")
DB_TOKEN = _env("DATABRICKS_TOKEN")


def run(cmd: str) -> int:
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"[exit {rc} in {time.time()-t0:.0f}s]", flush=True)
    return rc


def _mlflow():
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        return mlflow
    except Exception as e:
        print(f"[mlflow] unavailable: {e}", flush=True)
        return None


def weights_location_for_ft(ft_id: str) -> str:
    """Look up weights_volume_location from bionemo_weights via the SQL warehouse."""
    if not (SQL_WAREHOUSE_ID and DB_HOST and DB_TOKEN):
        raise RuntimeError("cannot query bionemo_weights: DATABRICKS_HOST/TOKEN/sql_warehouse_id "
                           "not available in-container")
    from databricks import sql
    host = DB_HOST.replace("https://", "").rstrip("/")
    with sql.connect(server_hostname=host,
                     http_path=f"/sql/1.0/warehouses/{SQL_WAREHOUSE_ID}",
                     access_token=DB_TOKEN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT weights_volume_location FROM {CATALOG}.{SCHEMA}.bionemo_weights "
                        f"WHERE ft_id = {int(ft_id)}")
            row = cur.fetchone()
    if not row:
        raise RuntimeError(f"finetune run {ft_id} not found in bionemo_weights")
    return row[0]


def resolve_weights() -> str:
    for d in (DATA_DIR, RESULTS_DIR, FT_WEIGHTS_DIR):
        os.makedirs(d, exist_ok=True)
    if IS_BASE_MODEL:
        from bionemo.core.data.load import load
        p = str(load(f"esm2/{ESM_VARIANT.lower()}:2.0"))
        print(f"[weights] base model: {p}", flush=True)
        return p
    vol = weights_location_for_ft(FINETUNE_RUN_ID)
    if os.path.isdir(FT_WEIGHTS_DIR):
        shutil.rmtree(FT_WEIGHTS_DIR)
    shutil.copytree(vol, FT_WEIGHTS_DIR)
    print(f"[weights] copied fine-tuned weights {vol} -> {FT_WEIGHTS_DIR}", flush=True)
    return FT_WEIGHTS_DIR


def main():
    import pandas as pd
    run("nvidia-smi -L")

    mlflow = _mlflow()
    if mlflow:
        try:
            if MLFLOW_RUN_ID:
                mlflow.start_run(run_id=MLFLOW_RUN_ID)
            else:
                mlflow.set_experiment(f"/Users/{USER_EMAIL}/mlflow_experiments/{EXPERIMENT_NAME}")
                mlflow.set_experiment_tag("used_by_genesis_workbench", "yes")
                mlflow.start_run(run_name=RUN_NAME)
            for k, v in {"origin": "genesis_workbench", "feature": "bionemo_esm_inference",
                         "created_by": USER_EMAIL, "result_location": RESULT_LOCATION,
                         "job_status": "running"}.items():
                mlflow.set_tag(k, v)
            mlflow.log_param("esm_variant", ESM_VARIANT)
            mlflow.log_param("is_base_model", str(IS_BASE_MODEL))
        except Exception as e:
            print(f"[mlflow] setup failed: {e}", flush=True)
            mlflow = None

    weights = resolve_weights()

    pd.read_csv(DATA_LOCATION)[[SEQUENCE_COLUMN]].rename(
        columns={SEQUENCE_COLUMN: "sequences"}).to_csv(DATA_CSV, index=False)

    rc = run(f"infer_esm2 --checkpoint-path {weights} --config-class ESM2FineTuneSeqConfig "
             f"--data-path {DATA_CSV} --results-path {RESULTS_DIR} --micro-batch-size 3 "
             f"--num-gpus 1 --precision bf16-mixed --include-embeddings --include-input-ids")
    if rc != 0:
        if mlflow:
            try:
                mlflow.set_tag("job_status", "failed"); mlflow.end_run(status="FAILED")
            except Exception:
                pass
        sys.exit(rc)

    import torch
    results = torch.load(f"{RESULTS_DIR}/predictions__rank_0.pt")
    results_df = pd.read_csv(DATA_CSV)
    if results.get("classification_output") is not None:
        results_df["predictions"] = [r.argmax().item() for r in results["classification_output"].tolist()]
    elif results.get("regression_output") is not None:
        results_df["predictions"] = [r[0] for r in results["regression_output"].tolist()]
    else:
        keys = [k for k, v in results.items() if v is not None]
        sys.exit(f"no regression_output/classification_output in results; keys: {keys}")

    os.makedirs(RESULT_LOCATION, exist_ok=True)
    results_file = f"{RESULT_LOCATION}/results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"[results] wrote {results_file} ({len(results_df)} rows)", flush=True)

    if mlflow:
        try:
            mlflow.log_metric("num_sequences", int(len(results_df)))
            mlflow.set_tag("results_file", results_file)
            mlflow.set_tag("job_status", "complete")
            mlflow.end_run()
        except Exception:
            pass
    print(f"\n=== DONE === results: {results_file}", flush=True)


if __name__ == "__main__":
    main()
