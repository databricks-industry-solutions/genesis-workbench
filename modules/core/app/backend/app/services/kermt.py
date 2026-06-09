"""KERMT fine-tune + deploy — dispatch + read helpers.

Dispatches the `kermt_finetune_job` orchestrator (GROVERbase → fine-tune → write
kermt_weights), pre-creating the MLflow run so Search Past Runs shows it in-flight;
the orchestrator advances job_status on that run. Also dispatches the
`kermt_deploy_job` (register a chosen fine-tuned model as a serving endpoint) and
lists deployable fine-tuned weights from the `kermt_weights` table.

Mirrors molecule_optimization.py (resolve-job-by-name + pre-create run + run_now).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import execute_select_query
from mlflow.tracking import MlflowClient

from app.services.databricks_links import job_run_url, mlflow_run_url

logger = logging.getLogger(__name__)

# Emoji block progress bar — same style as the other Search Past Runs tables.
_PROGRESS_MAP = {
    "submitted": "🟩⬜⬜⬜",
    "started": "🟩🟩⬜⬜",
    "training": "🟩🟩🟩⬜",
    "complete": "🟩🟩🟩🟩",
    "failed": "🟥",
    "unknown": "⬜⬜⬜⬜",
}


def _progress(status: str) -> str:
    return _PROGRESS_MAP.get(status or "", _PROGRESS_MAP["unknown"])


FINETUNE_JOB_NAME = "kermt_finetune_job"
DEPLOY_JOB_NAME = "kermt_deploy_job"
_job_id_cache: dict[str, int] = {}


def _use_databricks_tracking() -> None:
    """Pin MLflow to the Databricks tracking + UC registry stores (see the same
    helper in molecule_optimization — without re-pinning, search intermittently
    hits the wrong store and runs vanish from Search Past Runs)."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")


def _resolve_job_id(job_name: str, env_var: str, w: Optional[WorkspaceClient] = None) -> int:
    cached = _job_id_cache.get(job_name)
    if cached is not None:
        return cached
    env_id = os.environ.get(env_var)
    if env_id:
        _job_id_cache[job_name] = int(env_id)
        return int(env_id)
    workspace = w or WorkspaceClient()
    matches = list(workspace.jobs.list(name=job_name))
    if not matches:
        raise RuntimeError(
            f"Job '{job_name}' not found. Deploy the kermt submodule: "
            "`./deploy.sh small_molecule aws --only-submodule kermt/kermt_v1`"
        )
    _job_id_cache[job_name] = int(matches[0].job_id)
    return _job_id_cache[job_name]


def start_kermt_finetune(
    *,
    user_email: str,
    mlflow_experiment: str,
    mlflow_run_name: str,
    finetune_label: str,
    train_data_location: str,
    validation_data_location: str,
    test_data_location: str,
    target_names: str,
    dataset_type: str,
    epochs: int,
    batch_size: int,
    ffn_hidden_size: int,
) -> dict:
    _use_databricks_tracking()
    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment, user_email=user_email, host=None, token=None
    )
    w = WorkspaceClient()
    job_id = _resolve_job_id(FINETUNE_JOB_NAME, "KERMT_FINETUNE_JOB_ID", w)

    with mlflow.start_run(
        run_name=mlflow_run_name, experiment_id=experiment.experiment_id
    ) as pre:
        run_id = pre.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "kermt_finetune")
        mlflow.set_tag("created_by", user_email)
        mlflow.set_tag("job_status", "submitted")
        mlflow.log_params({
            "finetune_label": finetune_label,
            "dataset_type": dataset_type,
            "target_names": target_names,
            "epochs": epochs,
            "batch_size": batch_size,
            "ffn_hidden_size": ffn_hidden_size,
        })
        try:
            job_run = w.jobs.run_now(
                job_id=job_id,
                job_parameters={
                    "catalog": os.environ["CORE_CATALOG_NAME"],
                    "schema": os.environ["CORE_SCHEMA_NAME"],
                    "user_email": user_email,
                    "train_data_location": train_data_location,
                    "validation_data_location": validation_data_location,
                    "test_data_location": test_data_location,
                    "target_names": target_names,
                    "dataset_type": dataset_type,
                    "finetune_label": finetune_label,
                    "epochs": str(epochs),
                    "batch_size": str(batch_size),
                    "ffn_hidden_size": str(ffn_hidden_size),
                    "experiment_name": mlflow_experiment,
                    "mlflow_run_name": mlflow_run_name,
                    "mlflow_run_id": run_id,
                },
            )
        except Exception as e:
            mlflow.set_tag("job_status", "failed")
            mlflow.set_tag("error", str(e)[:500])
            raise
        mlflow.set_tag("job_run_id", str(job_run.run_id))

    return {
        "job_id": job_id,
        "job_run_id": int(job_run.run_id),
        "mlflow_run_id": run_id,
        "experiment_id": str(experiment.experiment_id),
        "run_url": job_run_url(job_id, job_run.run_id),
    }


def start_kermt_deploy(*, user_email: str, ft_id: str, model_name: str = "kermt_admet",
                       workload_type: str = "") -> dict:
    """Dispatch the deploy job for a chosen fine-tuned model (one-off action —
    registers the PyFunc + (re)deploys the serving endpoint)."""
    w = WorkspaceClient()
    job_id = _resolve_job_id(DEPLOY_JOB_NAME, "KERMT_DEPLOY_JOB_ID", w)
    params = {
        "catalog": os.environ["CORE_CATALOG_NAME"],
        "schema": os.environ["CORE_SCHEMA_NAME"],
        "user_email": user_email,
        "ft_id": str(ft_id),
        "model_name": model_name,
    }
    if workload_type:
        params["workload_type"] = workload_type
    job_run = w.jobs.run_now(job_id=job_id, job_parameters=params)
    return {
        "job_id": job_id,
        "job_run_id": int(job_run.run_id),
        "run_url": job_run_url(job_id, job_run.run_id),
    }


def list_weights() -> list[dict]:
    """Active fine-tuned KERMT models from kermt_weights (for the deploy selector).
    ft_id is a BIGINT — return it as a string so the React UI doesn't round it."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    try:
        df = execute_select_query(
            f"SELECT ft_id, ft_label, dataset_type, task_names, run_id, created_datetime "
            f"FROM {catalog}.{schema}.kermt_weights WHERE is_active = true "
            f"ORDER BY created_datetime DESC"
        )
    except Exception as e:
        logger.warning("kermt_weights query failed: %s", e)
        return []
    out = []
    for _, r in df.iterrows():
        out.append({
            "ft_id": str(r["ft_id"]),
            "ft_label": str(r["ft_label"]),
            "dataset_type": str(r["dataset_type"]),
            "task_names": str(r["task_names"]),
            "run_id": str(r["run_id"]) if pd.notna(r.get("run_id")) else "",
            "created_datetime": str(r["created_datetime"]) if pd.notna(r.get("created_datetime")) else "",
        })
    return out


def get_run_status(run_id: str) -> dict[str, Any]:
    _use_databricks_tracking()
    client = MlflowClient()
    run = client.get_run(run_id)
    return {
        "status": run.info.status,
        "job_status": run.data.tags.get("job_status", ""),
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "ft_id": run.data.tags.get("ft_id", ""),
        "weights_volume_location": run.data.tags.get("weights_volume_location", ""),
        "experiment_id": run.info.experiment_id,
        "params": {k: str(v) for k, v in run.data.params.items()},
        "metrics": {k: float(v) for k, v in run.data.metrics.items()},
    }


def _experiment_map() -> dict[str, str]:
    _use_databricks_tracking()
    experiments = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench = 'yes'"
    )
    return {e.experiment_id: e.name for e in experiments}


def search_runs(user_email: str, by: str, text: str) -> list[dict]:
    exp_map = _experiment_map()
    if not exp_map:
        return []
    if by == "experiment_name":
        needle = text.upper()
        exp_map = {eid: n for eid, n in exp_map.items() if needle in n.upper()}
        if not exp_map:
            return []
    runs = mlflow.search_runs(
        filter_string=(
            "tags.feature='kermt_finetune' AND "
            f"tags.created_by='{user_email}' AND tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return []
    if by == "run_name":
        runs = runs[
            runs["tags.mlflow.runName"].astype(str).str.contains(text, case=False, na=False)
        ]
        if runs.empty:
            return []

    def _g(r, col):
        return r[col] if col in r and pd.notna(r[col]) else None

    out = []
    for _, r in runs.iterrows():
        status = str(_g(r, "tags.job_status") or "")
        lifecycle = str(_g(r, "status") or "")
        if lifecycle in ("FAILED", "KILLED") and status not in ("complete", "failed"):
            status = "failed"
        exp_id = str(r["experiment_id"])
        label = _g(r, "params.finetune_label") or ""
        dtype = _g(r, "params.dataset_type") or ""
        out.append({
            "run_id": str(r["run_id"]),
            "run_name": str(_g(r, "tags.mlflow.runName") or ""),
            "experiment_name": exp_map.get(exp_id, ""),
            "status": status,
            "progress": _progress(status),
            "detail": f"{label} ({dtype})" if label else dtype,
            "start_time_ms": (int(r["start_time"].timestamp() * 1000) if "start_time" in r and pd.notna(r["start_time"]) else None),
            "run_url": mlflow_run_url(exp_id, str(r["run_id"])),
        })
    return out
