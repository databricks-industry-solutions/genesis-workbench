"""NVIDIA BioNeMo — fine-tune dispatch + MLflow run search/details.

Mirrors the genomics/enzyme batch-workflow arch so the Fine Tune tab gets the
same "Search Past Runs + View results" experience:

- start_finetune pre-creates the MLflow run (status "submitted") and dispatches
  the job with that run id; the notebook then advances job_status
  (training → complete) and writes a `result_location` tag.
- search_finetune_runs queries MLflow runs by the standard feature/created_by/
  origin tags (like genomics `_search`).
- get_finetune_run_details returns metrics + result location for the dialog.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from genesis_workbench.bionemo import start_esm2_finetuning
from genesis_workbench.models import set_mlflow_experiment
from genesis_workbench.workbench import UserInfo

from app.services.databricks_links import job_run_url
from app.services.workbench import get_job_id

logger = logging.getLogger(__name__)

FEATURE_TAG = "bionemo_esm_finetune"
FINETUNE_JOB_SETTING = "bionemo_esm_finetune_job_id"

_IN_PROGRESS = {"submitted", "started", "training"}
_PROGRESS_MAP = {
    "submitted": "🟩⬜⬜⬜",
    "started": "🟩🟩⬜⬜",
    "training": "🟩🟩🟩⬜",
    "complete": "🟩🟩🟩🟩",
    "failed": "🟥",
}


def _progress(status: str) -> str:
    return _PROGRESS_MAP.get(status, "⬜⬜⬜⬜")


def _start_time_ms(v) -> int | None:
    try:
        if hasattr(v, "value"):  # pandas Timestamp
            return int(v.value // 1_000_000)
        return int(v)
    except (ValueError, TypeError, AttributeError):
        return None


# ─── Dispatch (pre-create MLflow run, then run the job) ──────────────────────


@dataclass(frozen=True)
class FinetuneDispatch:
    job_run_id: int
    mlflow_run_id: str
    run_url: str


def start_finetune(
    *,
    user_info: UserInfo,
    esm_variant: str,
    train_data: str,
    evaluation_data: str,
    should_use_lora: bool,
    finetune_label: str,
    experiment_name: str,
    task_type: str,
    num_steps: int,
    micro_batch_size: int,
    precision: str,
    mlp_ft_dropout: float,
    mlp_hidden_size: int,
    mlp_target_size: int,
    mlp_lr: float,
    mlp_lr_multiplier: float,
) -> FinetuneDispatch:
    experiment = set_mlflow_experiment(
        experiment_tag=experiment_name, user_email=user_info.user_email
    )
    with mlflow.start_run(
        run_name=finetune_label, experiment_id=experiment.experiment_id
    ) as pre_run:
        mlflow_run_id = pre_run.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", FEATURE_TAG)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_status", "submitted")
        mlflow.log_param("esm_variant", esm_variant)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("num_steps", num_steps)
        try:
            run_id = start_esm2_finetuning(
                user_info=user_info,
                esm_variant=esm_variant,
                train_data_volume_location=train_data,
                validation_data_volume_location=evaluation_data,
                should_use_lora=should_use_lora,
                finetune_label=finetune_label,
                experiment_name=experiment_name,
                task_type=task_type,
                num_steps=int(num_steps),
                micro_batch_size=int(micro_batch_size),
                precision=precision,
                mlp_ft_dropout=mlp_ft_dropout,
                mlp_hidden_size=int(mlp_hidden_size),
                mlp_target_size=int(mlp_target_size),
                mlp_lr=mlp_lr,
                mlp_lr_multiplier=mlp_lr_multiplier,
                mlflow_run_id=mlflow_run_id,
            )
        except Exception as e:
            mlflow.set_tag("job_status", "failed")
            mlflow.set_tag("error", str(e)[:500])
            raise
        mlflow.set_tag("job_run_id", str(run_id))

    return FinetuneDispatch(
        job_run_id=int(run_id),
        mlflow_run_id=mlflow_run_id,
        run_url=job_run_url(get_job_id(FINETUNE_JOB_SETTING), run_id),
    )


# ─── Search past runs ────────────────────────────────────────────────────────


def _experiment_map() -> dict[str, str]:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    exps = mlflow.search_experiments(filter_string="tags.used_by_genesis_workbench='yes'")
    return {e.experiment_id: e.name.split("/")[-1] for e in exps}


def search_finetune_runs(user_email: str, by: str, text: str) -> list[dict]:
    """`by` is 'run_name' or 'experiment_name'; case-insensitive contains."""
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
            f"tags.feature='{FEATURE_TAG}' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
        order_by=["start_time DESC"],
    )
    if runs.empty:
        return []
    if by == "run_name":
        runs = runs[
            runs["tags.mlflow.runName"].astype(str).str.contains(text, case=False, na=False)
        ]
        if runs.empty:
            return []

    runs["experiment_name"] = runs["experiment_id"].map(exp_map)
    job_id = get_job_id(FINETUNE_JOB_SETTING)
    out: list[dict] = []
    for _, r in runs.iterrows():
        status = str(r.get("tags.job_status", "") or "")
        job_run_id = str(r.get("tags.job_run_id", "") or "")
        out.append(
            {
                "run_id": str(r.get("run_id", "")),
                "run_name": str(r.get("tags.mlflow.runName", "") or ""),
                "experiment_name": str(r.get("experiment_name", "") or ""),
                "status": status,
                "progress": _progress(status),
                "start_time_ms": _start_time_ms(r.get("start_time")),
                # surfaced as the table's detail column
                "detail": str(r.get("params.esm_variant", "") or ""),
                "run_url": job_run_url(job_id, job_run_id) if job_id else "",
            }
        )
    return out


def get_finetune_run_details(run_id: str) -> dict:
    """Metrics + result location + params/status for the View dialog."""
    run = MlflowClient().get_run(run_id)
    tags = run.data.tags
    return {
        "run_name": tags.get("mlflow.runName", ""),
        "status": run.info.status,
        "job_status": tags.get("job_status", ""),
        "result_location": tags.get("result_location", ""),
        "job_run_id": tags.get("job_run_id", ""),
        "params": {k: str(v) for k, v in run.data.params.items()},
        "metrics": {k: float(v) for k, v in run.data.metrics.items()},
    }
