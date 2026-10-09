"""RFD4-Proteina — fine-tune + deploy a protein-design model.

Thin FastAPI layer over `app.services.rfd4`. Fine-tune is a long-running Databricks
job (batch-workflow pattern: dispatch + Search Past Runs); deploy is a one-off job
that Express-registers a chosen fine-tuned adapter as the serving endpoint. The
orchestrator job ids + the app SP's CAN_MANAGE_RUN grant are provisioned by the
rfd4_proteina module deploy (`register_rfd4_jobs.py`).
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.services import rfd4 as svc

router = APIRouter(prefix="/api/rfd4", tags=["rfd4"])


# ─── Defaults + deployable weights ───────────────────────────────────────────


class DefaultsResponse(BaseModel):
    pretrain_ckpt: str
    train_data: str
    experiment_preset: str


# The pinned RFD4 flow (denoiser) checkpoint the rfd4_proteina deploy stages
# (matches variables.yml `rfd4_flow_ckpt`).
_RFD4_FLOW_CKPT = "chk_epoch_00000169_step_000000340000-ema.ckpt"


@router.get("/defaults", response_model=DefaultsResponse)
def defaults(_: CurrentUserDep) -> DefaultsResponse:
    """Pre-fill the fine-tune form so it runs out of the box: the staged RFD4 flow
    checkpoint as the base, and the repo's tutorial finetune preset. Training data
    is left BLANK on purpose — blank uses rfd4-train's bundled tutorial dataset
    (the `tutorial/finetune_test_dataset` preset); pointing at a non-existent
    ft_data dir would fail the run. A user supplies their own dir to override."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    return DefaultsResponse(
        pretrain_ckpt=f"/Volumes/{catalog}/{schema}/rfd4_proteina/flow_checkpoints/{_RFD4_FLOW_CKPT}",
        train_data="",
        experiment_preset="tutorial/finetune_test_dataset",
    )


class FinetunedWeight(BaseModel):
    # ft_id is a time_ns() BIGINT — string so the browser never rounds it.
    ft_id: str
    ft_label: str
    model_type: str
    experiment_name: str | None = None
    run_id: str | None = None
    created_datetime: str | None = None


class WeightsResponse(BaseModel):
    weights: list[FinetunedWeight]


@router.get("/weights", response_model=WeightsResponse)
def weights(_: CurrentUserDep) -> WeightsResponse:
    """Fine-tuned RFD4 adapters available to deploy."""
    return WeightsResponse(weights=[FinetunedWeight(**w) for w in svc.list_weights()])


# ─── Dispatch ────────────────────────────────────────────────────────────────


class DispatchResponse(BaseModel):
    job_run_id: int
    run_url: str


class FinetuneRequest(BaseModel):
    finetune_label: str = Field(..., min_length=1)
    pretrain_ckpt: str = ""
    train_data: str = ""
    experiment_preset: str = Field("tutorial/finetune_test_dataset", min_length=1)
    max_epochs: int = 5
    steps_per_epoch: int = 100
    experiment_name: str = Field("gwb_rfd4_finetune", min_length=1)
    run_name: str = Field(..., min_length=1)


def _require_volume(path: str, what: str) -> None:
    p = path.strip()
    if p and not p.startswith("/Volumes"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{what} must be a path under a UC Volume (/Volumes/...).",
        )


@router.post("/finetune", response_model=DispatchResponse)
def finetune(payload: FinetuneRequest, user: CurrentUserDep) -> DispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    _require_volume(payload.pretrain_ckpt, "Base checkpoint")
    _require_volume(payload.train_data, "Training data")
    if int(payload.max_epochs) < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "max_epochs must be >= 1")
    if int(payload.steps_per_epoch) < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "steps_per_epoch must be >= 1")
    try:
        result = svc.start_rfd4_finetune(
            user_email=user.email,
            mlflow_experiment=payload.experiment_name.strip(),
            mlflow_run_name=payload.run_name.strip(),
            finetune_label=payload.finetune_label.strip(),
            experiment_preset=payload.experiment_preset.strip(),
            max_epochs=int(payload.max_epochs),
            steps_per_epoch=int(payload.steps_per_epoch),
            pretrain_ckpt_path=payload.pretrain_ckpt.strip(),
            train_data_location=payload.train_data.strip(),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to launch RFD4 fine-tuning: {e}")
    return DispatchResponse(job_run_id=result["job_run_id"], run_url=result["run_url"])


class DeployRequest(BaseModel):
    ft_id: str = Field(..., min_length=1)
    model_name: str = "rfd4_proteina"
    workload_type: str = ""


@router.post("/deploy", response_model=DispatchResponse)
def deploy(payload: DeployRequest, user: CurrentUserDep) -> DispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    if payload.ft_id.strip() in ("", "0"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Select a fine-tuned model to deploy.")
    try:
        result = svc.start_rfd4_deploy(
            user_email=user.email,
            ft_id=payload.ft_id.strip(),
            model_name=payload.model_name.strip() or "rfd4_proteina",
            workload_type=payload.workload_type.strip(),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to launch RFD4 deploy: {e}")
    return DispatchResponse(job_run_id=result["job_run_id"], run_url=result["run_url"])


# ─── Search past fine-tune runs ──────────────────────────────────────────────


class DBRunRow(BaseModel):
    run_id: str
    run_name: str
    experiment_name: str
    status: str
    progress: str
    start_time_ms: int | None = None
    detail: str
    run_url: str = ""


class DBSearchResponse(BaseModel):
    runs: list[DBRunRow]


class FinetuneRunDetails(BaseModel):
    run_name: str
    status: str
    job_status: str
    ft_id: str
    adapter_location: str
    experiment_id: str
    params: dict[str, str]
    metrics: dict[str, float]


@router.get("/finetune/search", response_model=DBSearchResponse)
def finetune_search(user: CurrentUserDep, by: str = "run_name", text: str = "") -> DBSearchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    rows = svc.search_runs(user.email, by, text.strip())
    return DBSearchResponse(runs=[DBRunRow(**r) for r in rows])


@router.get("/finetune/run-details", response_model=FinetuneRunDetails)
def finetune_run_details(run_id: str, _: CurrentUserDep) -> FinetuneRunDetails:
    try:
        return FinetuneRunDetails(**svc.get_run_status(run_id))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not load run {run_id}: {e}")
