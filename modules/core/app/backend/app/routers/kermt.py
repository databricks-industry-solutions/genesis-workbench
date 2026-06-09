"""KERMT — fine-tune + deploy a small-molecule ADMET model.

Thin FastAPI layer over `app.services.kermt`. Fine-tune is a long-running
Databricks job (batch-workflow pattern: dispatch + Search Past Runs); deploy is
a one-off job that registers a chosen fine-tuned checkpoint as a serving endpoint.
The orchestrator job ids + the app SP's CAN_MANAGE_RUN grant are provisioned by
the kermt module deploy (`register_kermt_jobs.py`).
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.services import kermt as svc

router = APIRouter(prefix="/api/kermt", tags=["kermt"])


# ─── Defaults + deployable weights ───────────────────────────────────────────


class DefaultsResponse(BaseModel):
    train_data: str
    validation_data: str
    test_data: str
    target_names: str
    dataset_type: str


@router.get("/defaults", response_model=DefaultsResponse)
def defaults(_: CurrentUserDep) -> DefaultsResponse:
    """Pre-fill the TDC ClinTox sample the kermt module stages, so the form
    works out of the box."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    ft = f"/Volumes/{catalog}/{schema}/kermt/ft_data"
    return DefaultsResponse(
        train_data=f"{ft}/clintox_train.csv",
        validation_data=f"{ft}/clintox_val.csv",
        test_data=f"{ft}/clintox_test.csv",
        target_names="toxicity",
        dataset_type="classification",
    )


class FinetunedWeight(BaseModel):
    # ft_id is a time_ns() BIGINT — string so the browser never rounds it.
    ft_id: str
    ft_label: str
    dataset_type: str
    task_names: str
    run_id: str | None = None
    created_datetime: str | None = None


class WeightsResponse(BaseModel):
    weights: list[FinetunedWeight]


@router.get("/weights", response_model=WeightsResponse)
def weights(_: CurrentUserDep) -> WeightsResponse:
    """Fine-tuned KERMT models available to deploy."""
    return WeightsResponse(weights=[FinetunedWeight(**w) for w in svc.list_weights()])


# ─── Dispatch ────────────────────────────────────────────────────────────────


class DispatchResponse(BaseModel):
    job_run_id: int
    run_url: str


class FinetuneRequest(BaseModel):
    finetune_label: str = Field(..., min_length=1)
    train_data: str = Field(..., min_length=1)
    validation_data: str = Field(..., min_length=1)
    test_data: str = Field(..., min_length=1)
    target_names: str = Field("toxicity", min_length=1)
    dataset_type: str = "classification"
    epochs: int = 20
    batch_size: int = 16
    ffn_hidden_size: int = 700
    experiment_name: str = Field("gwb_kermt_finetune", min_length=1)
    run_name: str = Field(..., min_length=1)


def _require_volume_csv(path: str, what: str) -> None:
    p = path.strip()
    if not (p.startswith("/Volumes") and p.endswith(".csv")):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{what} must be a .csv file under a UC Volume (/Volumes/...).",
        )


@router.post("/finetune", response_model=DispatchResponse)
def finetune(payload: FinetuneRequest, user: CurrentUserDep) -> DispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    for p, what in ((payload.train_data, "Train data"),
                    (payload.validation_data, "Validation data"),
                    (payload.test_data, "Test data")):
        _require_volume_csv(p, what)
    if payload.dataset_type not in ("classification", "regression"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "dataset_type must be classification or regression")
    try:
        result = svc.start_kermt_finetune(
            user_email=user.email,
            mlflow_experiment=payload.experiment_name.strip(),
            mlflow_run_name=payload.run_name.strip(),
            finetune_label=payload.finetune_label.strip(),
            train_data_location=payload.train_data.strip(),
            validation_data_location=payload.validation_data.strip(),
            test_data_location=payload.test_data.strip(),
            target_names=payload.target_names.strip(),
            dataset_type=payload.dataset_type,
            epochs=int(payload.epochs),
            batch_size=int(payload.batch_size),
            ffn_hidden_size=int(payload.ffn_hidden_size),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to launch KERMT fine-tuning: {e}")
    return DispatchResponse(job_run_id=result["job_run_id"], run_url=result["run_url"])


class DeployRequest(BaseModel):
    ft_id: str = Field(..., min_length=1)
    model_name: str = "kermt_admet"
    workload_type: str = ""


@router.post("/deploy", response_model=DispatchResponse)
def deploy(payload: DeployRequest, user: CurrentUserDep) -> DispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    if payload.ft_id.strip() in ("", "0"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Select a fine-tuned model to deploy.")
    try:
        result = svc.start_kermt_deploy(
            user_email=user.email,
            ft_id=payload.ft_id.strip(),
            model_name=payload.model_name.strip() or "kermt_admet",
            workload_type=payload.workload_type.strip(),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to launch KERMT deploy: {e}")
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
    weights_volume_location: str
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
