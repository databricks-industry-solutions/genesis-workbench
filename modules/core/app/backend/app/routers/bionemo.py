"""NVIDIA BioNeMo — ESM2 fine-tuning + inference.

React port of the old Streamlit `views/nvidia/bionemo_esm.py`. Thin FastAPI
layer over the `genesis_workbench.bionemo` lib: both fine-tune and inference
are long-running Databricks jobs, so each endpoint dispatches a job run and
returns a clickable run URL (same shape as the other batch workflows).

The orchestrator job ids + the app SP's CAN_MANAGE_RUN grant are provisioned by
the bionemo module deploy (`modules/bionemo/notebooks/initialize.py`), which
writes `bionemo_esm_finetune_job_id` / `bionemo_esm_inference_job_id` into the
`settings` table (module='bionemo').
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from genesis_workbench.bionemo import (
    BionemoModelType,
    get_variants,
    list_finetuned_weights,
    start_esm2_finetuning,
    start_esm2_inference,
)
from genesis_workbench.workbench import UserInfo
from pydantic import BaseModel, Field

from app.auth import CurrentUser, CurrentUserDep
from app.services.databricks_links import job_run_url
from app.services.workbench import get_job_id

router = APIRouter(prefix="/api/bionemo", tags=["bionemo"])

_FINETUNE_JOB_SETTING = "bionemo_esm_finetune_job_id"
_INFERENCE_JOB_SETTING = "bionemo_esm_inference_job_id"


def _build_user_info(user: CurrentUser) -> UserInfo:
    return UserInfo(
        user_email=user.email or "",
        user_name=user.preferred_username or "",
        user_id=user.user_id or "",
        user_groups=[],
        user_access_token=user.access_token,
        user_display_name=user.preferred_username or "",
    )


# ─── Variants + weights ──────────────────────────────────────────────────────


class VariantsResponse(BaseModel):
    esm2: list[str]


@router.get("/variants", response_model=VariantsResponse)
def variants(_: CurrentUserDep) -> VariantsResponse:
    return VariantsResponse(esm2=get_variants(BionemoModelType.ESM2))


class FinetunedWeight(BaseModel):
    ft_id: int
    ft_label: str
    variant: str
    model_type: str
    experiment_name: str | None = None
    run_id: str | None = None
    created_by: str | None = None
    created_datetime: str | None = None


class WeightsResponse(BaseModel):
    weights: list[FinetunedWeight]


@router.get("/weights", response_model=WeightsResponse)
def weights(_: CurrentUserDep) -> WeightsResponse:
    """Fine-tuned ESM2 weights available for inference."""
    df = list_finetuned_weights(model_type=BionemoModelType.ESM2)
    out: list[FinetunedWeight] = []
    for _, r in df.iterrows():
        out.append(
            FinetunedWeight(
                ft_id=int(r["ft_id"]),
                ft_label=str(r["ft_label"]),
                variant=str(r["variant"]),
                model_type=str(r["model_type"]),
                experiment_name=_s(r.get("experiment_name")),
                run_id=_s(r.get("run_id")),
                created_by=_s(r.get("created_by")),
                created_datetime=_s(r.get("created_datetime")),
            )
        )
    return WeightsResponse(weights=out)


def _s(v) -> str | None:
    if v is None:
        return None
    s = str(v)
    return None if s in ("None", "nan", "") else s


# ─── Dispatch ────────────────────────────────────────────────────────────────


class DispatchResponse(BaseModel):
    job_run_id: int
    run_url: str


class FinetuneRequest(BaseModel):
    esm_variant: str
    train_data: str = Field(..., min_length=1)
    evaluation_data: str = Field(..., min_length=1)
    finetune_label: str = Field(..., min_length=1)
    experiment_name: str = Field(..., min_length=1)
    should_use_lora: bool = False
    task_type: str = "regression"
    num_steps: int = 50
    micro_batch_size: int = 2
    precision: str = "bf16-mixed"
    # Advanced
    mlp_ft_dropout: float = 0.25
    mlp_hidden_size: int = 256
    mlp_target_size: int = 1
    mlp_lr: float = 5e-3
    mlp_lr_multiplier: float = 1e2


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
    _require_volume_csv(payload.train_data, "Train data")
    _require_volume_csv(payload.evaluation_data, "Evaluation data")
    try:
        run_id = start_esm2_finetuning(
            user_info=_build_user_info(user),
            esm_variant=payload.esm_variant,
            train_data_volume_location=payload.train_data.strip(),
            validation_data_volume_location=payload.evaluation_data.strip(),
            should_use_lora=payload.should_use_lora,
            finetune_label=payload.finetune_label.strip(),
            experiment_name=payload.experiment_name.strip(),
            task_type=payload.task_type,
            num_steps=int(payload.num_steps),
            micro_batch_size=int(payload.micro_batch_size),
            precision=payload.precision,
            mlp_ft_dropout=payload.mlp_ft_dropout,
            mlp_hidden_size=int(payload.mlp_hidden_size),
            mlp_target_size=int(payload.mlp_target_size),
            mlp_lr=payload.mlp_lr,
            mlp_lr_multiplier=payload.mlp_lr_multiplier,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to launch fine-tuning job: {e}")
    return DispatchResponse(
        job_run_id=int(run_id),
        run_url=job_run_url(get_job_id(_FINETUNE_JOB_SETTING), run_id),
    )


class InferenceRequest(BaseModel):
    esm_variant: str
    is_base_model: bool = True
    finetune_run_id: int = 0  # ft_id of a fine-tuned weight; ignored for base model
    task_type: str = "regression"
    data_location: str = Field(..., min_length=1)
    sequence_column_name: str = Field(..., min_length=1)
    result_location: str = Field(..., min_length=1)


@router.post("/inference", response_model=DispatchResponse)
def inference(payload: InferenceRequest, user: CurrentUserDep) -> DispatchResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    _require_volume_csv(payload.data_location, "Data")
    if not payload.result_location.strip().startswith("/Volumes"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Result location must be a UC Volume folder (/Volumes/...).",
        )
    if not payload.is_base_model and not payload.finetune_run_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Select a fine-tuned weight, or use the base model.",
        )
    try:
        run_id = start_esm2_inference(
            user_info=_build_user_info(user),
            esm_variant=payload.esm_variant,
            is_base_model=payload.is_base_model,
            finetune_run_id=int(payload.finetune_run_id),
            task_type=payload.task_type,
            data_volume_location=payload.data_location.strip(),
            sequence_column_name=payload.sequence_column_name.strip(),
            result_location=payload.result_location.strip(),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to launch inference job: {e}")
    return DispatchResponse(
        job_run_id=int(run_id),
        run_url=job_run_url(get_job_id(_INFERENCE_JOB_SETTING), run_id),
    )
