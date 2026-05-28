from fastapi import APIRouter, HTTPException, status
from genesis_workbench.models import (
    ModelCategory,
    get_available_models,
    get_batch_models,
    get_deployed_models,
)
from pydantic import BaseModel

from app.auth import CurrentUserDep

router = APIRouter(prefix="/api/models", tags=["models"])


# `module` is the *bundle* module string (e.g. "single_cell"). It's a slight
# rename from ModelCategory because Disease Biology only has batch models —
# they live under module="disease_biology" but no ModelCategory.DISEASE_BIOLOGY
# row is created for them.
SUPPORTED_MODULES = {"single_cell", "protein_studies", "small_molecule", "disease_biology"}


def _module_to_category(module: str) -> ModelCategory:
    try:
        return ModelCategory(module)
    except ValueError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown module '{module}'. Expected one of: {sorted(SUPPORTED_MODULES)}",
        )


class AvailableModel(BaseModel):
    model_id: int
    model_name: str
    model_display_name: str
    model_source_version: str | None = None
    model_uc_name: str
    model_uc_version: int


class AvailableModelsResponse(BaseModel):
    models: list[AvailableModel]


class DeployedModel(BaseModel):
    model_id: int
    deployment_id: int
    deployment_name: str
    deployment_description: str | None = None
    model_display_name: str
    model_source_version: str | None = None
    uc_name: str
    model_endpoint_name: str


class DeployedModelsResponse(BaseModel):
    models: list[DeployedModel]


class BatchModel(BaseModel):
    model_display_name: str
    model_description: str | None = None
    job_name: str
    cluster_type: str | None = None


class BatchModelsResponse(BaseModel):
    models: list[BatchModel]


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    s = str(value)
    return None if s == "None" or s.strip() == "" else s


@router.get("/available", response_model=AvailableModelsResponse)
def available(module: str, _: CurrentUserDep) -> AvailableModelsResponse:
    category = _module_to_category(module)
    df = get_available_models(category)
    return AvailableModelsResponse(
        models=[
            AvailableModel(
                model_id=int(r["model_id"]),
                model_name=str(r["model_name"]),
                model_display_name=str(r["model_display_name"]),
                model_source_version=_str_or_none(r.get("model_source_version")),
                model_uc_name=str(r["model_uc_name"]),
                model_uc_version=int(r["model_uc_version"]),
            )
            for _, r in df.iterrows()
        ]
    )


@router.get("/deployed", response_model=DeployedModelsResponse)
def deployed(module: str, _: CurrentUserDep) -> DeployedModelsResponse:
    if module == "disease_biology":
        return DeployedModelsResponse(models=[])
    category = _module_to_category(module)
    df = get_deployed_models(category)
    return DeployedModelsResponse(
        models=[
            DeployedModel(
                model_id=int(r["model_id"]),
                deployment_id=int(r["deployment_id"]),
                deployment_name=str(r["deployment_name"]),
                deployment_description=_str_or_none(r.get("deployment_description")),
                model_display_name=str(r["model_display_name"]),
                model_source_version=_str_or_none(r.get("model_source_version")),
                uc_name=str(r["uc_name"]),
                model_endpoint_name=str(r["model_endpoint_name"]),
            )
            for _, r in df.iterrows()
        ]
    )


@router.get("/batch", response_model=BatchModelsResponse)
def batch(module: str, _: CurrentUserDep) -> BatchModelsResponse:
    if module not in SUPPORTED_MODULES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown module '{module}'. Expected one of: {sorted(SUPPORTED_MODULES)}",
        )
    df = get_batch_models(module)
    return BatchModelsResponse(
        models=[
            BatchModel(
                model_display_name=str(r["model_display_name"]),
                model_description=_str_or_none(r.get("model_description")),
                job_name=str(r["job_name"]),
                cluster_type=_str_or_none(r.get("cluster_type")),
            )
            for _, r in df.iterrows()
        ]
    )
