import mlflow
from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import CurrentUserDep
from app.services import workbench

router = APIRouter(prefix="/api", tags=["profile"])


class ProfileResponse(BaseModel):
    email: str
    user_settings: dict[str, str]


class ProfileSaveRequest(BaseModel):
    user_display_name: str = Field(..., min_length=1)
    mlflow_experiment_folder: str = Field(..., min_length=1)


class ThemeSaveRequest(BaseModel):
    theme: str = Field(..., pattern=r"^(dark|light)$")


class MlflowTestRequest(BaseModel):
    mlflow_experiment_folder: str


class MlflowTestResponse(BaseModel):
    ok: bool
    message: str


def _verify_mlflow_folder(user_email: str, base_folder: str) -> None:
    w = WorkspaceClient()
    base_path = f"Users/{user_email}/{base_folder}"
    w.workspace.mkdirs(f"/Workspace/{base_path}")
    experiment_path = f"/{base_path}/__test__"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_path)
    mlflow.delete_experiment(experiment_id=experiment.experiment_id)


@router.get("/profile", response_model=ProfileResponse)
def get_profile(user: CurrentUserDep) -> ProfileResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    return ProfileResponse(
        email=user.email,
        user_settings=workbench.get_user_settings(user_email=user.email),
    )


@router.put("/profile", response_model=ProfileResponse)
def save_profile(payload: ProfileSaveRequest, user: CurrentUserDep) -> ProfileResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")

    try:
        _verify_mlflow_folder(user.email, payload.mlflow_experiment_folder)
    except Exception as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"MLflow folder access failed. Confirm the folder exists and the app service principal has Can Manage. ({e})",
        )

    # Merge with existing settings so user-set values that aren't part of
    # this form (e.g. `theme`) aren't dropped by the underlying full
    # delete+insert in save_user_settings.
    existing = workbench.get_user_settings(user_email=user.email)
    merged = {
        **existing,
        "user_display_name": payload.user_display_name,
        "mlflow_experiment_folder": payload.mlflow_experiment_folder,
        "setup_done": "Y",
    }
    workbench.save_user_settings(user_email=user.email, user_settings=merged)
    return ProfileResponse(
        email=user.email,
        user_settings=workbench.get_user_settings(user_email=user.email),
    )


@router.put("/profile/theme", response_model=ProfileResponse)
def save_theme(payload: ThemeSaveRequest, user: CurrentUserDep) -> ProfileResponse:
    """Persist the user's UI theme preference without disturbing the rest
    of their profile (display name, MLflow folder, setup_done flag)."""
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    existing = workbench.get_user_settings(user_email=user.email)
    merged = {**existing, "theme": payload.theme}
    workbench.save_user_settings(user_email=user.email, user_settings=merged)
    return ProfileResponse(email=user.email, user_settings=merged)


@router.post("/mlflow/test", response_model=MlflowTestResponse)
def mlflow_test(payload: MlflowTestRequest, user: CurrentUserDep) -> MlflowTestResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email missing from headers")
    try:
        _verify_mlflow_folder(user.email, payload.mlflow_experiment_folder)
        return MlflowTestResponse(ok=True, message="Experiment access verified.")
    except Exception as e:
        return MlflowTestResponse(
            ok=False,
            message=f"Experiment folder access failed: {e}",
        )
