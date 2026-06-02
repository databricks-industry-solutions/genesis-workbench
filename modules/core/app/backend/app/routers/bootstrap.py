from databricks.sdk import WorkspaceClient
from fastapi import APIRouter
from pydantic import BaseModel

from app.auth import CurrentUserDep, WorkspaceClientDep
from app.config import get_settings
from app.services import workbench

router = APIRouter(prefix="/api", tags=["bootstrap"])


class EnvInfo(BaseModel):
    catalog: str
    schema_name: str
    warehouse_id: str
    llm_endpoint_name: str | None
    app_name: str | None
    app_service_principal_id: str | None
    admin_usage_dashboard_id: str | None


class UserInfo(BaseModel):
    email: str | None
    preferred_username: str | None
    display_name: str | None
    user_name: str | None


class BootstrapResponse(BaseModel):
    env: EnvInfo
    user: UserInfo
    user_settings: dict[str, str]
    deployed_modules: list[str]


def _app_service_principal_id(app_name: str | None) -> str | None:
    if not app_name:
        return None
    try:
        w = WorkspaceClient()
        return w.apps.get(name=app_name).service_principal_client_id
    except Exception:
        return None


@router.get("/bootstrap", response_model=BootstrapResponse)
def bootstrap(user: CurrentUserDep, w: WorkspaceClientDep) -> BootstrapResponse:
    s = get_settings()
    user_email = user.email or ""

    display_name: str | None = None
    user_name: str | None = None
    try:
        me = w.current_user.me()
        display_name = me.display_name
        user_name = me.user_name
    except Exception:
        pass

    deployed_modules = workbench.get_deployed_modules()
    user_settings = workbench.get_user_settings(user_email=user_email) if user_email else {}

    return BootstrapResponse(
        env=EnvInfo(
            catalog=s.catalog,
            schema_name=s.schema,
            warehouse_id=s.warehouse_id,
            llm_endpoint_name=s.llm_endpoint_name,
            app_name=s.app_name,
            app_service_principal_id=_app_service_principal_id(s.app_name),
            admin_usage_dashboard_id=s.admin_usage_dashboard_id,
        ),
        user=UserInfo(
            email=user.email,
            preferred_username=user.preferred_username,
            display_name=display_name,
            user_name=user_name,
        ),
        user_settings=user_settings,
        deployed_modules=deployed_modules,
    )
