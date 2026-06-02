from fastapi import APIRouter
from pydantic import BaseModel

from app.auth import CurrentUserDep, WorkspaceClientDep

router = APIRouter(prefix="/api", tags=["me"])


class HeaderIdentity(BaseModel):
    email: str | None = None
    preferred_username: str | None = None
    user_id: str | None = None


class SdkIdentity(BaseModel):
    user_name: str | None = None
    display_name: str | None = None
    id: str | None = None
    active: bool | None = None


class MeResponse(BaseModel):
    from_headers: HeaderIdentity
    from_workspace_client: SdkIdentity


@router.get("/me", response_model=MeResponse)
def me(user: CurrentUserDep, w: WorkspaceClientDep) -> MeResponse:
    sdk_me = w.current_user.me()
    return MeResponse(
        from_headers=HeaderIdentity(
            email=user.email,
            preferred_username=user.preferred_username,
            user_id=user.user_id,
        ),
        from_workspace_client=SdkIdentity(
            user_name=sdk_me.user_name,
            display_name=sdk_me.display_name,
            id=sdk_me.id,
            active=sdk_me.active,
        ),
    )
