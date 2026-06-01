from dataclasses import dataclass
from typing import Annotated

from databricks.sdk import WorkspaceClient
from fastapi import Depends, HTTPException, Request, status


@dataclass(frozen=True)
class CurrentUser:
    email: str | None
    preferred_username: str | None
    user_id: str | None
    access_token: str


def get_current_user(request: Request) -> CurrentUser:
    token = request.headers.get("X-Forwarded-Access-Token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Forwarded-Access-Token header missing — running outside Databricks Apps?",
        )
    return CurrentUser(
        email=request.headers.get("X-Forwarded-Email"),
        preferred_username=request.headers.get("X-Forwarded-Preferred-Username"),
        user_id=request.headers.get("X-Forwarded-User"),
        access_token=token,
    )


def get_workspace_client(
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> WorkspaceClient:
    return WorkspaceClient(token=user.access_token, auth_type="pat")


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
WorkspaceClientDep = Annotated[WorkspaceClient, Depends(get_workspace_client)]
