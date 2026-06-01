from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
