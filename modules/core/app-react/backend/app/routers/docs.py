from fastapi import APIRouter
from pydantic import BaseModel

from app.auth import CurrentUserDep
from app.services import docs as docs_service

router = APIRouter(prefix="/api/docs", tags=["docs"])


class DocEntry(BaseModel):
    file: str
    title: str
    content: str


class DocsResponse(BaseModel):
    docs: list[DocEntry]


@router.get("", response_model=DocsResponse)
def list_docs(_: CurrentUserDep) -> DocsResponse:
    return DocsResponse(
        docs=[DocEntry(file=d.file, title=d.title, content=d.content) for d in docs_service.list_documents()]
    )
