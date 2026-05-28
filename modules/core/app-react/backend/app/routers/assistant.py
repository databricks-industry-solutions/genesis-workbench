from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import WorkspaceClientDep
from app.config import get_settings
from app.services import docs as docs_service

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class AssistantQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AssistantQueryResponse(BaseModel):
    answer: str


def _build_doc_context() -> str:
    lines = []
    for doc in docs_service.list_documents():
        intro = ""
        sections: list[str] = []
        in_intro = False
        for line in doc.content.splitlines():
            if line.startswith("# ") and not in_intro:
                in_intro = True
                continue
            if line.startswith("## ") or line.startswith("### "):
                in_intro = False
                heading = line.lstrip("#").strip()
                if heading:
                    sections.append(heading)
                continue
            if in_intro:
                intro += line + " "
        entry = f"- **{doc.title}**: {intro.strip()}"
        if sections:
            entry += " Sections: " + ", ".join(sections) + "."
        lines.append(entry)
    return "\n".join(lines)


@router.post("/query", response_model=AssistantQueryResponse)
def query(payload: AssistantQueryRequest, w: WorkspaceClientDep) -> AssistantQueryResponse:
    endpoint = get_settings().llm_endpoint_name
    if not endpoint:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "LLM endpoint not configured (LLM_ENDPOINT_NAME)",
        )

    system_prompt = f"""You are a helpful assistant for Genesis Workbench, a bioinformatics platform.
Based on the user's request, suggest which workflows they should use and briefly explain why.
If the request doesn't match any available workflow, say so politely and suggest they check back later as new workflows are added regularly.

Keep your response concise — use bullet points with workflow names in bold.

Available workflows:
{_build_doc_context()}"""

    try:
        response = w.serving_endpoints.query(
            name=endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=ChatMessageRole.USER, content=payload.query),
            ],
            max_tokens=512,
        )
        return AssistantQueryResponse(answer=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"LLM endpoint call failed: {e}",
        )
