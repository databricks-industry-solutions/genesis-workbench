"""Vortex (ai_canvas) — FastAPI router.

Visible product name is "Vortex"; the feature id / route prefix / MLflow tags
all use `ai_canvas` so the display name can change in one place (the frontend
tab label) without touching the backend.

V1 surface: GET /api/ai_canvas/catalog. Generation, persistence, run dispatch,
and result endpoints are added in subsequent increments.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from genesis_workbench.workbench import UserInfo
from pydantic import BaseModel, Field

from app.auth import CurrentUser, CurrentUserDep
from app.config import get_settings
from app.services import ai_canvas as svc

router = APIRouter(prefix="/api/ai_canvas", tags=["ai_canvas"])


def _build_user_info(user: CurrentUser) -> UserInfo:
    return UserInfo(
        user_email=user.email or "",
        user_name=user.preferred_username or "",
        user_id=user.user_id or "",
        user_groups=[],
        user_access_token=user.access_token,
        user_display_name=user.preferred_username or "",
    )


class CatalogPort(BaseModel):
    name: str
    dtype: str
    label: str


class CatalogParam(BaseModel):
    name: str
    label: str
    type: str
    default: object | None = None
    options: list[str] = []
    required: bool = False
    help: str = ""


class CatalogNode(BaseModel):
    type: str
    label: str
    category: str
    description: str = ""
    module: str | None = None
    available: bool = True
    inputs: list[CatalogPort] = []
    outputs: list[CatalogPort] = []
    params: list[CatalogParam] = []


class CatalogResponse(BaseModel):
    nodes: list[CatalogNode]


@router.get("/catalog", response_model=CatalogResponse)
def catalog(_: CurrentUserDep) -> CatalogResponse:
    """All node types droppable on the Vortex canvas, with live availability."""
    return CatalogResponse(nodes=[CatalogNode(**n) for n in svc.build_catalog()])


# ─── AI graph generation ─────────────────────────────────────────────────────


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    params: dict = {}
    position: dict = {"x": 0, "y": 0}


class GraphEdge(BaseModel):
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None


class Graph(BaseModel):
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


class GenerateRequest(BaseModel):
    goal: str = Field(..., min_length=1)


class GenerateResponse(BaseModel):
    graph: Graph


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, _: CurrentUserDep) -> GenerateResponse:
    """Natural-language goal → a candidate workflow graph the user can edit."""
    endpoint = get_settings().llm_endpoint_name
    if not endpoint:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "LLM endpoint not configured (LLM_ENDPOINT_NAME)",
        )
    try:
        graph = svc.generate_graph(payload.goal, endpoint)
    except svc.GraphGenerationError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Graph generation failed: {e}")
    return GenerateResponse(graph=Graph(**graph))


# ─── Workflow persistence ────────────────────────────────────────────────────


class SaveWorkflowRequest(BaseModel):
    workflow_id: int | None = None
    name: str = Field(..., min_length=1)
    description: str = ""
    graph: Graph


class SaveWorkflowResponse(BaseModel):
    workflow_id: int


class WorkflowSummary(BaseModel):
    workflow_id: int
    name: str
    description: str
    updated_date: str


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowSummary]


class WorkflowDetail(BaseModel):
    workflow_id: int
    name: str
    description: str
    graph: Graph


@router.post("/workflows", response_model=SaveWorkflowResponse)
def save_workflow(payload: SaveWorkflowRequest, user: CurrentUserDep) -> SaveWorkflowResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    try:
        wid = svc.save_workflow(
            user_email=user.email,
            name=payload.name,
            description=payload.description,
            graph=payload.graph.model_dump(),
            workflow_id=payload.workflow_id,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Save failed: {e}")
    return SaveWorkflowResponse(workflow_id=wid)


@router.get("/workflows", response_model=WorkflowListResponse)
def list_workflows(user: CurrentUserDep) -> WorkflowListResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    rows = svc.list_workflows(user.email)
    return WorkflowListResponse(workflows=[WorkflowSummary(**r) for r in rows])


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetail)
def get_workflow(workflow_id: int, user: CurrentUserDep) -> WorkflowDetail:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    wf = svc.get_workflow(workflow_id, user.email)
    if wf is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workflow not found")
    return WorkflowDetail(**wf)


@router.delete("/workflows/{workflow_id}")
def delete_workflow(workflow_id: int, user: CurrentUserDep) -> dict:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    svc.deactivate_workflow(workflow_id, user.email)
    return {"ok": True}


# ─── Run dispatch / status / search / result ─────────────────────────────────


class RunRequest(BaseModel):
    graph: Graph
    run_name: str = Field("ai_canvas_run", min_length=1)


class RunResponse(BaseModel):
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str


class RunStatusResponse(BaseModel):
    status: str
    job_status: str
    node_status: dict[str, str]
    node_error: dict[str, str]
    run_name: str


class RunSummary(BaseModel):
    run_id: str
    run_name: str
    job_status: str
    node_count: int | None = None
    start_time: str
    run_url: str = ""


class RunsResponse(BaseModel):
    runs: list[RunSummary]


class RunResultResponse(BaseModel):
    result: dict


@router.post("/run", response_model=RunResponse)
def run(payload: RunRequest, user: CurrentUserDep) -> RunResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    if not payload.graph.nodes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Workflow is empty")
    try:
        result = svc.start_workflow_run(
            user_info=_build_user_info(user),
            graph=payload.graph.model_dump(),
            run_name=payload.run_name,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to dispatch workflow: {e}")
    return RunResponse(
        job_id=result.job_id,
        job_run_id=result.job_run_id,
        mlflow_run_id=result.mlflow_run_id,
        experiment_id=result.experiment_id,
    )


@router.get("/run/{run_id}/status", response_model=RunStatusResponse)
def run_status(run_id: str, _: CurrentUserDep) -> RunStatusResponse:
    return RunStatusResponse(**svc.get_run_status(run_id))


@router.get("/run/{run_id}/result", response_model=RunResultResponse)
def run_result(run_id: str, _: CurrentUserDep) -> RunResultResponse:
    return RunResultResponse(result=svc.get_run_result(run_id))


@router.get("/runs", response_model=RunsResponse)
def runs(user: CurrentUserDep, text: str = "") -> RunsResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    rows = svc.search_runs(user.email, text.strip())
    return RunsResponse(runs=[RunSummary(**r) for r in rows])
