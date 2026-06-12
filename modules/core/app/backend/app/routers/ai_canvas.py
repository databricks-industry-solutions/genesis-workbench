"""Vortex (ai_canvas) — FastAPI router.

Visible product name is "Vortex"; the feature id / route prefix / MLflow tags
all use `ai_canvas` so the display name can change in one place (the frontend
tab label) without touching the backend.

V1 surface: GET /api/ai_canvas/catalog. Generation, persistence, run dispatch,
and result endpoints are added in subsequent increments.
"""
from __future__ import annotations

import json
import logging
import time
import traceback

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from genesis_workbench.workbench import UserInfo
from pydantic import BaseModel, Field

from app.auth import CurrentUser, CurrentUserDep
from app.config import get_settings
from app.services import ai_canvas as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai_canvas", tags=["ai_canvas"])
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


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
    # Inline values for input ports (convertible fields). A wired edge to a port
    # overrides its inline value at run time; see the orchestrator's gather_inputs.
    inputs: dict = {}
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


@router.post("/generate/stream")
def generate_stream(payload: GenerateRequest, _: CurrentUserDep) -> StreamingResponse:
    """Streamed generation: emits `thought` events (the model's plan, paced for a
    live feel) then a `result` event carrying the graph. Falls back to `error`."""
    endpoint = get_settings().llm_endpoint_name

    def _events():
        if not endpoint:
            yield f"event: error\ndata: {json.dumps({'message': 'LLM endpoint not configured'})}\n\n"
            return
        # The service yields phase milestones as it works (draft → plan bullets →
        # review), so the feed keeps moving instead of sitting on one message
        # through the blocking LLM calls. A final 'result' event carries the graph.
        try:
            for kind, payload_evt in svc.generate_events(payload.goal, endpoint):
                if kind == "thought":
                    yield f"event: thought\ndata: {json.dumps({'text': payload_evt})}\n\n"
                    time.sleep(0.3)  # brief pace so bursts read as a live feed
                elif kind == "result":
                    yield f"event: result\ndata: {json.dumps(payload_evt)}\n\n"
        except Exception as e:  # noqa: BLE001 — surface as an SSE error event
            # Log the full traceback (the SSE path otherwise swallows it, so the
            # app log stayed empty) and surface the exception TYPE + where it came
            # from so a bare "'name'" isn't the only signal.
            logger.exception("generate_stream failed for goal=%r", payload.goal)
            tb = traceback.extract_tb(e.__traceback__)
            where = f"{tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}" if tb else "?"
            msg = f"{type(e).__name__}: {e} (at {where})"
            yield f"event: error\ndata: {json.dumps({'message': msg})}\n\n"
            return

    return StreamingResponse(_events(), media_type="text/event-stream", headers=_SSE_HEADERS)


class TransformSuggestRequest(BaseModel):
    source_dtype: str
    target_dtype: str
    source_label: str = ""
    target_label: str = ""


class TransformSuggestResponse(BaseModel):
    type: str | None = None
    label: str | None = None
    params: dict = Field(default_factory=dict)


@router.post("/transform-suggest", response_model=TransformSuggestResponse)
def transform_suggest(
    payload: TransformSuggestRequest, _: CurrentUserDep
) -> TransformSuggestResponse:
    """When two ports' dtypes don't match, suggest a transform node that bridges
    them. Returns type=None if nothing fits (the UI then shows the mismatch)."""
    endpoint = get_settings().llm_endpoint_name
    if not endpoint:
        return TransformSuggestResponse(type=None)
    res = svc.suggest_transform(
        source_dtype=payload.source_dtype,
        target_dtype=payload.target_dtype,
        source_label=payload.source_label,
        target_label=payload.target_label,
        llm_endpoint=endpoint,
    )
    return TransformSuggestResponse(**res) if res else TransformSuggestResponse(type=None)


# ─── Workflow persistence ────────────────────────────────────────────────────


# workflow_id is a time_ns() BIGINT — it MUST cross to the React UI as a string
# or JS rounds it (exceeds Number.MAX_SAFE_INTEGER) and Load lookups miss.
class SaveWorkflowRequest(BaseModel):
    workflow_id: str | None = None
    name: str = Field(..., min_length=1)
    description: str = ""
    graph: Graph


class SaveWorkflowResponse(BaseModel):
    workflow_id: str


class WorkflowSummary(BaseModel):
    workflow_id: str
    name: str
    description: str
    updated_date: str


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowSummary]


class WorkflowDetail(BaseModel):
    workflow_id: str
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
    return SaveWorkflowResponse(workflow_id=str(wid))


@router.get("/workflows", response_model=WorkflowListResponse)
def list_workflows(user: CurrentUserDep) -> WorkflowListResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    rows = svc.list_workflows(user.email)
    return WorkflowListResponse(workflows=[WorkflowSummary(**r) for r in rows])


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetail)
def get_workflow(workflow_id: str, user: CurrentUserDep) -> WorkflowDetail:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    wf = svc.get_workflow(workflow_id, user.email)
    if wf is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workflow not found")
    return WorkflowDetail(**wf)


@router.delete("/workflows/{workflow_id}")
def delete_workflow(workflow_id: str, user: CurrentUserDep) -> dict:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    svc.deactivate_workflow(workflow_id, user.email)
    return {"ok": True}


# ─── Run dispatch / status / search / result ─────────────────────────────────


class RunRequest(BaseModel):
    graph: Graph
    experiment_name: str = Field("gwb_ai_canvas", min_length=1)
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
    page: int = 1
    has_more: bool = False


class RunResultResponse(BaseModel):
    result: dict
    graph: dict | None = None
    node_status: dict = {}
    node_error: dict = {}


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
            experiment_name=payload.experiment_name,
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


class NodeJobErrorResponse(BaseModel):
    found: bool
    job_run_id: str = ""
    run_page_url: str = ""
    node_error: str = ""
    message: str = ""
    tasks: list[dict] = []


@router.get("/run/{run_id}/node/{node_id}/job-error", response_model=NodeJobErrorResponse)
def node_job_error(run_id: str, node_id: str, _: CurrentUserDep) -> NodeJobErrorResponse:
    """Dig into the originating Databricks job behind a failed node — its real
    error/stack trace + a link to the job run page."""
    return NodeJobErrorResponse(**svc.get_node_job_error(run_id, node_id))


@router.get("/run/{run_id}/result", response_model=RunResultResponse)
def run_result(run_id: str, _: CurrentUserDep) -> RunResultResponse:
    d = svc.get_run_result(run_id)
    return RunResultResponse(
        result=d.get("result", {}), graph=d.get("graph"),
        node_status=d.get("node_status", {}), node_error=d.get("node_error", {}),
    )


@router.get("/runs", response_model=RunsResponse)
def runs(user: CurrentUserDep, text: str = "", page: int = 1) -> RunsResponse:
    if not user.email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User email unavailable")
    rows, has_more = svc.search_runs(user.email, text.strip(), page=page, page_size=20)
    return RunsResponse(
        runs=[RunSummary(**r) for r in rows], page=max(1, page), has_more=has_more
    )
