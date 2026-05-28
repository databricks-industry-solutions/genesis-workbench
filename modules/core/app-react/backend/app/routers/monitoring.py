import os

from fastapi import APIRouter
from pydantic import BaseModel

from app.auth import CurrentUserDep
from app.config import get_settings
from app.services import workbench

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


class WorkflowRun(BaseModel):
    job_id: int
    job_name: str
    run_id: int
    lifecycle_state: str | None
    result_state: str | None
    start_time_ms: int | None
    end_time_ms: int | None
    creator_user_name: str | None
    run_url: str


class WorkflowRunsResponse(BaseModel):
    runs: list[WorkflowRun]


class AdminDashboardResponse(BaseModel):
    embed_url: str | None


def _host() -> str:
    host = os.getenv("DATABRICKS_HOSTNAME", "")
    return host if host.startswith("https://") else f"https://{host}"


def _make_run_url(job_id: int, run_id: int) -> str:
    return f"{_host()}/jobs/{job_id}/runs/{run_id}"


@router.get("/runs", response_model=WorkflowRunsResponse)
def runs(user: CurrentUserDep, days_back: int = 7) -> WorkflowRunsResponse:
    creator = user.preferred_username or None
    job_status_dict = workbench.get_workflow_job_status(
        tag_key="application",
        tag_value="genesis_workbench",
        days_back=days_back,
        creator_filter=creator,
    )

    out: list[WorkflowRun] = []
    for job_name, job_data in job_status_dict.items():
        for run in job_data.get("runs", []):
            state = run.get("state")
            result_state = run.get("result_state")
            out.append(
                WorkflowRun(
                    job_id=job_data["job_id"],
                    job_name=job_name,
                    run_id=run["run_id"],
                    lifecycle_state=getattr(state, "value", None) if state else None,
                    result_state=getattr(result_state, "value", None) if result_state else None,
                    start_time_ms=run.get("start_time"),
                    end_time_ms=run.get("end_time"),
                    creator_user_name=run.get("creator_user_name"),
                    run_url=_make_run_url(job_data["job_id"], run["run_id"]),
                )
            )
    out.sort(key=lambda r: r.start_time_ms or 0, reverse=True)
    return WorkflowRunsResponse(runs=out)


@router.get("/admin-dashboard", response_model=AdminDashboardResponse)
def admin_dashboard(_: CurrentUserDep) -> AdminDashboardResponse:
    dash_id = get_settings().admin_usage_dashboard_id
    if not dash_id:
        return AdminDashboardResponse(embed_url=None)
    return AdminDashboardResponse(embed_url=f"{_host()}/embed/dashboardsv3/{dash_id}")
