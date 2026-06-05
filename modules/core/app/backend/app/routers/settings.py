import os
from datetime import datetime, timedelta, timezone

from databricks.sdk import WorkspaceClient
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.auth import CurrentUserDep
from app.config import get_settings
from app.services import workbench

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingRow(BaseModel):
    key: str
    value: str
    module: str


class SystemSettingsResponse(BaseModel):
    catalog: str
    schema_name: str
    warehouse_id: str
    settings: list[SettingRow]
    workflows: list[SettingRow]


class EndpointRow(BaseModel):
    deployment: str
    endpoint: str
    model: str
    status: str


class EndpointStatusResponse(BaseModel):
    endpoints: list[EndpointRow]


class BatchModelRow(BaseModel):
    model_display_name: str
    model_category: str
    module: str
    cluster_type: str | None
    job_name: str | None
    job_id: str | None


class BatchModelsResponse(BaseModel):
    batch_models: list[BatchModelRow]


class StartEndpointsStatusResponse(BaseModel):
    active: bool
    run_id: int | None = None
    start_time_iso: str | None = None
    duration_hours: int | None = None
    remaining_minutes: int | None = None


class StartEndpointsTriggerRequest(BaseModel):
    num_hours: int


class StartEndpointsTriggerResponse(BaseModel):
    run_id: str


def _endpoint_status_from_resp(resp: dict) -> str:
    state = resp.get("state", {}) or {}
    ready = state.get("ready", "UNKNOWN")
    config_update = state.get("config_update", "NOT_UPDATING")
    entity_state, entity_msg = "", ""
    config = resp.get("config", {}) or {}
    entities = config.get("served_entities") or config.get("served_models") or []
    if entities:
        e_state = entities[0].get("state", {}) or {}
        entity_state = e_state.get("deployment", "") or ""
        entity_msg = e_state.get("deployment_state_message", "") or ""
    if ready == "NOT_READY" and config_update == "IN_PROGRESS":
        return "🟡 Updating"
    if ready == "NOT_READY" and config_update == "UPDATE_FAILED":
        return "🔴 Update failed"
    if ready == "NOT_READY":
        return "⚪ Stopped"
    if entity_state == "DEPLOYMENT_ABORTED":
        return "🔴 Failed"
    if entity_msg == "Scaled to zero":
        return "⚪ Scaled to zero"
    if "Scaling from zero" in entity_msg:
        return "🟡 Starting"
    if ready == "READY":
        return "🟢 Ready"
    return f"⚪ {entity_msg or ready}"


@router.get("/system", response_model=SystemSettingsResponse)
def system_settings(_: CurrentUserDep) -> SystemSettingsResponse:
    s = get_settings()
    rows = workbench.execute_select_query(
        f"SELECT key, value, module FROM {s.catalog}.{s.schema}.settings ORDER BY module, key"
    )
    settings_rows: list[SettingRow] = []
    workflow_rows: list[SettingRow] = []
    for _, r in rows.iterrows():
        item = SettingRow(key=str(r["key"]), value=str(r["value"]), module=str(r["module"]))
        (workflow_rows if item.key.endswith("_job_id") else settings_rows).append(item)

    return SystemSettingsResponse(
        catalog=s.catalog,
        schema_name=s.schema,
        warehouse_id=s.warehouse_id,
        settings=settings_rows,
        workflows=workflow_rows,
    )


@router.get("/endpoints", response_model=EndpointStatusResponse)
def endpoint_statuses(_: CurrentUserDep) -> EndpointStatusResponse:
    s = get_settings()
    df = workbench.execute_select_query(
        f"SELECT deployment_name, model_endpoint_name, deploy_model_uc_name "
        f"FROM {s.catalog}.{s.schema}.model_deployments "
        f"WHERE is_active = true ORDER BY deployment_name"
    )
    w = WorkspaceClient()  # app SP — OBO user tokens lack 'model-serving' scope
    rows: list[EndpointRow] = []
    for _, r in df.iterrows():
        ep_name = r["model_endpoint_name"]
        try:
            resp = w.api_client.do("GET", f"/api/2.0/serving-endpoints/{ep_name}")
            ep_status = _endpoint_status_from_resp(resp)
        except Exception as e:
            cls = type(e).__name__
            msg = str(e).splitlines()[0][:120] if str(e) else ""
            ep_status = f"⚠️ {cls}: {msg}" if msg else f"⚠️ {cls}"
        rows.append(
            EndpointRow(
                deployment=r["deployment_name"],
                endpoint=r["model_endpoint_name"],
                model=r["deploy_model_uc_name"],
                status=ep_status,
            )
        )
    return EndpointStatusResponse(endpoints=rows)


@router.get("/batch-models", response_model=BatchModelsResponse)
def batch_models(_: CurrentUserDep) -> BatchModelsResponse:
    s = get_settings()
    df = workbench.execute_select_query(
        f"SELECT model_display_name, model_category, module, cluster_type, job_name, job_id "
        f"FROM {s.catalog}.{s.schema}.batch_models "
        f"WHERE is_active = true ORDER BY module, model_display_name"
    )
    rows = [
        BatchModelRow(
            model_display_name=str(r["model_display_name"]),
            model_category=str(r["model_category"]),
            module=str(r["module"]),
            cluster_type=(str(r["cluster_type"]) if r.get("cluster_type") is not None else None),
            job_name=(str(r["job_name"]) if r.get("job_name") is not None else None),
            job_id=(str(r["job_id"]) if r.get("job_id") is not None else None),
        )
        for _, r in df.iterrows()
    ]
    return BatchModelsResponse(batch_models=rows)


@router.get("/start-endpoints/status", response_model=StartEndpointsStatusResponse)
def start_endpoints_status(_: CurrentUserDep) -> StartEndpointsStatusResponse:
    job_id = workbench.get_job_id("start_all_endpoints_job_id") or None
    if not job_id:
        return StartEndpointsStatusResponse(active=False)
    try:
        w = WorkspaceClient()
        resp = w.api_client.do(
            "GET", "/api/2.1/jobs/runs/list", query={"job_id": str(job_id), "limit": "5"}
        )
        for run in resp.get("runs", []):
            lifecycle = run.get("state", {}).get("life_cycle_state", "")
            if lifecycle in ("PENDING", "RUNNING", "BLOCKED"):
                start_ms = run.get("start_time")
                start_time = (
                    datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc) if start_ms else None
                )
                num_hours = None
                for p in run.get("job_parameters", []):
                    if p.get("name") == "num_hours":
                        try:
                            # Runs started with the job default carry "default"
                            # but no "value" — fall back so duration/remaining
                            # don't render as "h"/"unknown".
                            num_hours = int(p.get("value") or p.get("default"))
                        except (TypeError, ValueError):
                            num_hours = None
                        break
                remaining_minutes = None
                if start_time and num_hours:
                    end_time = start_time + timedelta(hours=num_hours)
                    remaining = end_time - datetime.now(timezone.utc)
                    remaining_minutes = max(int(remaining.total_seconds() // 60), 0)
                return StartEndpointsStatusResponse(
                    active=True,
                    run_id=run.get("run_id"),
                    start_time_iso=start_time.isoformat() if start_time else None,
                    duration_hours=num_hours,
                    remaining_minutes=remaining_minutes,
                )
        return StartEndpointsStatusResponse(active=False)
    except Exception:
        return StartEndpointsStatusResponse(active=False)


@router.post("/start-endpoints/trigger", response_model=StartEndpointsTriggerResponse)
def start_endpoints_trigger(
    payload: StartEndpointsTriggerRequest, _: CurrentUserDep
) -> StartEndpointsTriggerResponse:
    job_id = workbench.get_job_id("start_all_endpoints_job_id") or None
    if not job_id:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Start All Endpoints job is not configured. Redeploy the core module.",
        )
    s = get_settings()
    try:
        run_id = workbench.execute_workflow(
            int(job_id),
            {
                "catalog": s.catalog,
                "schema": s.schema,
                "sql_warehouse_id": s.warehouse_id,
                "num_hours": str(payload.num_hours),
            },
        )
    except Exception as e:
        msg = str(e)
        if "PERMISSION_DENIED" in msg or "does not have" in msg or "permission" in msg.lower():
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                "The app's service principal lacks run permission on the Start All Endpoints "
                "job. Grant it CAN_MANAGE_RUN (a full core redeploy regrants it).",
            )
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"Failed to start the endpoints job: {e}"
        )
    return StartEndpointsTriggerResponse(run_id=str(run_id))
