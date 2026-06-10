"""Genesis Workbench MCP server — P0 spike (live-model endpoints).

Registry-as-source: capabilities are read live from the existing deployed-models
Delta registry (`model_deployments` ⋈ `models`, via the genesis_workbench lib).
Each deployed serving endpoint is exposed as one MCP tool, plus a
`list_capabilities` discovery tool.

Hosted as a Databricks App (`mcp-genesis-workbench`): the FastMCP streamable-HTTP
ASGI app is served by uvicorn at `/mcp` on the app port (8000). Endpoint calls use
the app service principal (ambient in Databricks Apps).

P1 will add typed-per-capability schemas (promoting the shared registry) and
workflow tools backed by a new prebuilt-workflows registry + shared executor.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
from typing import Optional

import uvicorn
from databricks.sdk import WorkspaceClient
from mcp.server.fastmcp import FastMCP

from genesis_workbench.models import ModelCategory, get_deployed_models
from genesis_workbench.workbench import execute_select_query, initialize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gwb_mcp")

_MODULES = ("large_molecule", "small_molecule", "single_cell", "genomics")

mcp = FastMCP("genesis-workbench")


def _init_lib() -> None:
    """Wire the genesis_workbench lib exactly like the core app does, so the
    deployed-models registry + warehouse connection are available."""
    initialize(
        core_catalog_name=os.environ["CORE_CATALOG_NAME"],
        core_schema_name=os.environ["CORE_SCHEMA_NAME"],
        sql_warehouse_id=os.environ["SQL_WAREHOUSE"],
    )


def _deployed_endpoints() -> list[dict]:
    """Live list of deployed serving endpoints from the Delta registry."""
    rows: list[dict] = []
    for module in _MODULES:
        try:
            df = get_deployed_models(ModelCategory(module))
        except Exception as e:  # noqa: BLE001 — degrade gracefully per module
            logger.warning("deployed lookup failed for %s: %s", module, e)
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "display_name": str(r["model_display_name"]),
                    "uc_name": str(r["uc_name"]),
                    "endpoint_name": str(r["model_endpoint_name"]),
                    "module": module,
                }
            )
    return rows


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_") or "endpoint"


def _make_endpoint_tool(ep: dict):
    def call_endpoint(inputs: list[str]) -> list:
        w = WorkspaceClient()
        resp = w.serving_endpoints.query(name=ep["endpoint_name"], inputs=inputs)
        return resp.predictions

    return call_endpoint


def _prebuilt_workflows() -> list[dict]:
    """Prebuilt-workflow capabilities from the `prebuilt_workflows` Delta
    registry (the workflows counterpart to the endpoints registry)."""
    cat = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    try:
        df = execute_select_query(
            f"SELECT workflow_key, label, kind, module, job_name, inputs_json, "
            f"outputs_json, params_json, description FROM {cat}.{schema}.prebuilt_workflows "
            f"WHERE is_active = true"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("prebuilt_workflows read failed: %s", e)
        return []
    out: list[dict] = []
    for _, r in df.iterrows():
        out.append({k: (None if r[k] is None else str(r[k])) for k in
                    ("workflow_key", "label", "kind", "module", "job_name",
                     "inputs_json", "outputs_json", "params_json", "description")})
    return out


def _job_id_for(job_name: str):
    for j in WorkspaceClient().jobs.list(name=job_name):
        return int(j.job_id)
    return None


_PYTYPE = {"int": int, "float": float, "bool": bool}


def _arg_fields(wf: dict) -> list[tuple]:
    """Typed args for a workflow tool: required inputs (str) + optional params
    (typed, default None → falls back to the job's own defaults). Returns
    (name, annotation, required) tuples."""
    fields: list[tuple] = []
    seen: set[str] = set()
    for p in json.loads(wf.get("inputs_json") or "[]"):
        n = p.get("name")
        if n and n not in seen:
            seen.add(n)
            fields.append((n, str, True))
    for p in json.loads(wf.get("params_json") or "[]"):
        n = p.get("name")
        if n and n not in seen:
            seen.add(n)
            fields.append((n, Optional[_PYTYPE.get(p.get("type"), str)], False))
    return fields


def _make_workflow_tool(wf: dict):
    fields = _arg_fields(wf)

    def impl(**kwargs) -> dict:
        w = WorkspaceClient()
        job_id = _job_id_for(wf["job_name"])
        if job_id is None:
            raise RuntimeError(f"Job '{wf['job_name']}' not found / not deployed.")
        # Only pass set args; unset optional params defer to the job's defaults.
        params = {k: str(v) for k, v in kwargs.items() if v is not None}
        run = w.jobs.run_now(job_id=job_id, job_parameters=params)
        host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        return {
            "run_id": str(run.run_id),
            "job_id": str(job_id),
            "run_url": f"{host}/jobs/{job_id}/runs/{run.run_id}" if host else "",
            "status_tool": "get_workflow_run_status",
        }

    sig_params = []
    annotations: dict = {}
    for name, ann, required in fields:
        if required:
            sig_params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=ann))
        else:
            sig_params.append(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=ann)
            )
        annotations[name] = ann
    impl.__signature__ = inspect.Signature(sig_params)
    impl.__annotations__ = annotations
    return impl


def _fmt_io(spec_json: str | None) -> str:
    try:
        items = json.loads(spec_json or "[]")
        return ", ".join(f"{p.get('name')}:{p.get('dtype', p.get('type', 'any'))}" for p in items) or "—"
    except Exception:  # noqa: BLE001
        return "—"


def _register_tools() -> int:
    seen: set[str] = set()
    for ep in _deployed_endpoints():
        tool_name = f"endpoint_{_slug(ep['display_name'])}"
        if tool_name in seen:
            continue
        seen.add(tool_name)
        mcp.add_tool(
            _make_endpoint_tool(ep),
            name=tool_name,
            description=(
                f"Call the '{ep['display_name']}' Genesis Workbench model-serving "
                f"endpoint ({ep['module']}). Pass `inputs` as a list of strings "
                f"(e.g. protein sequences or SMILES); returns the model predictions."
            ),
        )

    workflows = _prebuilt_workflows()
    n_wf = 0
    for wf in workflows:
        # Endpoint-chain composites aren't dispatchable yet (shared executor lands
        # with the run-wiring) — list them in capabilities but don't add a tool.
        if wf["kind"] != "databricks_job" or not wf["job_name"]:
            continue
        tool_name = f"workflow_{_slug(wf['workflow_key'])}"
        if tool_name in seen:
            continue
        try:
            mcp.add_tool(
                _make_workflow_tool(wf),
                name=tool_name,
                description=(
                    f"Dispatch the '{wf['label']}' Genesis Workbench workflow ({wf['module']}). "
                    f"{wf['description']} Returns the run id + URL — poll get_workflow_run_status. "
                    f"Inputs are file paths / sequences / SMILES; optional params override job defaults."
                ),
            )
        except Exception as e:  # noqa: BLE001 — one bad tool must not skip the rest
            logger.warning("skipping workflow tool %s: %s", tool_name, e)
            continue
        seen.add(tool_name)
        n_wf += 1

    def get_workflow_run_status(run_id: str) -> dict:
        """Status of a dispatched workflow run by its run id (life-cycle +
        result state + a link)."""
        w = WorkspaceClient()
        run = w.jobs.get_run(run_id=int(run_id))
        st = run.state
        host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        return {
            "run_id": str(run_id),
            "life_cycle_state": st.life_cycle_state.value if st and st.life_cycle_state else None,
            "result_state": st.result_state.value if st and st.result_state else None,
            "run_url": f"{host}/jobs/{run.job_id}/runs/{run_id}" if host and run.job_id else "",
        }

    mcp.add_tool(get_workflow_run_status, name="get_workflow_run_status")

    def list_capabilities() -> list[dict]:
        """List all Genesis Workbench capabilities exposed as MCP tools: deployed
        model-serving endpoints and prebuilt workflows (with the tool name to
        call). Endpoint-chain workflows are listed but not yet dispatchable."""
        caps: list[dict] = []
        for e in _deployed_endpoints():
            caps.append({"kind": "endpoint", "tool": f"endpoint_{_slug(e['display_name'])}",
                         "label": e["display_name"], "module": e["module"]})
        for wf in _prebuilt_workflows():
            runnable = wf["kind"] == "databricks_job" and bool(wf["job_name"])
            caps.append({"kind": wf["kind"],
                         "tool": f"workflow_{_slug(wf['workflow_key'])}" if runnable else None,
                         "label": wf["label"], "module": wf["module"], "runnable": runnable})
        return caps

    mcp.add_tool(list_capabilities, name="list_capabilities")
    logger.info("Registered %d endpoint + %d workflow tools", len(seen) - n_wf, n_wf)
    return len(seen)


def build_app():
    """Initialize the lib + register tools, then return the streamable-HTTP ASGI
    app. Registration is best-effort so the server always starts (and /mcp is
    reachable) even if the registry is briefly unavailable."""
    try:
        _init_lib()
        n = _register_tools()
        logger.info("Registered %d endpoint tools + list_capabilities", n)
    except Exception as e:  # noqa: BLE001 — never block startup
        logger.exception("Tool registration failed (serving empty tool set): %s", e)
    return mcp.streamable_http_app()


app = build_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
