"""Genesis Workbench MCP server — a thin adapter over the shared capability core.

The MCP pathway owns only protocol concerns: it generates one typed MCP tool per
capability (from genesis_workbench.capabilities) and runs them via the shared
executor (genesis_workbench.executor). All capability logic — the registry, the
endpoint/job/chain/transform execution — lives in the wheel and is shared with the
UI and Vortex pathways.

Hosted as a Databricks App (mcp-genesis-workbench): FastMCP streamable-HTTP at /mcp
on port 8000. Endpoint/job calls run as the app service principal.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
from typing import Optional

import uvicorn
from mcp.server.fastmcp import FastMCP

from genesis_workbench.capabilities import CHAIN, ENDPOINT, JOB, list_capabilities
from genesis_workbench.executor import RUNNABLE_CHAINS, execute_capability, run_status
from genesis_workbench.workbench import initialize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gwb_mcp")

mcp = FastMCP("genesis-workbench")

_PYTYPE = {"int": int, "float": float, "bool": bool}


def _init_lib() -> None:
    initialize(
        core_catalog_name=os.environ["CORE_CATALOG_NAME"],
        core_schema_name=os.environ["CORE_SCHEMA_NAME"],
        sql_warehouse_id=os.environ["SQL_WAREHOUSE"],
    )


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.strip().lower()).strip("_") or "x"


def _tool_for(cap):
    """A tool fn whose signature is the capability's typed inputs (required) +
    params (optional). The handler splits kwargs back into inputs/params and runs
    the shared executor."""
    input_names = [p.name for p in cap.inputs]
    param_names = [p.name for p in cap.params]

    def impl(**kwargs):
        inputs = {k: kwargs[k] for k in input_names if kwargs.get(k) is not None}
        params = {k: kwargs[k] for k in param_names if kwargs.get(k) is not None}
        return execute_capability(cap, inputs=inputs, params=params)

    sig: list[inspect.Parameter] = []
    annotations: dict = {}
    for p in cap.inputs:
        sig.append(inspect.Parameter(p.name, inspect.Parameter.KEYWORD_ONLY, annotation=str))
        annotations[p.name] = str
    for p in cap.params:
        ann = Optional[_PYTYPE.get(p.type, str)]
        sig.append(inspect.Parameter(p.name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=ann))
        annotations[p.name] = ann
    impl.__signature__ = inspect.Signature(sig)
    impl.__annotations__ = annotations
    return impl


def _tool_name(cap) -> str | None:
    short = cap.id.split(":", 1)[-1]
    if cap.kind == ENDPOINT:
        return f"endpoint_{_slug(short)}"
    if cap.kind in (JOB, CHAIN):
        return f"workflow_{_slug(short)}"
    return None  # transforms are canvas plumbing, not MCP tools


def _register_tools() -> int:
    n = 0
    for cap in list_capabilities():
        if cap.kind == CHAIN and (cap.chain_id not in RUNNABLE_CHAINS):
            continue  # listed in capabilities, not yet a runnable tool
        name = _tool_name(cap)
        if name is None:
            continue
        if cap.kind == JOB:
            tail = "Returns the run id + URL — poll get_workflow_run_status."
        else:
            tail = "Runs synchronously and returns the result."
        try:
            mcp.add_tool(
                _tool_for(cap),
                name=name,
                description=f"{cap.label} ({cap.module or 'gwb'}). {cap.description} {tail}",
            )
            n += 1
        except Exception as e:  # noqa: BLE001 — one bad tool must not skip the rest
            logger.warning("skipping tool %s: %s", name, e)

    def get_workflow_run_status(run_id: str) -> dict:
        """Status of a dispatched workflow run by its run id (life-cycle + result + link)."""
        return run_status(run_id)

    def list_capabilities_tool() -> list[dict]:
        """List all Genesis Workbench capabilities exposed here — endpoints and
        prebuilt workflows — with the tool name to call and whether it's runnable."""
        out = []
        for cap in list_capabilities():
            name = _tool_name(cap)
            runnable = name is not None and not (cap.kind == CHAIN and cap.chain_id not in RUNNABLE_CHAINS)
            out.append({"kind": cap.kind, "label": cap.label, "module": cap.module,
                        "tool": name if runnable else None, "runnable": runnable})
        return out

    mcp.add_tool(get_workflow_run_status, name="get_workflow_run_status")
    mcp.add_tool(list_capabilities_tool, name="list_capabilities")
    logger.info("Registered %d capability tools", n)
    return n


def build_app():
    try:
        _init_lib()
        _register_tools()
    except Exception as e:  # noqa: BLE001 — never block startup
        logger.exception("Tool registration failed (serving minimal tool set): %s", e)
    return mcp.streamable_http_app()


app = build_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
