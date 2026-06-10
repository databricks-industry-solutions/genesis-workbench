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

import logging
import os
import re

import uvicorn
from databricks.sdk import WorkspaceClient
from mcp.server.fastmcp import FastMCP

from genesis_workbench.models import ModelCategory, get_deployed_models
from genesis_workbench.workbench import initialize

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

    def list_capabilities() -> list[dict]:
        """List the Genesis Workbench capabilities (deployed model-serving
        endpoints) currently available as MCP tools — display name, module,
        endpoint, and the tool name to call."""
        return [
            {
                "tool": f"endpoint_{_slug(e['display_name'])}",
                "display_name": e["display_name"],
                "module": e["module"],
                "endpoint": e["endpoint_name"],
            }
            for e in _deployed_endpoints()
        ]

    mcp.add_tool(list_capabilities, name="list_capabilities")
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
