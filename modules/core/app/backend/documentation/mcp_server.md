# MCP Server

## Introduction

Genesis Workbench ships a companion **Model Context Protocol (MCP)** server — `mcp-genesis-workbench` — a Databricks App that exposes every deployed model endpoint and prebuilt workflow as MCP tools. Any MCP client (the Databricks AI Playground, Claude, Cursor, or your own agents) can connect, discover the available tools, and call them. The server reuses the same capability core as the Vortex canvas, so an MCP tool call runs exactly the same code path as the UI.

## What It Achieves

- Makes the platform's models and workflows callable by LLM agents and MCP clients, not just the UI.
- Auto-generates one tool per capability from the live registry — no per-tool wiring to maintain.
- Keeps calls governed and attributable: the server runs as its own Databricks App service principal with explicit grants on the endpoints, jobs, volumes, and models it can use.

## How to Use

After deploying `core` (see the Installation Guide), the MCP app is reachable at:

```
https://<mcp-genesis-workbench-app-url>/mcp
```

Add that URL as a custom MCP server in your client:

- **Databricks AI Playground** — custom MCP servers named `mcp-…` are auto-discovered; select it and call its tools.
- **Claude / Cursor / other clients** — register the `/mcp` URL (OAuth) as a streamable-HTTP MCP server.

Typical flow: call **`list_capabilities`** to see what's available and the tool name to use, then:

- **`endpoint_<name>`** — invoke a model-serving endpoint. Runs synchronously and returns predictions.
- **`workflow_<name>`** — dispatch a prebuilt workflow (a Databricks Job or endpoint-chain). Job-backed workflows return a run id + URL; poll **`get_workflow_run_status`** for life-cycle, result, and a link. Chain-backed workflows run synchronously.

### Inputs

- Each tool's arguments are the capability's typed inputs (required) plus its params (optional) — the same inputs/params the Vortex node exposes.

### Outputs

- Endpoints/chains return their prediction/result payload directly.
- Jobs return `{run_id, run_url, …}`; `get_workflow_run_status(run_id)` returns the run's status, result, and link.

## Security & access control

The MCP server invokes every endpoint/workflow under **the app's service principal**, which is granted `CAN_QUERY` on the endpoints and `CAN_MANAGE_RUN` on the jobs at deploy. There is currently **no per-caller authorization** on the MCP path: anyone who can open the app can call any tool the app SP is entitled to (the same model the UI app uses today). Per-user authorization — gating each call against the caller's `AppPermissionsManager` module access — is a tracked follow-up.

**Until that lands, the app's accessor list IS the access control.** It is pinned declaratively in `resources/mcp_app.yml` (so it survives redeploys):

- The **deployer** gets `CAN_MANAGE` and workspace **admins** always retain access.
- The group in **`var.mcp_app_access_group`** gets `CAN_USE` — set this per workspace to the group entitled to invoke Genesis Workbench workflows/endpoints. It defaults to `admins` (deny-by-default: only admins + the deployer can call the server until you scope it).

To scope access, deploy with `--var mcp_app_access_group=<your-entitled-group>` (or set it in your module env). Do **not** grant the app to "all users" while the per-caller gate is absent.

## How It's Implemented

### Architecture

- A **FastMCP** server serving **streamable HTTP** on port 8000, mounted at `/mcp`, hosted as the Databricks App `mcp-genesis-workbench`.
- On startup it initializes the `genesis_workbench` library and **registers one tool per capability** from `list_capabilities()`: `endpoint_<slug>` for serving endpoints and `workflow_<slug>` for runnable jobs/chains, plus the fixed `list_capabilities` and `get_workflow_run_status` tools. Transforms (canvas plumbing) are not exposed.
- Tool handlers call the **shared executor** (`execute_capability` / `run_status`) — the same core that backs the Vortex orchestrator — so endpoint queries and job dispatch behave identically to the UI. Calls run as the app service principal (OBO tokens lack model-serving scope).
- Deployment is wired into the `core` deploy: `update.sh` stages the wheel + app code into `mcp_app/`, deploys the `mcp_genesis_workbench_app` bundle resource, and the `grant_app_permissions_job` grants **both** app service principals (UI + MCP) CAN_QUERY on endpoints, CAN_MANAGE_RUN on jobs, plus volume/model/catalog access.

### Key Files

- `modules/core/mcp_app/backend/mcp_server.py` — the FastMCP server + tool registration
- `modules/core/mcp_app/app.yml`, `modules/core/mcp_app/requirements.txt` — the Databricks App definition
- `modules/core/mcp_app/scripts/seed_prebuilt_workflows.py` — seeds the prebuilt-workflow capabilities
- `modules/core/library/genesis_workbench/src/genesis_workbench/{capabilities,executor}.py` — shared capability core
- `modules/core/update.sh` — stages + deploys the MCP app and grants its service principal
