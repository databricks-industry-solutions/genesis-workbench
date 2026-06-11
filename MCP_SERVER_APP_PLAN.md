# Genesis Workbench — MCP Server App: Design & Plan

> Status: **PLAN (no code yet)** — 2026-06-10. Supersedes the earlier parked plan
> (one capability = one MCP tool = one Vortex node, now grounded in registries).

## Goal

Expose Genesis Workbench's **live models** (deployed serving endpoints) and
**prebuilt workflows** (jobs + endpoint-chains) as **MCP tools**, hosted as a
sibling Databricks App (`mcp-genesis-workbench`), so external agents (AI
Playground, Claude, Cursor) can drive GWB. The same capability definitions feed
the **Vortex** canvas, so MCP and the canvas are two consumers of one source of
truth — not two implementations.

**Unifying principle:** *one capability = one MCP tool = one Vortex node.*

## Decisions (locked)

1. **Registry-as-source.** A shared capability registry is the source of truth
   (rich, *typed* model: input/output port dtypes, params, executor). The MCP
   server is **generated from it**; Vortex reads it. MCP is the external face,
   not the registry owner. (Can converge Vortex onto MCP later — not now.)
2. **Typed tool per capability** (not 2 generic tools): one MCP tool per
   endpoint / workflow, each with its own input/output schema — best agent
   ergonomics, enabled by the registry.
3. **Shared executor** — extract "run capability X with inputs Y" once; both the
   MCP server **and** Vortex's run-path call it. This folds in Vortex
   **Increment 3** (transform + endpoint-chain execution): build execution once.
4. **Two registry sources:**
   - **Live models → reuse the existing Delta registry** (`model_deployments` ⋈
     `models`) for the list + availability + endpoint name. No hand-curation of
     which endpoints exist.
   - **Prebuilt workflows → build a registry** (none exists today; `batch_models`
     is only partial). New/extended Delta table is the source for workflow tools
     + Vortex prebuilt-workflow nodes.

## Architecture

```
            ┌──────────────────────── Capability Registry (source of truth) ────────────────────────┐
            │  Live models:  model_deployments ⋈ models  (+ thin typed-I/O overlay per endpoint)     │
            │  Workflows:    NEW prebuilt_workflows table (kind, job/chain, typed I/O, params, …)    │
            └───────────────┬───────────────────────────────────────────────┬───────────────────────┘
                            │ projects to                                    │ projects to
                  ┌─────────▼─────────┐                          ┌───────────▼───────────┐
                  │  MCP server tools │  (typed tool/capability) │   Vortex catalog       │
                  │  list_capabilities│                          │   (palette nodes)      │
                  └─────────┬─────────┘                          └───────────┬───────────┘
                            │ both call                                      │ run-path calls
                            └──────────────► Shared Executor ◄───────────────┘
                                  (endpoint query | job dispatch+poll | endpoint-chain | transform)
```

### 1. Shared capability registry
- **Promote `app/services/ai_canvas_registry.py` → a module-neutral `capabilities`
  module** (it is currently Vortex-owned). Keep the `NodeType`-style model:
  `id, kind (endpoint|job|endpoint_chain|transform), typed input/output ports,
  params, description, module, executor handle`.
- **Live models:** the *list + availability + endpoint name* come from
  `get_deployed_models()` / `model_deployments ⋈ models` (already used by
  `get_endpoint_name_for_uc_model`). The *typed I/O contract* (sequence/pdb/
  smiles/json/…) is a **thin curated overlay** keyed by UC short name (today's
  `_ENDPOINT_NODES`); endpoints with no overlay fall back to a **generic
  single-in/out** tool (today's `_generic_endpoint_node`). No change to how
  endpoints are discovered — we reuse the Delta registry.
- **Prebuilt workflows:** introduce a **`prebuilt_workflows` Delta registry**
  (evolve `batch_models`, which only has job_name/display). Columns ≈
  `workflow_key, label, kind (databricks_job|endpoint_chain), module, job_name |
  chain_handler, inputs_json (typed ports), outputs_json, params_json,
  description, is_active`. Each module **registers its workflows at deploy time**
  (the way models register), replacing the hand-curated `_WORKFLOW_NODES` /
  `_CHAIN_NODES`. This is the "manage this better" the user asked for earlier.

### 2. Shared executor (folds Vortex Increment 3)
A single `execute_capability(cap, inputs, params, user) -> result` used by both
the MCP server and the Vortex orchestrator:
- **endpoint** → `WorkspaceClient().serving_endpoints.query(...)` (app SP).
- **databricks_job** → `jobs.run_now(...)`, return run id; **async** — companion
  `get_run_status` / `get_run_result` (mirrors Vortex's existing polling).
- **endpoint_chain** (Protein Design, ADMET Screen) → reuse the app orchestration
  (`services/protein_design`, `services/admet_safety`).
- **transform** (read_text_file / extract_field / field_mapper / …) →
  deterministic reshape ops.
Vortex's orchestrator notebook becomes a thin interpreter that calls this layer;
the MCP tool handlers call the same functions in-process.

### 3. MCP server (generated)
- **`mcp-genesis-workbench`** Databricks App: streamable HTTP at `/mcp`, port
  8000, name `mcp-*` (AI-Playground discoverable). Build with FastMCP.
- **Tools generated from the registry:** one typed tool per capability —
  `endpoint.<short>`, `workflow.<key>` — each with input schema from the typed
  ports/params and `outputSchema`/`annotations` carrying the GWB port dtypes
  (so a future Vortex-on-MCP convergence is possible). Plus `list_capabilities`
  (discovery), and `get_run_status` / `get_run_result` for async workflows.
- **Reuse:** the MCP app vendors the core backend (`app/services/*` + the
  `genesis_workbench` wheel) — same vendoring pattern as the core app's
  `backend/lib/`. Service-layer in-process reuse → no cross-app auth.
- **Auth:** app SP for endpoint/job calls (OBO lacks model-serving scope);
  `X-Forwarded-*` for per-user attribution. The repo's **multi-app grant
  plumbing already exists** (`variables.yml` `app_names`,
  `grant_app_permissions.py` loops over apps) — pass
  `app_names=genesis-workbench:mcp-genesis-workbench`.

## Phasing (spike-first)

- **P0 — endpoints spike.** Promote the registry; build the shared executor for
  **endpoints**; stand up `mcp-genesis-workbench` with `list_capabilities` +
  endpoint tools generated from the existing Delta registry. Connect from AI
  Playground; confirm a real endpoint round-trip. Grant via the multi-app job.
- **P1 — workflows registry + tools.** Build the `prebuilt_workflows` table +
  per-module registration; generate workflow tools (dispatch + status/result);
  this is where the **shared executor finishes Vortex Increment 3** (endpoint-
  chains + transforms execute for real, in both surfaces).
- **P2 — (optional) converge Vortex onto MCP.** Once port/dtype annotations on
  MCP tools are proven, Vortex can build its catalog from `tools/list` instead of
  the registry directly. Only if it earns its keep.

## Reuse / references
- Endpoint registry + names: `genesis_workbench/models.py::get_deployed_models`,
  `get_endpoint_name_for_uc_model`, tables `model_deployments`, `models`.
- Workflow seed: `batch_models` + `register_batch_model`, settings `*_job_id`.
- Typed I/O overlay + executors: today's `app/services/ai_canvas_registry.py`,
  `app/services/ai_canvas.py` (`_exec_descriptor`), `services/protein_design.py`,
  `services/admet_safety.py`, `services/enzyme_optimization.py`, `services/genomics.py`.
- Hosting + grants: `modules/core/resources/app.yml`, `app/app.yml`,
  `notebooks/grant_app_permissions.py`, `variables.yml` (`app_names`), `update.sh`.
- MCP hosting on Databricks Apps: docs.databricks.com/.../generative-ai/mcp/custom-mcp.

## Open questions to settle at build time
- **Workflow registry home:** extend `batch_models` vs a fresh `prebuilt_workflows`
  table (leaning fresh table — clean typed schema).
- **Async UX in MCP:** dispatch+poll tools vs a single blocking tool with a
  timeout for short jobs.
- **Endpoint typed-I/O overlay** location: keep curated in code vs a metadata
  column on the model registry (start curated; revisit).
- **Per-user vs SP attribution** for workflow runs invoked via MCP (X-Forwarded
  present in-app; confirm it flows for all consumer types).

## Out of scope (for now)
- Vortex-on-MCP convergence (P2, optional).
- Multiple MCP servers (start with one app; split only if ops needs it).
