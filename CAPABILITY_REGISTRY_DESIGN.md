# Capability Registry — single source of truth for nodes/params

**Status:** Proposed (design). **Author:** Claude Code, 2026-06-11.

## Problem

A node/capability definition (its inputs, outputs, and **params — including valid values**) is authored in **three** places today, and the consumers don't read the same one:

| Definition lives in | Read by | Has valid-values (`options`)? | Has ranges (`min/max`)? |
|---|---|---|---|
| `app/.../ai_canvas_registry.py` (`ParamField`) | Vortex UI + AI generator | yes | no |
| `mcp_app/scripts/seed_prebuilt_workflows.py` (hardcoded `WORKFLOWS`) → `prebuilt_workflows` Delta | MCP server + executor (via wheel `list_capabilities`) | yes | no |
| endpoint contracts / hardcoded transforms (wheel) | executor + MCP | partial | no |

Consequences:
- The **AI generator only sees param *names***, not allowed values → it invented `strategy:"guided"` (valid: `resample|noop`) and crashed the enzyme job. Numeric params are picked blind → over-strict `qed_min` produced an empty `top_k` (the `alphafold_dock_tox_screen` failure).
- The same definitions drift across the three copies.

## Goal

**One catalog table is the single source of truth for every capability** — internal endpoints, internal workflows (jobs/chains), transforms, **and (future) external MCP tools** — with the **full param contract** (type, default, **options/enum**, **min/max**, required, help). Vortex, the AI generator, and the MCP server all read the same catalog. Adding/changing a node is one edit in one place; the AI sees exactly what MCP sees and what the UI renders.

### Why this unlocks external MCP servers
The strategic driver: GWB should be able to **consume** third-party MCP servers, not just expose its own. An external MCP tool already advertises its params as **JSON Schema** (`enum`, `minimum`/`maximum`, `type`, `default`) via `tools/list`. If our catalog's param contract is JSON-Schema-shaped, ingesting an external tool is a **direct mapping** — register the server, read its `tools/list`, write one catalog row per tool. From then on it's just another node in Vortex / the generator / our MCP surface, dispatched by a new `mcp` kind. A uniform, source-agnostic catalog is the prerequisite.

---

## Design

### 1. One param model + JSON-Schema emitter (in the wheel)

Promote the param model to the shared core (`genesis_workbench/capabilities.py`) and enrich it once. Everything (UI, generator, MCP, executor) imports this; delete the app-side `ParamField`.

```python
@dataclass
class Param:
    name: str
    type: str = "string"            # string | int | float | bool | select | text
    default: object | None = None
    options: list[str] = field(default_factory=list)   # enum (valid values)
    minimum: float | None = None    # numeric lower bound (None = unbounded)
    maximum: float | None = None    # numeric upper bound
    required: bool = False
    label: str = ""
    help: str = ""

def param_schema(p: Param) -> dict:
    """Canonical JSON-Schema property — the single descriptor every consumer derives from."""
    t = {"int": "integer", "float": "number", "bool": "boolean"}.get(p.type, "string")
    s: dict = {"type": t}
    if p.options:        s["enum"] = p.options
    if p.default is not None: s["default"] = p.default
    if p.minimum is not None:  s["minimum"] = p.minimum
    if p.maximum is not None:  s["maximum"] = p.maximum
    if p.help:           s["description"] = p.help
    return s
```

All new fields are optional/defaulted, so existing `Param(**p)` / positional construction keeps working.

### 2. The catalog table (single source of truth)

Extend the existing `prebuilt_workflows` Delta into a general **`capabilities`** table (rename or superset). One row per capability:

```
capability_id   STRING   -- e.g. "workflow:enzyme_optimization", "endpoint:chemprop_clintox", "mcp:acme/dock"
kind            STRING   -- endpoint | job | chain | transform | mcp
label           STRING
module          STRING
source          STRING   -- "builtin" | "mcp:<server_id>"   (provenance)
inputs_json     STRING   -- [{name,dtype,label}]
outputs_json    STRING   -- [{name,dtype,label}]
params_json     STRING   -- [{name,type,default,options,minimum,maximum,required,label,help}]  ← min/max added
dispatch_json   STRING   -- kind-specific: {endpoint_name} | {job_name} | {chain_id} | {op} | {server_id, remote_tool}
description     STRING
is_active       BOOLEAN
```

`params_json` is the durable param contract; `dispatch_json` tells the executor how to run it.

### 3. Read path — everyone reads the table, merged with live availability

`genesis_workbench.capabilities.list_capabilities()` (the wheel, already MCP's source) becomes the single reader:
- Load rows from the `capabilities` table → `Capability`/`Param` objects (with options + min/max).
- **Merge live availability**: which endpoints/jobs actually exist this workspace is still resolved at runtime (Serving/Jobs API) and stamped onto `available` — definitions are static, deployment is live. (The wheel already does this split.)

Then:
- **MCP server** — already calls `list_capabilities()`. Add: emit each param via `param_schema()` so the tool `inputSchema` carries `enum`/`minimum`/`maximum` (annotate handler params `Literal[...]` for enums, `Annotated[..., Field(ge=,le=)]` for ranges). Standard discoverability, zero client code.
- **Vortex catalog (`/ai_canvas/catalog`)** — switch `build_catalog()` to read from `list_capabilities()` instead of the in-code `ai_canvas_registry`. UI renders dropdowns (options) and number min/max from the same rows.
- **AI generator** — `_catalog_prompt_lines` renders the contract inline: `strategy(enum: resample|noop = resample)`, `num_iterations(int 1..50 = 10)`, `qed_min(float 0..1 = 0.5)`, required markers. Same data, now visible to the model.

### 4. Write path — one authoring source publishes to the table

Author the builtin catalog in **one** place (a single Python module in the wheel, git-reviewable) and **publish to the table at deploy** (replaces the hardcoded `WORKFLOWS` in the seed script). For external MCP servers, a **registration step**: connect → `tools/list` → map each tool's JSON Schema to a `params_json` row (`enum→options`, `minimum/maximum→min/max`) → upsert with `kind="mcp"`, `source="mcp:<server_id>"`, `dispatch_json={server_id, remote_tool}`.

### 5. Dispatch — executor gains an `mcp` branch

`execute_capability` dispatches by `kind`: endpoint→serving query, job→`jobs.run_now`, chain→chain fn, transform→transform fn, **`mcp`→ MCP client call to the external server's tool** (relay inputs/params, return result). External tools then run in Vortex pipelines and via our MCP surface identically to builtins.

### 6. Validation — one contract-driven gate

`validate_params(capability, params)` in the wheel, called inside `execute_capability` (covers UI/Vortex/MCP/external uniformly):
- **enum** not in `options` → **reject** (no sensible nearest): `strategy:"guided"` → error.
- **numeric** out of `[minimum, maximum]` → **clamp to nearest bound + log a coercion note** (forgiving for long jobs; never silent). `qed_min>1` style hard-invalids can be `strict` per-param → reject.
- coerce types; check `required` present.

This **subsumes the one-off patches** (the `strategy` and motif `"A:50"` band-aids become contract rules, not bespoke code).

---

## Sequencing

1. **Enrich `Param`** (min/max/label/required/help) + `param_schema()` + `validate_params()` in the wheel; wire `validate_params` into `execute_capability`. *(Stops the crash class immediately, even before the table move.)*
2. **Generator + MCP read the contract** — render options/ranges in the prompt; `Literal`/`Field` in MCP tool signatures.
3. **Table becomes source of truth** — extend schema (`min/max`, `source`, `dispatch_json`), one publisher, `list_capabilities()` reads it + live-merge.
4. **Vortex reads the table** — `build_catalog()` off `list_capabilities()`; delete `ai_canvas_registry` duplication.
5. **External MCP ingestion** — registration (`tools/list`→rows) + executor `mcp` dispatch branch + auth.

Steps 1–2 are the quick win (kill the invalid-value crashes); 3–4 remove the drift; 5 delivers the external-MCP capability.

## Open questions
- **External-MCP auth/identity** — how GWB authenticates to a third-party server, and how that composes with the interim "accessor-list" control (Security review Vuln 1). Per-caller authorization becomes more pressing once external tools are callable.
- **Availability for external tools** — health-check the external server; mark `available=false` on failure so a dead server doesn't break catalog load (failure isolation).
- **Authoring location** — builtin catalog in the wheel (code→publish) vs. authored directly in the table. Recommend code→publish for git review.
- **clamp vs strict** default per numeric param.
