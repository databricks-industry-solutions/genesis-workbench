"""Vortex (ai_canvas) — backend service.

V1 surface implemented here: the node **catalog** builder. Later increments add
the LLM graph generator, workflow persistence, the run dispatcher, and MLflow
result loaders (all modelled on `services/enzyme_optimization.py`).
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.service.workspace import ImportFormat
from mlflow.tracking import MlflowClient

from genesis_workbench.models import (
    ModelCategory,
    get_batch_models,
    get_deployed_models,
    get_endpoint_name_for_uc_model,
    set_mlflow_experiment,
)
from genesis_workbench.workbench import (
    UserInfo,
    execute_non_select_query,
    execute_select_query,
    get_user_settings,
)

from app.services.ai_canvas_registry import (
    CURATED_BY_ENDPOINT,
    CURATED_BY_TYPE,
    CURATED_NODES,
    NodeCategory,
    NodeType,
    Port,
    PortType,
)
from app.services.databricks_links import job_run_url
from app.services.endpoints import DISPLAY_TO_UC, get_endpoint_name
from app.services.workbench import get_job_id

logger = logging.getLogger(__name__)

_MODULES = ("single_cell", "large_molecule", "small_molecule", "genomics")

EXPERIMENT_TAG = "gwb_ai_canvas"          # single shared experiment for all runs
FEATURE_TAG = "ai_canvas"
ORCHESTRATOR_JOB_SETTING = "run_ai_canvas_workflow_job_id"
ORCHESTRATOR_JOB_NAME = "run_ai_canvas_workflow_gwb"
VOLUME_DIR_NAME = "ai_canvas"             # /Volumes/<cat>/<schema>/ai_canvas/<uuid>/graph.json


def _serialize_port(p: Port) -> dict:
    return {"name": p.name, "dtype": str(p.dtype), "label": p.label or p.name}


def _serialize_node(node: NodeType, available: bool) -> dict:
    d = {
        "type": node.type,
        "label": node.label,
        "category": str(node.category),
        "kind": node.kind,            # "" | databricks_job | endpoint_chain | mcp
        "chain": node.chain,          # endpoint_chain handler id (else None)
        "description": node.description,
        "module": node.module,
        "available": available,
        "inputs": [_serialize_port(p) for p in node.inputs],
        "outputs": [_serialize_port(p) for p in node.outputs],
        "params": [
            {
                "name": f.name,
                "label": f.label,
                "type": f.type,
                "default": f.default,
                "options": list(f.options),
                "required": f.required,
                "help": f.help,
            }
            for f in node.params
        ],
    }
    return d


def _uc_short(uc_name: str) -> str:
    """Extract the model short name from a deployed model's uc_name
    (`catalog.schema.<short>/<version>` → `<short>`). The short name — not the
    display name — is the stable join key between curated nodes (via
    DISPLAY_TO_UC) and live deployments (registration sets display names freely,
    e.g. "Boltz-1", "NetSolP-1.0 …")."""
    return uc_name.split("/")[0].split(".")[-1].strip().lower()


def _deployed_endpoints() -> tuple[set[str], dict[str, str]]:
    """Return (set of deployed UC short names, short -> friendly display name)."""
    shorts: set[str] = set()
    display: dict[str, str] = {}
    for module in _MODULES:
        if module == "genomics":
            continue  # genomics has no real-time endpoints
        try:
            df = get_deployed_models(ModelCategory(module))
        except Exception as e:  # noqa: BLE001 — degrade gracefully if DB is down
            logger.warning("catalog: deployed lookup failed for %s: %s", module, e)
            continue
        for _, r in df.iterrows():
            short = _uc_short(str(r["uc_name"]))
            shorts.add(short)
            display[short] = str(r["model_display_name"])
    return shorts, display


def _batch_job_names() -> set[str]:
    names: set[str] = set()
    for module in _MODULES:
        try:
            df = get_batch_models(module)
        except Exception as e:  # noqa: BLE001
            logger.warning("catalog: batch lookup failed for %s: %s", module, e)
            continue
        for _, r in df.iterrows():
            names.add(str(r["job_name"]))
    return names


# Prebuilt-workflow jobs (genomics, fine-tunes, KERMT, …) aren't all registered
# as batch_models — they're plain Jobs. List every job name once (cached for the
# app's lifetime; deploys restart the app) so availability reflects what's
# actually deployed without a per-node lookup.
_all_job_names_cache: set[str] | None = None


def _all_job_names() -> set[str]:
    global _all_job_names_cache
    if _all_job_names_cache is not None:
        return _all_job_names_cache
    names: set[str] = set()
    try:
        for j in WorkspaceClient().jobs.list():
            settings = getattr(j, "settings", None)
            if settings and settings.name:
                names.add(str(settings.name))
    except Exception as e:  # noqa: BLE001 — degrade gracefully
        logger.warning("catalog: jobs list failed: %s", e)
    _all_job_names_cache = names
    return names


def _generic_endpoint_node(short: str, display_name: str) -> dict:
    """A deployed endpoint with no curated schema — exposed as a permissive
    single-in / single-out JSON node so it's still composable. Keyed on the UC
    short name so the orchestrator can resolve it via the deployments table."""
    node = NodeType(
        type=f"endpoint::{short}",
        label=display_name,
        category=NodeCategory.ENDPOINT,
        endpoint_display_name=display_name,
        description=f"Deployed endpoint '{display_name}' (no curated schema).",
        inputs=[Port("input", PortType.ANY)],
        outputs=[Port("output", PortType.JSON)],
    )
    return _serialize_node(node, available=True)


def build_catalog() -> list[dict]:
    """Curated node types merged with live deployment/job availability.

    Availability is keyed on the **UC short name** (esmfold, netsolp_v1, …),
    not the display name — registration sets display names freely, so a
    display-name match would wrongly mark deployed endpoints unavailable.

    - Curated endpoint nodes: available iff their DISPLAY_TO_UC short is deployed.
    - Curated batch nodes: available iff their job is in batch_models.
    - IO nodes: always available.
    - Deployed endpoints with no curated entry: appended as generic nodes.
    """
    deployed_shorts, short_to_display = _deployed_endpoints()
    batch_jobs = _batch_job_names() | _all_job_names()

    catalog: list[dict] = []
    curated_shorts: set[str] = set()
    for node in CURATED_NODES:
        if node.category == NodeCategory.ENDPOINT:
            short = DISPLAY_TO_UC.get(node.endpoint_display_name or "")
            if short:
                curated_shorts.add(short)
            available = bool(short) and short in deployed_shorts
        elif node.category == NodeCategory.BATCH:
            if node.kind == "endpoint_chain":
                # A composite chain is runnable iff every endpoint it needs is deployed.
                available = all(
                    (DISPLAY_TO_UC.get(name) or "") in deployed_shorts
                    for name in node.requires_endpoints
                )
            else:  # databricks_job
                available = node.job_name in batch_jobs
        else:  # IO + TRANSFORM — always available (no external dependency)
            available = True
        catalog.append(_serialize_node(node, available))

    # Deployed endpoints we have no curated schema for → generic nodes.
    for short in sorted(deployed_shorts - curated_shorts):
        catalog.append(_generic_endpoint_node(short, short_to_display.get(short, short)))

    logger.info(
        "ai_canvas catalog: %d nodes (%d deployed endpoints, %d batch jobs)",
        len(catalog), len(deployed_shorts), len(batch_jobs),
    )
    return catalog


# ─── AI graph generation ─────────────────────────────────────────────────────


class GraphGenerationError(RuntimeError):
    """Raised when the LLM response can't be parsed into a valid graph."""


def _catalog_prompt_lines(catalog: list[dict]) -> str:
    """Compact one-line-per-node-type description for the LLM system prompt."""
    lines: list[str] = []
    for n in catalog:
        if not n["available"]:
            continue  # only offer nodes the user can actually run
        if not _offerable(n):
            continue  # skip nodes that can't run via the generic orchestrator path
        ins = ", ".join(f"{p['name']}:{p['dtype']}" for p in n["inputs"]) or "—"
        outs = ", ".join(f"{p['name']}:{p['dtype']}" for p in n["outputs"]) or "—"
        params = ", ".join(p["name"] for p in n["params"]) or "—"
        lines.append(
            f'- type="{n["type"]}" ({n["category"]}): {n["label"]}. '
            f"in[{ins}] out[{outs}] params[{params}]"
        )
    return "\n".join(lines)


_SYSTEM_PROMPT_TEMPLATE = """You are a workflow architect for Genesis Workbench, a bioinformatics platform.
Given a user's goal, design a directed workflow graph using ONLY the node types listed below.

Node types (connect an output port to an input port only when their dtypes match; `any` matches anything):
{catalog}

Rules:
- PREFER building the workflow around at least one **Prebuilt Workflow** (a node whose category is `batch` — e.g. Guided Enzyme Optimization, Guided Molecule Optimization, Protein Design, ADMET Screen, AlphaFold, GWAS). These are the platform's headline multi-step capabilities; make one the centerpiece whenever the goal allows.
- Build a complete, realistic pipeline, not a single node: an input → (optional transforms) → a Prebuilt Workflow (and/or a few chained endpoints) → optional follow-up analysis → output_sink. Aim for roughly 4–7 nodes when the goal supports it.
- Start with an input node (volume_input, delta_input, or text_input) that provides the initial data.
- End with an output_sink node that collects the final result.
- Use each node's `type` EXACTLY as written above. Do not invent node types.
- Give every node a unique `id` (e.g. "n1", "n2") and a short human `label`.
- Only add edges between compatible ports; set sourceHandle to the source node's output port name and targetHandle to the target node's input port name. If two ports' dtypes don't match, insert a transform node between them.
- Protein Design (type="protein_design") redesigns a marked region: its `sequence` input MUST contain that region wrapped in square brackets, e.g. "MKT[AYIAK]QRQ". If you feed it a text_input, put the brackets in the value. If the goal has no region to redesign, do NOT use Protein Design.
- Fill `params` with sensible values where helpful; otherwise use {{}}.

Also include a `plan`: a list of 3-5 very short present-tense bullet strings narrating your reasoning as you design — which Prebuilt Workflow you center on and why, then each pipeline step (e.g. "Center on Guided Enzyme Optimization for the substrate", "Fold the top design with ESMFold", "Collect the best candidate"). Keep each bullet under ~10 words.

Also include a `name`: a short Title Case workflow name (2-5 words) summarizing what it does, e.g. "Enzyme Optimization + Fold" or "ADMET Screen & Optimize". No punctuation beyond + & -.

Respond with ONLY a JSON object, no prose, no markdown fences, in exactly this shape:
{{"name":"<short title>","plan":["<thought>","<thought>"],
 "nodes":[{{"id":"n1","type":"<type>","label":"<label>","params":{{}}}}],
 "edges":[{{"source":"n1","target":"n2","sourceHandle":"<out_port>","targetHandle":"<in_port>"}}]}}"""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (tolerates code fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise GraphGenerationError("LLM did not return a JSON object.")
        candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Cheap repair for the most common LLM slip: trailing commas before } or ].
        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise GraphGenerationError(f"Could not parse graph JSON: {e}") from e


def _auto_layout(nodes: list[dict], edges: list[dict]) -> None:
    """Assign left-to-right positions by longest-path depth (in place)."""
    by_id = {n["id"]: n for n in nodes}
    succ: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    indeg: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        if e["source"] in by_id and e["target"] in by_id:
            succ[e["source"]].append(e["target"])
            indeg[e["target"]] += 1

    # Longest-path depth via Kahn's topological order; cycles fall back to 0.
    depth = {nid: 0 for nid in by_id}
    queue = [nid for nid, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        nid = queue.pop(0)
        seen += 1
        for s in succ[nid]:
            depth[s] = max(depth[s], depth[nid] + 1)
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)

    per_depth: dict[int, int] = {}
    for n in nodes:
        d = depth[n["id"]]
        row = per_depth.get(d, 0)
        per_depth[d] = row + 1
        n["position"] = {"x": d * 240, "y": row * 130 + 20}


def _catalog_ctx() -> tuple[set[str], dict]:
    """(valid node types, ports_by_type, port_dtypes) for validating an
    LLM-produced graph. port_dtypes maps type -> ({in_name: dtype}, {out_name: dtype})."""
    catalog = build_catalog()
    valid_types = {n["type"] for n in catalog}
    ports_by_type = {
        n["type"]: ({p["name"] for p in n["inputs"]}, {p["name"] for p in n["outputs"]})
        for n in catalog
    }
    port_dtypes = {
        n["type"]: (
            {p["name"]: p["dtype"] for p in n["inputs"]},
            {p["name"]: p["dtype"] for p in n["outputs"]},
        )
        for n in catalog
    }
    return valid_types, ports_by_type, port_dtypes


# Node types NOT offered to the generator: they can't run via the orchestrator's
# generic endpoint call (need a multi-stage / special payload), so a graph using
# them would fail at runtime. DiffDock (2-stage + ESM embeddings), RFDiffusion /
# Boltz (dict payloads), the single-cell AnnData endpoints, and every generic
# (schema-less) "endpoint::*" node. They remain usable via their UI tabs / chains.
_GENERATOR_EXCLUDE = {
    "diffdock", "rfdiffusion", "boltz", "teddy", "scgpt_perturbation",
    "scgpt_embeddings", "scimilarity_get_embedding", "scimilarity_gene_order",
}


def _offerable(node: dict) -> bool:
    t = node.get("type", "")
    return t not in _GENERATOR_EXCLUDE and not t.startswith("endpoint::")


def _dtypes_compatible(src: str | None, dst: str | None) -> bool:
    """Mirror the frontend portsCompatible: `any` matches anything, equal dtypes
    match, and singular/plural normalize (sequence ~ sequences). Unknown → allow."""
    if not src or not dst or src == "any" or dst == "any" or src == dst:
        return True
    norm = lambda d: d[:-1] if d.endswith("s") else d  # noqa: E731
    return norm(src) == norm(dst)


def _llm_generate_parsed(goal: str, llm_endpoint: str) -> dict:
    """Call the LLM once (with a one-shot stricter retry) and return parsed JSON
    (which may carry `plan`, `nodes`, `edges`)."""
    catalog = build_catalog()
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(catalog=_catalog_prompt_lines(catalog))
    w = WorkspaceClient()

    def _attempt(extra_system: str = "") -> dict:
        response = w.serving_endpoints.query(
            name=llm_endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt + extra_system),
                ChatMessage(role=ChatMessageRole.USER, content=goal),
            ],
            max_tokens=4000,      # headroom so larger graphs don't truncate mid-JSON
            temperature=0.2,      # low temp → fewer JSON formatting slips
        )
        return _extract_json(response.choices[0].message.content)

    try:
        return _attempt()
    except GraphGenerationError:
        # LLMs occasionally drop a comma in a big graph — one stricter retry.
        return _attempt(
            "\n\nIMPORTANT: Respond with ONE strictly-valid JSON object only. Every array/"
            "object element MUST be comma-separated; no trailing commas; no prose; no markdown."
        )


def _validate_graph(
    parsed: dict, valid_types: set[str], ports_by_type: dict, port_dtypes: dict | None = None
) -> dict:
    """Validate + normalize an LLM-produced graph; drop invalid nodes, nodes we
    don't offer the generator, and edges with bad/hallucinated ports or
    dtype-incompatible endpoints — so the graph aligns with the real node contracts."""
    port_dtypes = port_dtypes or {}
    clean_nodes: list[dict] = []
    seen_ids: set[str] = set()
    type_by_id: dict[str, str] = {}
    for n in parsed.get("nodes", []):
        ntype = n.get("type")
        nid = n.get("id")
        if ntype not in valid_types or not nid or nid in seen_ids:
            logger.info("generate_graph: dropping invalid node %s (%s)", nid, ntype)
            continue
        if not _offerable({"type": ntype}):
            logger.info("generate_graph: dropping non-offerable node %s (%s)", nid, ntype)
            continue
        seen_ids.add(nid)
        type_by_id[str(nid)] = ntype
        clean_nodes.append(
            {
                "id": str(nid),
                "type": ntype,
                "label": str(n.get("label") or ntype),
                "params": n.get("params") if isinstance(n.get("params"), dict) else {},
                "position": {"x": 0, "y": 0},
            }
        )

    clean_edges: list[dict] = []
    for e in parsed.get("edges", []):
        s, t = str(e.get("source")), str(e.get("target"))
        if s not in seen_ids or t not in seen_ids:
            continue
        sh, th = e.get("sourceHandle"), e.get("targetHandle")
        s_type, t_type = type_by_id[s], type_by_id[t]
        # Drop handles the node type doesn't actually have (LLM occasionally
        # hallucinates port names); the canvas then leaves them unconnected.
        s_outs = ports_by_type.get(s_type, (set(), set()))[1]
        t_ins = ports_by_type.get(t_type, (set(), set()))[0]
        sh = sh if sh in s_outs else None
        th = th if th in t_ins else None
        # Drop the edge entirely if the two connected ports' dtypes don't line up
        # (e.g. pdb -> smiles) — keeps the wiring aligned with real contracts.
        if sh and th:
            s_dt = port_dtypes.get(s_type, ({}, {}))[1].get(sh)
            t_dt = port_dtypes.get(t_type, ({}, {}))[0].get(th)
            if not _dtypes_compatible(s_dt, t_dt):
                logger.info("generate_graph: dropping dtype-incompatible edge %s.%s(%s) -> %s.%s(%s)",
                            s, sh, s_dt, t, th, t_dt)
                continue
        clean_edges.append(
            {
                "source": str(s),
                "target": str(t),
                "sourceHandle": sh,
                "targetHandle": th,
            }
        )

    if not clean_nodes:
        raise GraphGenerationError(
            "The model didn't produce any valid nodes for that goal. Try rephrasing."
        )

    _auto_layout(clean_nodes, clean_edges)
    return {"nodes": clean_nodes, "edges": clean_edges}


# ─── Self-review: a deterministic linter + an LLM repair pass ────────────────
# dtype validation keeps every *edge* legal, but a graph can still be legal-yet-
# broken: a node with no input can't run, a node whose output feeds nothing is a
# dead branch, and two disconnected sub-flows aren't one pipeline. We catch those
# deterministically, then hand the draft + the findings back to the model to fix
# (generate → review → repair, the way Claude Code self-corrects).

_INPUT_TYPES = {"text_input", "volume_input", "delta_input"}


def _graph_issues(graph: dict, ports_by_type: dict) -> list[str]:
    """Semantic problems an LLM draft commonly has, in human-readable form."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    by_id = {n["id"]: n for n in nodes}
    indeg: dict[str, int] = {n["id"]: 0 for n in nodes}
    outdeg: dict[str, int] = {n["id"]: 0 for n in nodes}
    adj: dict[str, set] = {n["id"]: set() for n in nodes}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in by_id and t in by_id and e.get("targetHandle") and e.get("sourceHandle"):
            outdeg[s] += 1
            indeg[t] += 1
            adj[s].add(t)
            adj[t].add(s)

    def _label(n):
        return n.get("label") or n.get("type")

    issues: list[str] = []
    for n in nodes:
        nid, ntype = n["id"], n["type"]
        n_in, n_out = ports_by_type.get(ntype, (set(), set()))
        # has input ports but nothing feeds them → can't run
        if n_in and ntype not in _INPUT_TYPES and indeg[nid] == 0:
            issues.append(
                f'"{_label(n)}" ({ntype}) has no incoming connection — it has inputs '
                f"but nothing feeds them, so it cannot run."
            )
        # produces output but nothing consumes it (and it's not the sink) → dead end
        if n_out and ntype != "output_sink" and outdeg[nid] == 0:
            issues.append(
                f'"{_label(n)}" ({ntype}) output is not consumed — it dead-ends. '
                f"Wire it onward (e.g. into the next step or an output_sink) or remove it."
            )

    # fragmentation: more than one weakly-connected component
    if len(nodes) > 1:
        unseen = set(by_id)
        comps = 0
        while unseen:
            comps += 1
            stack = [next(iter(unseen))]
            while stack:
                cur = stack.pop()
                if cur in unseen:
                    unseen.discard(cur)
                    stack.extend(adj[cur] & unseen)
        if comps > 1:
            issues.append(
                f"The graph is split into {comps} disconnected sub-flows — they should "
                f"connect into a single pipeline ending at one output_sink."
            )
    return issues


_REVIEW_PROMPT_TEMPLATE = """You are reviewing a draft Genesis Workbench workflow graph for the user's goal.
A linter found the problems listed below. Fix them while preserving the user's intent
and the overall pipeline shape. You may add/rewire edges, insert transform nodes to bridge
dtype gaps, or remove a node that is genuinely redundant.

Node types (connect an output port to an input port only when dtypes match; `any` matches anything):
{catalog}

Hard requirements for the corrected graph:
- Every node except input nodes (text_input/volume_input/delta_input) MUST have at least one incoming edge feeding a real input port.
- Every node except output_sink MUST have its output consumed by a downstream node.
- The whole graph MUST be one connected pipeline that ends at an output_sink.
- Use node `type` values EXACTLY as listed; keep existing node ids where possible.
- Set sourceHandle to the source node's output port name and targetHandle to the target node's input port name; only connect dtype-compatible ports.

Respond with ONLY a JSON object, no prose, no markdown fences, in exactly this shape:
{{"review":["<short note on what you fixed>","<...>"],
 "nodes":[{{"id":"n1","type":"<type>","label":"<label>","params":{{}}}}],
 "edges":[{{"source":"n1","target":"n2","sourceHandle":"<out_port>","targetHandle":"<in_port>"}}]}}"""


def _llm_repair(goal: str, graph: dict, issues: list[str], llm_endpoint: str) -> dict:
    """One LLM pass that returns a corrected graph for the listed issues."""
    catalog = build_catalog()
    system_prompt = _REVIEW_PROMPT_TEMPLATE.format(catalog=_catalog_prompt_lines(catalog))
    draft = {
        "nodes": [{"id": n["id"], "type": n["type"], "label": n.get("label"),
                   "params": n.get("params", {})} for n in graph.get("nodes", [])],
        "edges": graph.get("edges", []),
    }
    user = (
        f"GOAL:\n{goal}\n\nDRAFT GRAPH:\n{json.dumps(draft)}\n\n"
        f"PROBLEMS TO FIX:\n- " + "\n- ".join(issues)
    )
    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=llm_endpoint,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER, content=user),
        ],
        max_tokens=4000,
        temperature=0.1,
    )
    return _extract_json(response.choices[0].message.content)


def _review_graph(
    goal: str, graph: dict, llm_endpoint: str,
    valid_types: set[str], ports_by_type: dict, port_dtypes: dict,
    max_rounds: int = 2,
) -> tuple[dict, list[str]]:
    """Lint the draft; if broken, repair via the LLM (up to max_rounds), keeping
    whichever graph has the fewest issues. Returns (graph, review-note bullets).
    Fail-soft: any error leaves the draft untouched."""
    issues = _graph_issues(graph, ports_by_type)
    if not issues:
        return graph, ["Reviewed the draft — every node is wired and the flow reaches the output."]

    notes = ["Reviewing the draft for dangling nodes and dead-end connections…"]
    notes.append("Found " + (f"{len(issues)} issue(s): " if len(issues) > 1 else "") + issues[0])

    for _ in range(max_rounds):
        try:
            repaired = _validate_graph(
                _llm_repair(goal, graph, issues, llm_endpoint),
                valid_types, ports_by_type, port_dtypes,
            )
        except Exception as e:  # noqa: BLE001 — never let review break generation
            logger.info("generate_graph: review repair failed, keeping draft (%s)", e)
            break
        new_issues = _graph_issues(repaired, ports_by_type)
        if len(new_issues) < len(issues):
            graph, issues = repaired, new_issues
            notes.append("Rewired the graph to connect the loose ends.")
        else:
            break  # repair didn't help — stop and keep the better graph
        if not issues:
            notes.append("Re-checked — the workflow is now fully connected.")
            break

    if issues:
        notes.append(
            f"{len(issues)} connection issue(s) may remain — flagged on the canvas for you to confirm."
        )
    return graph, notes


def generate_graph(goal: str, llm_endpoint: str) -> dict:
    """Goal → validated, self-reviewed canvas graph (only catalog node types; fail-soft)."""
    valid_types, ports_by_type, port_dtypes = _catalog_ctx()
    graph = _validate_graph(_llm_generate_parsed(goal, llm_endpoint), valid_types, ports_by_type, port_dtypes)
    graph, _ = _review_graph(goal, graph, llm_endpoint, valid_types, ports_by_type, port_dtypes)
    return graph


def generate_plan_and_graph(goal: str, llm_endpoint: str) -> tuple[list[str], dict, str, list[str]]:
    """Goal → (plan bullets, validated+reviewed graph, short name, review notes).
    The plan and the review notes both surface in the streamed 'thoughts' UX, so
    the user watches the model design and then self-correct the draft."""
    valid_types, ports_by_type, port_dtypes = _catalog_ctx()
    parsed = _llm_generate_parsed(goal, llm_endpoint)
    plan = [str(b).strip() for b in (parsed.get("plan") or []) if str(b).strip()][:6]
    name = str(parsed.get("name") or "").strip()[:60]
    graph = _validate_graph(parsed, valid_types, ports_by_type, port_dtypes)
    graph, review = _review_graph(goal, graph, llm_endpoint, valid_types, ports_by_type, port_dtypes)
    return plan, graph, name, review


# ─── AI transform suggestion (auto-bridge incompatible connections) ──────────


def _transform_catalog_lines() -> str:
    """One line per transform node-type for the suggestion prompt."""
    lines: list[str] = []
    for n in CURATED_NODES:
        if n.category != NodeCategory.TRANSFORM:
            continue
        ins = ", ".join(f"{p.name}:{p.dtype}" for p in n.inputs) or "—"
        outs = ", ".join(f"{p.name}:{p.dtype}" for p in n.outputs) or "—"
        params = ", ".join(
            f.name + (f"({'/'.join(f.options)})" if f.options else "") for f in n.params
        ) or "—"
        lines.append(f'- type="{n.type}": {n.label}. in[{ins}] out[{outs}] params[{params}]')
    return "\n".join(lines)


_TRANSFORM_SUGGEST_PROMPT = """You wire nodes in a bioinformatics workflow canvas. The user connected an output of \
dtype "{src}" (from "{src_label}") into an input of dtype "{dst}" (on "{dst_label}"), but the dtypes don't \
directly match. Pick exactly ONE transform from the list that converts a "{src}" value into a "{dst}" value, \
and fill its params sensibly for that conversion. If NONE can bridge it, return null for type.

Transforms:
{transforms}

Respond with ONLY a JSON object, no prose, no markdown:
{{"type": "<transform type, or null>", "params": {{}}}}"""


def suggest_transform(
    *, source_dtype: str, target_dtype: str,
    source_label: str = "", target_label: str = "", llm_endpoint: str,
) -> dict | None:
    """Ask the LLM to pick a single transform node (+params) that bridges an
    incompatible source-output → target-input connection. Returns
    {type, label, params} or None if nothing fits / the call fails."""
    transforms = {n.type: n for n in CURATED_NODES if n.category == NodeCategory.TRANSFORM}
    prompt = _TRANSFORM_SUGGEST_PROMPT.format(
        src=source_dtype, dst=target_dtype,
        src_label=source_label or source_dtype, dst_label=target_label or target_dtype,
        transforms=_transform_catalog_lines(),
    )
    try:
        response = WorkspaceClient().serving_endpoints.query(
            name=llm_endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=prompt),
                ChatMessage(role=ChatMessageRole.USER, content="Choose the transform."),
            ],
            max_tokens=300,
        )
        parsed = _extract_json(response.choices[0].message.content)
    except Exception as e:  # noqa: BLE001 — fail-soft to the mismatch message
        logger.info("suggest_transform failed (%s→%s): %s", source_dtype, target_dtype, e)
        return None

    ttype = parsed.get("type")
    if not ttype or ttype not in transforms:
        return None
    nt = transforms[ttype]
    valid = {f.name for f in nt.params}
    suggested = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}
    merged = {f.name: f.default for f in nt.params if f.default is not None}
    merged.update({k: v for k, v in suggested.items() if k in valid})
    return {"type": nt.type, "label": nt.label, "params": merged}


# ─── Workflow persistence (ai_canvas_workflows table) ────────────────────────


def _sql_str(val: str | None) -> str:
    """SQL string literal with single-quote escaping; NULL for None."""
    if val is None:
        return "NULL"
    return "'" + str(val).replace("\\", "\\\\").replace("'", "''") + "'"


def _table() -> str:
    return f"{os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.ai_canvas_workflows"


def _safe_filename(name: str) -> str:
    """A filesystem-safe slug for the workflow JSON filename."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()) or "workflow"
    return slug[:120]


def _save_graph_to_experiment_folder(user_email: str, name: str, graph: dict) -> None:
    """Best-effort: write the graph JSON to the user's MLflow experiment folder
    (from Profile settings) under a new `/ai_canvas` subfolder, so saved
    workflows are visible/portable alongside the user's experiments. Never
    raises — a workspace-write hiccup must not fail the (table-backed) save."""
    try:
        folder = get_user_settings(user_email=user_email).get("mlflow_experiment_folder")
        if not folder:
            return
        base = f"/Workspace/Users/{user_email}/{folder}/ai_canvas"
        w = WorkspaceClient()
        w.workspace.mkdirs(base)
        w.workspace.upload(
            f"{base}/{_safe_filename(name)}.json",
            io.BytesIO(json.dumps(graph, indent=2).encode("utf-8")),
            format=ImportFormat.AUTO,
            overwrite=True,
        )
    except Exception as e:  # noqa: BLE001 — best-effort mirror of the DB save
        logger.info("ai_canvas: experiment-folder save skipped for %s: %s", user_email, e)


def save_workflow(
    *,
    user_email: str,
    name: str,
    description: str,
    graph: dict,
    workflow_id: int | str | None = None,
) -> int:
    """Insert a new workflow or update an existing one (owned by the user).
    Returns the workflow_id. Graph is stored as JSON text in `graph_json`."""
    graph_json = json.dumps(graph)
    if workflow_id is None:
        workflow_id = time.time_ns()
        execute_non_select_query(
            f"""INSERT INTO {_table()}
                (workflow_id, workflow_name, workflow_description, created_by,
                 created_date, updated_date, graph_json, is_active, deactivated_timestamp)
                VALUES ({workflow_id}, {_sql_str(name)}, {_sql_str(description)},
                        {_sql_str(user_email)}, current_timestamp(), current_timestamp(),
                        {_sql_str(graph_json)}, true, NULL)"""
        )
    else:
        execute_non_select_query(
            f"""UPDATE {_table()} SET
                    workflow_name = {_sql_str(name)},
                    workflow_description = {_sql_str(description)},
                    graph_json = {_sql_str(graph_json)},
                    updated_date = current_timestamp(),
                    is_active = true,
                    deactivated_timestamp = NULL
                WHERE workflow_id = {int(workflow_id)} AND created_by = {_sql_str(user_email)}"""
        )
    # Mirror the graph JSON into the user's MLflow experiment folder (best-effort).
    _save_graph_to_experiment_folder(user_email, name, graph)
    return int(workflow_id)


def list_workflows(user_email: str) -> list[dict]:
    """Saved workflows for a user (no graph payload), newest first."""
    df = execute_select_query(
        f"""SELECT workflow_id, workflow_name, workflow_description, created_date, updated_date
            FROM {_table()}
            WHERE created_by = {_sql_str(user_email)} AND is_active = true
            ORDER BY updated_date DESC"""
    )
    out: list[dict] = []
    for _, r in df.iterrows():
        out.append(
            {
                "workflow_id": str(int(r["workflow_id"])),  # BIGINT → string for JS
                "name": str(r["workflow_name"]),
                "description": str(r["workflow_description"] or ""),
                "updated_date": str(r["updated_date"]),
            }
        )
    return out


def get_workflow(workflow_id: int, user_email: str) -> dict | None:
    """A single saved workflow including its graph, or None if not found."""
    df = execute_select_query(
        f"""SELECT workflow_id, workflow_name, workflow_description, graph_json
            FROM {_table()}
            WHERE workflow_id = {int(workflow_id)} AND created_by = {_sql_str(user_email)}
              AND is_active = true LIMIT 1"""
    )
    if df.empty:
        return None
    r = df.iloc[0]
    try:
        graph = json.loads(r["graph_json"])
    except (TypeError, json.JSONDecodeError):
        graph = {"nodes": [], "edges": []}
    return {
        "workflow_id": str(int(r["workflow_id"])),  # BIGINT → string for JS
        "name": str(r["workflow_name"]),
        "description": str(r["workflow_description"] or ""),
        "graph": graph,
    }


def deactivate_workflow(workflow_id: int, user_email: str) -> None:
    """Soft-delete a workflow (keeps history; hidden from the list)."""
    execute_non_select_query(
        f"""UPDATE {_table()} SET is_active = false, deactivated_timestamp = current_timestamp()
            WHERE workflow_id = {int(workflow_id)} AND created_by = {_sql_str(user_email)}"""
    )


# ─── Run dispatch ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunDispatchResult:
    job_id: int
    job_run_id: int
    mlflow_run_id: str
    experiment_id: str


_job_id_cache: dict[str, int] = {}


def _resolve_orchestrator_job_id(w: WorkspaceClient) -> int:
    """Orchestrator job id from the settings table, falling back to a name
    lookup against the Jobs API. Cached after first resolution."""
    cached = _job_id_cache.get(ORCHESTRATOR_JOB_NAME)
    if cached is not None:
        return cached
    # The settings-table id goes stale on every redeploy (DAB recreates the job
    # with a new id). Trust it only if the job still exists; otherwise fall back
    # to a live name lookup and self-heal. Avoids "Job <id> does not exist" 502s.
    setting = get_job_id(ORCHESTRATOR_JOB_SETTING)
    if setting:
        try:
            w.jobs.get(job_id=int(setting))
            _job_id_cache[ORCHESTRATOR_JOB_NAME] = int(setting)
            return int(setting)
        except Exception as e:  # noqa: BLE001 — stale id; fall through to name lookup
            logger.warning(
                "Orchestrator job id %s from settings is stale (%s); "
                "resolving '%s' by name instead.", setting, e, ORCHESTRATOR_JOB_NAME
            )
    matches = list(w.jobs.list(name=ORCHESTRATOR_JOB_NAME))
    if not matches:
        raise RuntimeError(
            f"Orchestrator job '{ORCHESTRATOR_JOB_NAME}' not found. Redeploy core: "
            "`cd modules/core && ./update.sh <cloud>`."
        )
    _job_id_cache[ORCHESTRATOR_JOB_NAME] = int(matches[0].job_id)
    return _job_id_cache[ORCHESTRATOR_JOB_NAME]


def _resolve_batch_job_id(job_name: str, w: WorkspaceClient) -> int | None:
    cached = _job_id_cache.get(job_name)
    if cached is not None:
        return cached
    matches = list(w.jobs.list(name=job_name))
    if not matches:
        return None
    _job_id_cache[job_name] = int(matches[0].job_id)
    return _job_id_cache[job_name]


def _exec_descriptor(type_key: str, w: WorkspaceClient) -> dict:
    """Resolve a node type to a self-contained execution descriptor the
    orchestrator notebook can run without the registry. Endpoint names / job
    ids are resolved live here (the app owns the registry, not the notebook)."""
    nt: NodeType | None = CURATED_BY_TYPE.get(type_key)

    # Generic deployed endpoint with no curated schema (type "endpoint::<short>").
    if nt is None and type_key.startswith("endpoint::"):
        short = type_key.split("::", 1)[1]
        ex = {"kind": "endpoint", "invoke_style": "records",
              "inputs": ["input"], "outputs": ["output"], "endpoint_name": None}
        try:
            ex["endpoint_name"] = get_endpoint_name_for_uc_model(short)
        except Exception as e:  # noqa: BLE001
            logger.info("exec: generic endpoint %s not resolvable: %s", short, e)
        return ex

    if nt is None:
        return {"kind": "unknown"}

    ports = ([p.name for p in nt.inputs], [p.name for p in nt.outputs])
    if nt.category == NodeCategory.IO:
        return {"kind": "io", "io_kind": nt.io_kind, "inputs": ports[0], "outputs": ports[1]}
    if nt.category == NodeCategory.ENDPOINT:
        ex = {"kind": "endpoint", "invoke_style": nt.invoke_style,
              "inputs": ports[0], "outputs": ports[1], "endpoint_name": None}
        try:
            ex["endpoint_name"] = get_endpoint_name(nt.endpoint_display_name)
        except Exception as e:  # noqa: BLE001
            logger.info("exec: endpoint %s not resolvable: %s", nt.endpoint_display_name, e)
        return ex
    if nt.category == NodeCategory.BATCH:
        # Endpoint-chain composites aren't Databricks jobs — run-wiring lands in a
        # later increment. Emit a descriptor the orchestrator treats as not-yet-
        # runnable (never falls through to a job dispatch with no job_name).
        if nt.kind == "endpoint_chain":
            return {"kind": "chain", "chain": nt.chain,
                    "inputs": ports[0], "outputs": ports[1], "runnable": False}
        return {"kind": "batch", "job_id": _resolve_batch_job_id(nt.job_name, w),
                "job_name": nt.job_name, "inputs": ports[0], "outputs": ports[1]}
    if nt.category == NodeCategory.TRANSFORM:
        # Deterministic reshape op (read_text_file / extract_field / …). Execution
        # wiring lands with the run path; flagged not-yet-runnable until then.
        return {"kind": "transform", "op": nt.type,
                "inputs": ports[0], "outputs": ports[1], "runnable": False}
    return {"kind": "unknown"}


def _enrich_graph(graph: dict, w: WorkspaceClient) -> dict:
    """Embed an `exec` block into every node so the orchestrator is a pure
    interpreter. Returns a new graph dict (does not mutate the input)."""
    nodes = []
    for n in graph.get("nodes", []):
        nodes.append({**n, "exec": _exec_descriptor(n.get("type", ""), w)})
    return {"nodes": nodes, "edges": graph.get("edges", [])}


def _upload_graph(graph: dict, catalog: str, schema: str) -> str:
    """Write the enriched graph to a per-run UC Volume dir; return its path.
    The Apps sandbox blocks direct open('/Volumes/..'), so go via the Files API."""
    run_uuid = uuid.uuid4().hex[:12]
    path = f"/Volumes/{catalog}/{schema}/{VOLUME_DIR_NAME}/{run_uuid}/graph.json"
    WorkspaceClient().files.upload(
        file_path=path,
        contents=io.BytesIO(json.dumps(graph).encode("utf-8")),
        overwrite=True,
    )
    return path


def start_workflow_run(
    *, user_info: UserInfo, graph: dict, run_name: str,
    experiment_name: str = EXPERIMENT_TAG,
) -> RunDispatchResult:
    """Dispatch a canvas graph as one async orchestrator job run (one run per
    execution). Logs to the user's own MLflow experiment folder
    (/Users/<email>/<mlflow_experiment_folder>/<experiment_name>), like the other
    workflows. Pre-creates the MLflow run so search lights up immediately, then
    run_now; flips to `failed` if dispatch raises."""
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    experiment_name = (experiment_name or EXPERIMENT_TAG).strip()

    w = WorkspaceClient()
    job_id = _resolve_orchestrator_job_id(w)
    enriched = _enrich_graph(graph, w)
    graph_path = _upload_graph(enriched, catalog, schema)

    experiment = set_mlflow_experiment(
        experiment_tag=experiment_name, user_email=user_info.user_email, shared=False
    )

    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment.experiment_id
    ) as pre_run:
        mlflow_run_id = pre_run.info.run_id
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", FEATURE_TAG)
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_status", "submitted")
        mlflow.log_param("node_count", len(enriched["nodes"]))
        mlflow.log_param("edge_count", len(enriched["edges"]))
        # Log the original graph (with positions/labels/types) so the Past-Runs
        # result viewer can re-render the canvas read-only with per-node status.
        try:
            mlflow.log_dict(graph, "graph.json")
        except Exception as e:  # noqa: BLE001
            logger.info("could not log graph.json for run %s: %s", mlflow_run_id, e)
        try:
            job_run = w.jobs.run_now(
                job_id=job_id,
                job_parameters={
                    "catalog": catalog,
                    "schema": schema,
                    "sql_warehouse_id": os.environ.get("SQL_WAREHOUSE", ""),
                    "user_email": user_info.user_email,
                    "mlflow_experiment": experiment_name,
                    "mlflow_run_name": run_name,
                    "mlflow_run_id": mlflow_run_id,
                    "graph_path": graph_path,
                },
            )
        except Exception as e:
            mlflow.set_tag("job_status", "failed")
            mlflow.set_tag("error", str(e)[:500])
            raise
        mlflow.set_tag("job_run_id", str(job_run.run_id))

    return RunDispatchResult(
        job_id=job_id,
        job_run_id=int(job_run.run_id),
        mlflow_run_id=mlflow_run_id,
        experiment_id=str(experiment.experiment_id),
    )


# ─── Run status / search / result ────────────────────────────────────────────


def get_run_status(run_id: str) -> dict:
    """MLflow run status + overall job_status + per-node statuses for the
    live canvas overlay."""
    client = MlflowClient()
    run = client.get_run(run_id)
    tags = run.data.tags
    node_status = {
        k.split(":", 2)[1]: v
        for k, v in tags.items()
        if k.startswith("node:") and k.endswith(":status")
    }
    node_error = {
        k.split(":", 2)[1]: v
        for k, v in tags.items()
        if k.startswith("node:") and k.endswith(":error")
    }
    return {
        "status": run.info.status,
        "job_status": tags.get("job_status", ""),
        "node_status": node_status,
        "node_error": node_error,
        "run_name": tags.get("mlflow.runName", ""),
    }


def _download_run_json(client, run_id: str, artifact_path: str):
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, artifact_path, dst_path=tmp)
            with open(local) as f:
                return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.info("%s not available for run %s: %s", artifact_path, run_id, e)
        return None


def get_run_result(run_id: str) -> dict:
    """Everything the Past-Runs result viewer needs: the workflow outputs, the
    graph (to re-render the canvas read-only), and per-node status/errors."""
    client = MlflowClient()
    node_status: dict = {}
    node_error: dict = {}
    try:
        tags = client.get_run(run_id).data.tags
        node_status = {k.split(":", 2)[1]: v for k, v in tags.items()
                       if k.startswith("node:") and k.endswith(":status")}
        node_error = {k.split(":", 2)[1]: v for k, v in tags.items()
                      if k.startswith("node:") and k.endswith(":error")}
    except Exception as e:  # noqa: BLE001
        logger.info("tags not available for run %s: %s", run_id, e)
    return {
        "result": _download_run_json(client, run_id, "results/workflow_results.json") or {},
        "graph": _download_run_json(client, run_id, "graph.json"),
        "node_status": node_status,
        "node_error": node_error,
    }


def _experiment_ids() -> list[str]:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    exps = mlflow.search_experiments(filter_string="tags.used_by_genesis_workbench='yes'")
    return [e.experiment_id for e in exps]


def search_runs(
    user_email: str, text: str = "", page: int = 1, page_size: int = 20
) -> tuple[list[dict], bool]:
    """One page of the user's ai_canvas runs (newest first), optionally filtered
    by run-name substring. Returns (rows, has_more). Modelled on
    enzyme_optimization.search_runs."""
    exp_ids = _experiment_ids()
    if not exp_ids:
        return [], False
    df = mlflow.search_runs(
        filter_string=(
            f"tags.feature='{FEATURE_TAG}' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=exp_ids,
        order_by=["start_time DESC"],
    )
    if df.empty:
        return [], False
    if text:
        df = df[df["tags.mlflow.runName"].astype(str).str.contains(text, case=False, na=False)]
    # Page the newest-first frame; fetch one extra row to know if more remain.
    page = max(1, int(page or 1))
    start_i = (page - 1) * page_size
    window = df.iloc[start_i : start_i + page_size + 1]
    has_more = len(window) > page_size
    window = window.iloc[:page_size]
    out: list[dict] = []
    try:
        job_id = _resolve_orchestrator_job_id(WorkspaceClient())
    except Exception:
        job_id = 0
    for _, r in window.iterrows():
        job_run_id = str(r.get("tags.job_run_id", "") or "")
        start = r.get("start_time")
        # Emit ISO-8601 (tz-aware UTC from MLflow) so the browser can render it in
        # the user's local time; fall back to the raw string if it's not a Timestamp.
        if start is not None and hasattr(start, "isoformat"):
            try:
                start_str = start.isoformat()
                if start_str == "NaT":
                    start_str = ""
            except Exception:  # noqa: BLE001
                start_str = str(start)
        else:
            start_str = str(start) if start is not None else ""
        out.append(
            {
                "run_id": str(r.get("run_id", "")),
                "run_name": str(r.get("tags.mlflow.runName", "") or ""),
                "job_status": str(r.get("tags.job_status", "") or ""),
                "node_count": _safe_int(r.get("params.node_count")),
                "start_time": start_str,
                "run_url": job_run_url(job_id, job_run_id) if job_id else "",
            }
        )
    return out, has_more


def _safe_int(v) -> int | None:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None
