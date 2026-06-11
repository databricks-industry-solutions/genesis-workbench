"""Shared node-catalog model — the canonical definition of a Vortex/MCP node.

This is the single home for the rich node model (`NodeType` + its `Port`/`ParamField`
sub-models). It lives in the wheel so every pathway shares one shape:
  - the app authors the node instances (`ai_canvas_registry.CURATED_NODES`) using
    these classes (imported from here),
  - a publisher serializes them to the catalog table (`node_to_dict`),
  - the wheel reads the table back into `NodeType` (`node_from_dict`) so MCP / the
    executor / Vortex all see the same definitions (incl. valid values + ranges).

No HTTP/MCP/React concerns here — pure data model + (de)serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class NodeCategory(StrEnum):
    ENDPOINT = "endpoint"   # real-time model serving endpoint (seconds)
    BATCH = "batch"         # "Prebuilt Workflows" — job OR endpoint-chain OR (future) MCP
    IO = "io"               # data input / output (UC Volume, Delta table)
    TRANSFORM = "transform" # reshape/parse/map one node's output to the next node's input


class PortType(StrEnum):
    SEQUENCE = "sequence"       # protein amino-acid sequence
    SEQUENCES = "sequences"     # list of sequences
    PDB = "pdb"                 # PDB structure (text)
    SMILES = "smiles"           # small-molecule SMILES
    EMBEDDING = "embedding"     # numeric vector(s)
    SCORE = "score"             # scalar metric / property value
    TABLE = "table"             # Delta table reference
    PATH = "path"               # UC Volume path
    JSON = "json"               # arbitrary structured payload
    ANY = "any"


@dataclass(frozen=True)
class Port:
    name: str
    dtype: PortType
    label: str = ""


@dataclass(frozen=True)
class ParamField:
    name: str
    label: str
    type: str = "string"        # string | int | float | bool | select | text
    default: object | None = None
    options: list[str] = field(default_factory=list)   # enum: valid values
    minimum: float | None = None    # numeric lower bound (None = unbounded)
    maximum: float | None = None    # numeric upper bound
    required: bool = False
    help: str = ""


@dataclass(frozen=True)
class NodeType:
    type: str                   # stable key used in the graph JSON + executors
    label: str                  # palette / canvas display name
    category: NodeCategory
    description: str = ""
    module: str | None = None   # single_cell | large_molecule | small_molecule | genomics
    #   "databricks_job"  → dispatched as a Jobs run (job_name)
    #   "endpoint_chain"  → an app-orchestrated chain of real-time endpoints (chain id in `chain`)
    #   "mcp"             → (future) a tool exposed by an MCP server
    kind: str = ""
    chain: str | None = None
    requires_endpoints: list[str] = field(default_factory=list)  # chain availability gate
    endpoint_display_name: str | None = None   # ENDPOINT → DISPLAY_TO_UC key
    job_name: str | None = None                # BATCH → Jobs API name
    io_kind: str | None = None                 # IO → volume_input | delta_input | text_input | output_sink
    invoke_style: str = "records"              # ENDPOINT query style: "records" | "inputs"
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    params: list[ParamField] = field(default_factory=list)


# ─── (de)serialization: NodeType <-> plain dict (the catalog-table row payload) ──

def _port_to_dict(p: Port) -> dict:
    return {"name": p.name, "dtype": str(p.dtype), "label": p.label}


def _port_from_dict(d: dict) -> Port:
    return Port(d["name"], PortType(d.get("dtype", "any")), d.get("label", "") or "")


def _param_to_dict(p: ParamField) -> dict:
    return {"name": p.name, "label": p.label, "type": p.type, "default": p.default,
            "options": list(p.options), "minimum": p.minimum, "maximum": p.maximum,
            "required": p.required, "help": p.help}


def _param_from_dict(d: dict) -> ParamField:
    return ParamField(d["name"], d.get("label", "") or "", d.get("type", "string"),
                      d.get("default"), list(d.get("options") or []),
                      d.get("minimum"), d.get("maximum"),
                      bool(d.get("required", False)), d.get("help", "") or "")


def node_to_dict(n: NodeType) -> dict:
    """Full-fidelity dict form of a NodeType (the catalog-table row payload)."""
    return {
        "type": n.type, "label": n.label, "category": str(n.category),
        "description": n.description, "module": n.module, "kind": n.kind,
        "chain": n.chain, "requires_endpoints": list(n.requires_endpoints),
        "endpoint_display_name": n.endpoint_display_name, "job_name": n.job_name,
        "io_kind": n.io_kind, "invoke_style": n.invoke_style,
        "inputs": [_port_to_dict(p) for p in n.inputs],
        "outputs": [_port_to_dict(p) for p in n.outputs],
        "params": [_param_to_dict(p) for p in n.params],
    }


def node_from_dict(d: dict) -> NodeType:
    """Reconstruct a NodeType from its dict form (round-trips with node_to_dict)."""
    return NodeType(
        type=d["type"], label=d.get("label", "") or "",
        category=NodeCategory(d.get("category", "io")),
        description=d.get("description", "") or "", module=d.get("module"),
        kind=d.get("kind", "") or "", chain=d.get("chain"),
        requires_endpoints=list(d.get("requires_endpoints") or []),
        endpoint_display_name=d.get("endpoint_display_name"), job_name=d.get("job_name"),
        io_kind=d.get("io_kind"), invoke_style=d.get("invoke_style", "records") or "records",
        inputs=[_port_from_dict(p) for p in d.get("inputs") or []],
        outputs=[_port_from_dict(p) for p in d.get("outputs") or []],
        params=[_param_from_dict(p) for p in d.get("params") or []],
    )


# ─── catalog table contract (single runtime source of truth) ─────────────────
# One row per node. `node_json` is the full node_to_dict payload (round-trips via
# node_from_dict); the other columns are denormalized for filtering. `source`
# distinguishes provenance: "builtin" (published from CURATED_NODES) vs a future
# "mcp:<server>" for ingested external MCP tools.
NODE_CATALOG_TABLE = "node_catalog"


def node_catalog_ddl(catalog: str, schema: str) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{NODE_CATALOG_TABLE} ("
        "type STRING, category STRING, kind STRING, module STRING, source STRING, "
        "node_json STRING, is_active BOOLEAN)"
    )
