"""Shared capability registry — the single source of truth for Genesis Workbench
capabilities, consumed by all three pathways (UI, Vortex, MCP).

A *capability* is one thing you can run: a live model-serving endpoint, a prebuilt
workflow (Databricks job or endpoint-chain), or a data transform. Existence +
availability come from the Delta registries (`model_deployments` ⋈ `models` for
endpoints; `prebuilt_workflows` for workflows); the typed I/O *contract* for
endpoints comes from the small curated overlay below (the one place it lives).

This module has NO presentation/protocol concerns (no HTTP, MCP, React) — those
belong to the per-pathway adapters. See `executor.py` for running a capability.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .models import ModelCategory, get_deployed_models
from .workbench import execute_select_query

# Execution kinds.
ENDPOINT = "endpoint"
JOB = "databricks_job"
CHAIN = "endpoint_chain"
TRANSFORM = "transform"

_MODULES = ("large_molecule", "small_molecule", "single_cell", "genomics")


@dataclass
class Port:
    name: str
    dtype: str = "any"
    label: str = ""


@dataclass
class Param:
    name: str
    type: str = "string"  # string | int | float | bool | select | text
    default: object | None = None
    options: list[str] = field(default_factory=list)


@dataclass
class Capability:
    id: str
    label: str
    kind: str
    module: str | None = None
    endpoint_name: str | None = None   # ENDPOINT: resolved serving endpoint name
    invoke_style: str = "inputs"       # ENDPOINT: "inputs" | "records"
    job_name: str | None = None        # JOB
    chain_id: str | None = None        # CHAIN
    op: str | None = None              # TRANSFORM
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    description: str = ""
    available: bool = True


def _ports(spec: str) -> list[Port]:
    """'sequence:sequence, smiles:smiles' -> [Port,...]; '' -> []."""
    out: list[Port] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        name, _, dtype = tok.partition(":")
        out.append(Port(name=name.strip(), dtype=(dtype.strip() or "any")))
    return out


# Curated endpoint I/O contract, keyed by UC short name. inputs / outputs /
# invoke_style / params — the single home for the typed endpoint contract. An
# endpoint deployed without an entry here falls back to a generic single port.
_ENDPOINT_CONTRACTS: dict[str, dict] = {
    "esmfold": {"in": "sequence:sequence", "out": "pdb:pdb", "style": "inputs"},
    "boltz": {"in": "sequence:sequence", "out": "pdb:pdb", "style": "inputs"},
    "proteinmpnn": {"in": "pdb:pdb", "out": "sequences:sequences", "style": "records"},
    "rfdiffusion_inpainting": {"in": "pdb:pdb", "out": "pdb:pdb", "style": "inputs"},
    "esm2_embeddings": {"in": "sequence:sequence", "out": "embedding:embedding", "style": "inputs"},
    "netsolp_v1": {"in": "sequence:sequence", "out": "solubility:score", "style": "inputs"},
    "pltnum_v1": {"in": "sequence:sequence", "out": "half_life:score", "style": "inputs"},
    "deepstabp_v1": {"in": "sequence:sequence", "out": "tm:score", "style": "inputs"},
    "mhcflurry_v2": {"in": "sequence:sequence", "out": "immuno:score", "style": "inputs"},
    "chemprop_admet": {"in": "smiles:smiles", "out": "admet:json", "style": "inputs"},
    "chemprop_bbbp": {"in": "smiles:smiles", "out": "bbbp:score", "style": "inputs"},
    "chemprop_clintox": {"in": "smiles:smiles", "out": "clintox:score", "style": "inputs"},
    "kermt_admet": {"in": "smiles:smiles", "out": "admet:json", "style": "inputs"},
    "diffdock": {"in": "pdb:pdb, smiles:smiles", "out": "poses:json", "style": "records"},
    "teddy": {"in": "data:table", "out": "annotations:table", "style": "inputs"},
    "scgpt_perturbation": {"in": "data:table", "out": "predictions:table", "style": "inputs"},
}


def _uc_short(uc_name: str) -> str:
    return uc_name.split("/")[0].split(".")[-1].strip().lower()


def endpoint_capabilities() -> list[Capability]:
    """Live deployed endpoints (Delta registry) + typed overlay; generic fallback."""
    caps: list[Capability] = []
    for module in _MODULES:
        if module == "genomics":
            continue  # no real-time endpoints
        try:
            df = get_deployed_models(ModelCategory(module))
        except Exception:  # noqa: BLE001
            continue
        for _, r in df.iterrows():
            short = _uc_short(str(r["uc_name"]))
            c = _ENDPOINT_CONTRACTS.get(short)
            if c:
                ins, outs, style = _ports(c["in"]), _ports(c["out"]), c.get("style", "inputs")
                params = [Param(**p) for p in c.get("params", [])]
            else:
                ins, outs, style, params = [Port("input")], [Port("output", "json")], "records", []
            caps.append(Capability(
                id=f"endpoint:{short}", label=str(r["model_display_name"]), kind=ENDPOINT,
                module=module, endpoint_name=str(r["model_endpoint_name"]), invoke_style=style,
                inputs=ins, outputs=outs, params=params, available=True,
                description=f"Deployed model-serving endpoint '{r['model_display_name']}'.",
            ))
    return caps


def workflow_capabilities() -> list[Capability]:
    """Prebuilt workflows from the `prebuilt_workflows` Delta registry."""
    cat = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]
    try:
        df = execute_select_query(
            f"SELECT workflow_key, label, kind, module, job_name, inputs_json, "
            f"outputs_json, params_json, description FROM {cat}.{schema}.prebuilt_workflows "
            f"WHERE is_active = true"
        )
    except Exception:  # noqa: BLE001
        return []
    caps: list[Capability] = []
    for _, r in df.iterrows():
        def _p(col):
            try:
                return json.loads(r[col]) if r[col] is not None else []
            except Exception:  # noqa: BLE001
                return []
        kind = str(r["kind"])
        caps.append(Capability(
            id=f"workflow:{r['workflow_key']}", label=str(r["label"]), kind=kind,
            module=(None if r["module"] is None else str(r["module"])),
            job_name=(None if r["job_name"] is None else str(r["job_name"])),
            chain_id=(str(r["workflow_key"]) if kind == CHAIN else None),
            inputs=[Port(p.get("name"), p.get("dtype", "any")) for p in _p("inputs_json")],
            outputs=[Port(p.get("name"), p.get("dtype", "any")) for p in _p("outputs_json")],
            params=[Param(p.get("name"), p.get("type", "string"),
                          p.get("default"), p.get("options", []) or []) for p in _p("params_json")],
            description=str(r["description"] or ""),
        ))
    return caps


# Deterministic data transforms (canvas plumbing). No external dependency.
_TRANSFORMS: list[Capability] = [
    Capability(id="transform:read_text_file", label="Read Text File", kind=TRANSFORM,
               op="read_text_file", inputs=[Port("file", "path")], outputs=[Port("text", "any")],
               description="Read a UC Volume text file into a string."),
    Capability(id="transform:parse_fasta", label="Parse FASTA", kind=TRANSFORM, op="parse_fasta",
               inputs=[Port("file", "path")], outputs=[Port("sequences", "sequences")],
               description="Parse a FASTA file into a list of sequences."),
    Capability(id="transform:csv_column", label="CSV Column", kind=TRANSFORM, op="csv_column",
               inputs=[Port("table", "any")], outputs=[Port("values", "any")],
               params=[Param("column", "string")], description="Extract one CSV column as a list."),
    Capability(id="transform:extract_field", label="Extract Field", kind=TRANSFORM, op="extract_field",
               inputs=[Port("data", "json")], outputs=[Port("value", "any")],
               params=[Param("path", "string")], description="Pull a value out of JSON by dotted path."),
    Capability(id="transform:field_mapper", label="Field Mapper", kind=TRANSFORM, op="field_mapper",
               inputs=[Port("data", "json")], outputs=[Port("mapped", "json")],
               params=[Param("mappings", "text", "{}")], description="Map output fields to input fields."),
    Capability(id="transform:select_top_k", label="Select / Top-K", kind=TRANSFORM, op="select_top_k",
               inputs=[Port("items", "json")], outputs=[Port("top", "json")],
               params=[Param("k", "int", 5), Param("by", "string"),
                       Param("order", "select", "desc", ["desc", "asc"])],
               description="Keep the top K items of a JSON list, ranked by a field."),
    Capability(id="transform:smiles_to_pdb", label="SMILES → PDB", kind=TRANSFORM,
               op="smiles_to_pdb", inputs=[Port("smiles", "smiles")], outputs=[Port("pdb", "pdb")],
               description="Convert a SMILES string into a 3D-embedded PDB block "
                           "(RDKit ETKDGv3 → MMFF94)."),
]


def list_capabilities() -> list[Capability]:
    """All capabilities across kinds (endpoints + workflows + transforms)."""
    return endpoint_capabilities() + workflow_capabilities() + list(_TRANSFORMS)


def get_capability(cap_id: str) -> Capability | None:
    for c in list_capabilities():
        if c.id == cap_id:
            return c
    return None
