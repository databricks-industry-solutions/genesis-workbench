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
from .node_catalog import (
    NODE_CATALOG_TABLE,
    NodeCategory,
    node_catalog_ddl,
    node_from_dict,
    node_to_dict,
)
from .workbench import execute_non_select_query, execute_select_query

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
    options: list[str] = field(default_factory=list)   # enum: the valid values
    # New fields are appended (keyword-only in practice) so existing positional /
    # **dict construction keeps working. minimum/maximum bound numeric params the
    # way `options` bounds enums; None = unbounded.
    minimum: float | None = None
    maximum: float | None = None
    required: bool = False
    label: str = ""
    help: str = ""


def param_schema(p: Param) -> dict:
    """Canonical JSON-Schema property for a param — the single descriptor every
    consumer (UI, generator prompt, MCP inputSchema, validator) derives from."""
    t = {"int": "integer", "float": "number", "bool": "boolean"}.get(p.type, "string")
    s: dict = {"type": t}
    if p.options:
        s["enum"] = list(p.options)
    if p.default is not None:
        s["default"] = p.default
    if p.minimum is not None:
        s["minimum"] = p.minimum
    if p.maximum is not None:
        s["maximum"] = p.maximum
    if p.help:
        s["description"] = p.help
    return s


class ParamValidationError(ValueError):
    """A supplied param value violates the capability's declared contract."""


def validate_params(params: list, values: dict) -> dict:
    """Validate + coerce supplied `values` against a param contract (a list of
    Param-like objects with .name/.type/.options/.minimum/.maximum/.required).

    Contract:
      - enum (`options`) not matched  -> reject (ParamValidationError); no sensible
        nearest value (e.g. strategy="guided").
      - numeric out of [minimum, maximum] -> CLAMP to the nearest bound; the caller
        should log the coercion (never silently surprising).
      - required + missing/blank        -> reject.
      - bool/int/float                  -> light type coercion.
    Unknown keys pass through untouched (forward-compatible). Returns coerced dict.
    Works on both the wheel `Param` and the app `ParamField` (duck-typed)."""
    by_name = {p.name: p for p in (params or [])}
    out = dict(values or {})
    for name, p in by_name.items():
        present = name in out and out[name] not in (None, "")
        if getattr(p, "required", False) and not present:
            raise ParamValidationError(f"required param '{name}' is missing")
        if not present:
            continue
        v = out[name]
        opts = getattr(p, "options", None) or []
        if opts and v not in opts:
            raise ParamValidationError(
                f"param '{name}'={v!r} is not one of {opts}"
            )
        ptype = getattr(p, "type", "string")
        try:
            if ptype == "int":
                v = int(v)
            elif ptype == "float":
                v = float(v)
            elif ptype == "bool":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")
        except (TypeError, ValueError):
            raise ParamValidationError(f"param '{name}'={out[name]!r} is not a valid {ptype}")
        if ptype in ("int", "float"):
            lo, hi = getattr(p, "minimum", None), getattr(p, "maximum", None)
            if lo is not None and v < lo:
                v = type(v)(lo)
            if hi is not None and v > hi:
                v = type(v)(hi)
        out[name] = v
    return out


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
# Each entry is keyed by UC short name; ports/invoke_style/params mirror the deployed
# model's MLflow signature (verified per endpoint). For "records" models, columns the
# model requires beyond the primary input are encoded as params (with a sensible
# default) so a generic call always sends a complete record.
_PROTEINA = {  # proteina_complexa family share one signature
    "in": "target_pdb:pdb", "out": "binders:json", "style": "records",
    "params": [{"name": "binder_length_min", "type": "int", "default": 50},
               {"name": "binder_length_max", "type": "int", "default": 80},
               {"name": "num_samples", "type": "int", "default": 2},
               {"name": "hotspot_residues", "type": "string", "default": ""},
               {"name": "target_chain", "type": "string", "default": "A"}],
}
_SCGPT_EMB = {  # scgpt / teddy: anndata-style sparse matrix + obs/var (real genes needed)
    "in": "adata_sparsematrix:json, adata_obs:json, adata_var:json",
    "out": "embeddings:json", "style": "records",
}
_ENDPOINT_CONTRACTS: dict[str, dict] = {
    # single-input "inputs"-style (verified working)
    "esmfold": {"in": "sequence:sequence", "out": "pdb:pdb", "style": "inputs"},
    "esm2_embeddings": {"in": "sequence:sequence", "out": "embedding:embedding", "style": "inputs"},
    "netsolp_v1": {"in": "sequence:sequence", "out": "solubility:score", "style": "inputs"},
    "pltnum_v1": {"in": "sequence:sequence", "out": "half_life:score", "style": "inputs"},
    "chemprop_admet": {"in": "smiles:smiles", "out": "admet:json", "style": "inputs"},
    "chemprop_bbbp": {"in": "smiles:smiles", "out": "bbbp:score", "style": "inputs"},
    "chemprop_clintox": {"in": "smiles:smiles", "out": "clintox:score", "style": "inputs"},
    "kermt_admet": {"in": "smiles:smiles", "out": "admet:json", "style": "inputs"},
    "proteinmpnn": {"in": "pdb:pdb", "out": "sequences:sequences", "style": "records"},
    # records-style, corrected to the deployed models' signatures
    "boltz": {"in": "input:sequence", "out": "pdb:pdb", "style": "records",
              "params": [{"name": "msa", "type": "string", "default": ""},
                         {"name": "use_msa_server", "type": "string", "default": "true"}]},
    "rfdiffusion_inpainting": {"in": "pdb:pdb", "out": "pdb:pdb", "style": "records",
                               "params": [{"name": "start_idx", "type": "int", "default": 1},
                                          {"name": "end_idx", "type": "int", "default": 10}]},
    "rfdiffusion_unconditional": {"in": "contig:any", "out": "pdb:pdb", "style": "inputs"},
    "deepstabp_v1": {"in": "sequence:sequence", "out": "tm:score", "style": "records",
                     "params": [{"name": "growth_temp", "type": "float", "default": 37.0},
                                {"name": "mt_mode", "type": "string", "default": "Cell"}]},
    "mhcflurry_v2": {"in": "sequence:sequence", "out": "immuno:score", "style": "records",
                     "params": [{"name": "alleles", "type": "string", "default": "HLA-A*02:01"}]},
    "genmol": {"in": "fragment:smiles", "out": "molecules:json", "style": "records"},
    "diffdock": {"in": "protein_pdb:pdb, ligand_smiles:smiles", "out": "poses:json", "style": "records",
                 "params": [{"name": "samples_per_complex", "type": "int", "default": 10},
                            {"name": "esm_embeddings_b64", "type": "string", "default": "{}"}]},
    "diffdock_esm_embeddings": {"in": "protein_pdb:pdb", "out": "embedding:embedding", "style": "records"},
    "proteina_complexa": _PROTEINA,
    "proteina_complexa_ame": _PROTEINA,
    "proteina_complexa_ligand": _PROTEINA,
    "scgpt": _SCGPT_EMB,
    "teddy": _SCGPT_EMB,
    "scgpt_perturbation": {"in": "expression:json, gene_names:json, genes_to_perturb:any",
                           "out": "predictions:json", "style": "records",
                           "params": [{"name": "perturbation_type", "type": "string", "default": "knockout"}]},
    "scimilarity_get_embedding": {"in": "celltype_sample:json", "out": "embedding:embedding",
                                  "style": "records",
                                  "params": [{"name": "celltype_sample_obs", "type": "string", "default": ""}]},
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


def _sql_lit(v) -> str:
    """SQL string literal with single-quote escaping; NULL for None."""
    if v is None:
        return "NULL"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "''") + "'"


def publish_node_catalog(catalog: str | None = None, schema: str | None = None,
                         source: str = "builtin") -> int:
    """Publish the built-in CURATED_NODES to the `node_catalog` table — the single
    runtime source of truth read by the wheel (MCP/executor) and Vortex.

    Full-overwrite of this `source`'s rows (idempotent); rows from other sources
    (future "mcp:<server>" external tools) are untouched. Runs from a deploy
    notebook (the wheel carries both the data and DB access). Returns row count."""
    from .builtin_nodes import CURATED_NODES  # local import: data module, no cycle
    cat = catalog or os.environ["CORE_CATALOG_NAME"]
    sch = schema or os.environ["CORE_SCHEMA_NAME"]
    tbl = f"{cat}.{sch}.{NODE_CATALOG_TABLE}"
    execute_non_select_query(node_catalog_ddl(cat, sch))
    execute_non_select_query(f"DELETE FROM {tbl} WHERE source = {_sql_lit(source)}")
    rows = [
        f"({_sql_lit(n.type)}, {_sql_lit(str(n.category))}, {_sql_lit(n.kind or '')}, "
        f"{_sql_lit(n.module)}, {_sql_lit(source)}, {_sql_lit(json.dumps(node_to_dict(n)))}, true)"
        for n in CURATED_NODES
    ]
    if rows:
        execute_non_select_query(
            f"INSERT INTO {tbl} (type, category, kind, module, source, node_json, is_active) "
            f"VALUES {', '.join(rows)}"
        )
    return len(rows)


def read_catalog_nodes():
    """Read the `node_catalog` table → list[NodeType]; [] if the table is absent,
    empty, or unreadable (callers fall back to legacy sources). This is the single
    runtime source of truth published from CURATED_NODES (see publish_node_catalog)."""
    cat = os.environ.get("CORE_CATALOG_NAME")
    schema = os.environ.get("CORE_SCHEMA_NAME")
    if not cat or not schema:
        return []
    try:
        df = execute_select_query(
            f"SELECT node_json FROM {cat}.{schema}.{NODE_CATALOG_TABLE} WHERE is_active = true"
        )
    except Exception:  # noqa: BLE001 — missing table / pre-publish → fall back
        return []
    nodes = []
    for _, r in df.iterrows():
        try:
            nodes.append(node_from_dict(json.loads(r["node_json"])))
        except Exception:  # noqa: BLE001 — skip a bad row, keep the rest
            continue
    return nodes


def _param_from_field(pf) -> Param:
    """ParamField (rich node-catalog model) → Param (capability model)."""
    return Param(pf.name, pf.type, pf.default, list(pf.options),
                 minimum=pf.minimum, maximum=pf.maximum, required=pf.required,
                 label=pf.label, help=pf.help)


def _batch_node_to_capability(n) -> Capability | None:
    """A BATCH NodeType (databricks_job | endpoint_chain) → a runnable Capability."""
    ins = [Port(p.name, str(p.dtype)) for p in n.inputs]
    outs = [Port(p.name, str(p.dtype)) for p in n.outputs]
    params = [_param_from_field(p) for p in n.params]
    common = dict(id=f"workflow:{n.type}", label=n.label, module=n.module,
                  inputs=ins, outputs=outs, params=params, description=n.description)
    if n.kind == "databricks_job":
        return Capability(kind=JOB, job_name=n.job_name, **common)
    if n.kind == "endpoint_chain":
        return Capability(kind=CHAIN, chain_id=(n.chain or n.type), **common)
    return None


def workflow_capabilities() -> list[Capability]:
    """Prebuilt workflows. Source of truth is the `node_catalog` table (published
    from CURATED_NODES); falls back to the legacy `prebuilt_workflows` registry if
    node_catalog has no BATCH rows (pre-publish / missing table)."""
    batch = [n for n in read_catalog_nodes() if n.category == NodeCategory.BATCH]
    if batch:
        return [c for c in (_batch_node_to_capability(n) for n in batch) if c]
    return _workflow_capabilities_legacy()


def _workflow_capabilities_legacy() -> list[Capability]:
    """Prebuilt workflows from the `prebuilt_workflows` Delta registry (fallback)."""
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
                          p.get("default"), p.get("options", []) or [],
                          minimum=p.get("minimum"), maximum=p.get("maximum"),
                          required=bool(p.get("required", False)),
                          label=p.get("label", "") or "", help=p.get("help", "") or "")
                    for p in _p("params_json")],
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
