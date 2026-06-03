"""Vortex (ai_canvas) — node-type registry.

The registry is the single source of truth for what can be dropped onto the
canvas. Each `NodeType` declares its display label, category, typed input /
output ports, and editable param fields — plus *how to invoke it* (an endpoint
display name resolved via `services.endpoints.DISPLAY_TO_UC`, a batch-job name
resolved against the Jobs API, or a built-in IO handler).

Two layers make up the catalog (see `ai_canvas.build_catalog`):

1. **Curated** — the hand-authored `CURATED_NODES` below carry rich I/O and
   param schemas, mirroring the typed wrappers in `services/protein.py` and
   `services/enzyme_optimization.py`. These drive validation and let the LLM
   compose valid graphs.
2. **Dynamic** — live `model_deployments` / `batch_models` rows merged in at
   request time so the palette reflects what is actually deployed. A deployed
   model with no curated entry still shows up as a generic single-port node.

V2 hook: "dynamically created" endpoints/jobs only need to append a NodeType
here (or be picked up by the dynamic merge) and add an executor branch in the
orchestrator notebook — nothing else in the stack changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class NodeCategory(StrEnum):
    ENDPOINT = "endpoint"   # real-time model serving endpoint (seconds)
    BATCH = "batch"         # long-running Databricks job (minutes → hours)
    IO = "io"               # data input / output (UC Volume, Delta table)


# Port data types. Edges are valid when the upstream output dtype matches the
# downstream input dtype, or either side is ANY. Kept deliberately coarse — the
# orchestrator coerces within a dtype (e.g. a single sequence vs a list).
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
    options: list[str] = field(default_factory=list)
    required: bool = False
    help: str = ""


@dataclass(frozen=True)
class NodeType:
    type: str                   # stable key used in the graph JSON + executors
    label: str                  # palette / canvas display name
    category: NodeCategory
    description: str = ""
    module: str | None = None   # single_cell | large_molecule | small_molecule | genomics
    # Invocation handle (exactly one is meaningful per category):
    endpoint_display_name: str | None = None   # ENDPOINT → DISPLAY_TO_UC key
    job_name: str | None = None                # BATCH → Jobs API name
    io_kind: str | None = None                 # IO → volume_input | delta_input | text_input | output_sink
    # How the orchestrator queries an ENDPOINT node:
    #   "records" → serving_endpoints.query(dataframe_records=[{port: value, **params}])
    #   "inputs"  → serving_endpoints.query(inputs=[<first input value>])
    invoke_style: str = "records"
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    params: list[ParamField] = field(default_factory=list)


# ─── IO nodes ────────────────────────────────────────────────────────────────

_IO_NODES: list[NodeType] = [
    NodeType(
        type="volume_input",
        label="Volume Input",
        category=NodeCategory.IO,
        io_kind="volume_input",
        description="Read input data from a Unity Catalog Volume path.",
        outputs=[Port("data", PortType.PATH, "Path")],
        params=[
            ParamField("path", "UC Volume path", "string", required=True,
                       help="e.g. /Volumes/<catalog>/<schema>/<volume>/input.fasta"),
        ],
    ),
    NodeType(
        type="delta_input",
        label="Delta Table Input",
        category=NodeCategory.IO,
        io_kind="delta_input",
        description="Read input rows from a Delta table.",
        outputs=[Port("data", PortType.TABLE, "Table")],
        params=[
            ParamField("table", "Fully-qualified table", "string", required=True,
                       help="catalog.schema.table"),
        ],
    ),
    NodeType(
        type="text_input",
        label="Text Input",
        category=NodeCategory.IO,
        io_kind="text_input",
        description="A literal value typed into the canvas (sequence, SMILES, …).",
        outputs=[Port("value", PortType.ANY, "Value")],
        params=[
            ParamField("value", "Value", "text", required=True),
        ],
    ),
    NodeType(
        type="output_sink",
        label="Output",
        category=NodeCategory.IO,
        io_kind="output_sink",
        description="Collect the upstream result as the workflow output "
                    "(logged to MLflow artifacts).",
        inputs=[Port("data", PortType.ANY, "Result")],
        params=[
            ParamField("name", "Artifact name", "string", default="result"),
        ],
    ),
]


# ─── Curated endpoint nodes ──────────────────────────────────────────────────
# `endpoint_display_name` MUST exist in services/endpoints.py::DISPLAY_TO_UC.

_ENDPOINT_NODES: list[NodeType] = [
    # Large Molecule
    NodeType(
        type="esmfold", label="ESMFold", category=NodeCategory.ENDPOINT,
        module="large_molecule", endpoint_display_name="ESMFold", invoke_style="inputs",
        description="Predict a 3D structure (PDB) from a protein sequence.",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("pdb", PortType.PDB)],
    ),
    NodeType(
        type="boltz", label="Boltz", category=NodeCategory.ENDPOINT,
        module="large_molecule", endpoint_display_name="Boltz", invoke_style="inputs",
        description="Multi-chain structure prediction from a sequence.",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("pdb", PortType.PDB)],
    ),
    NodeType(
        type="proteinmpnn", label="ProteinMPNN", category=NodeCategory.ENDPOINT,
        module="large_molecule", endpoint_display_name="ProteinMPNN",
        description="Design sequences for a given backbone (inverse folding).",
        inputs=[Port("pdb", PortType.PDB)],
        outputs=[Port("sequences", PortType.SEQUENCES)],
        params=[ParamField("fixed_positions", "Fixed positions", "string", default="")],
    ),
    NodeType(
        type="rfdiffusion", label="RFDiffusion", category=NodeCategory.ENDPOINT,
        module="large_molecule", endpoint_display_name="RFDiffusion",
        description="Generate protein backbones (inpainting).",
        inputs=[Port("pdb", PortType.PDB)],
        outputs=[Port("pdb", PortType.PDB)],
    ),
    NodeType(
        type="esm2_embeddings", label="ESM2 Embeddings", category=NodeCategory.ENDPOINT,
        module="large_molecule", endpoint_display_name="ESM2 Embeddings", invoke_style="inputs",
        description="Embed a protein sequence for similarity search.",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("embedding", PortType.EMBEDDING)],
    ),
    # Small Molecule — developability predictors
    NodeType(
        type="netsolp", label="NetSolP Solubility", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="NetSolP Solubility",
        description="Predict protein solubility from a sequence.",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("solubility", PortType.SCORE)],
    ),
    NodeType(
        type="pltnum", label="PLTNUM Half-Life", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="PLTNUM Half-Life Stability",
        description="Predict protein stability / half-life from a sequence.",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("half_life", PortType.SCORE)],
    ),
    NodeType(
        type="deepstabp", label="DeepSTABp Tm", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="DeepSTABp Tm",
        description="Predict melting temperature (Tm).",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("tm", PortType.SCORE)],
        params=[
            ParamField("growth_temp", "Growth temp (°C)", "float", default=37.0),
            ParamField("mt_mode", "Mode", "select", default="Cell", options=["Cell", "Lysate"]),
        ],
    ),
    NodeType(
        type="mhcflurry", label="MHCflurry Immunogenicity", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="MHCflurry Immunogenicity",
        description="Predict MHC-I immunogenic burden.",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("immuno", PortType.SCORE)],
        params=[ParamField("alleles", "HLA alleles (csv)", "string", default="")],
    ),
    NodeType(
        type="chemprop_admet", label="Chemprop ADMET", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="Chemprop ADMET",
        description="ADMET property profile from a SMILES string.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("admet", PortType.JSON)],
    ),
    NodeType(
        type="chemprop_bbbp", label="Chemprop BBBP", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="Chemprop BBBP",
        description="Blood-brain-barrier penetration from SMILES.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("bbbp", PortType.SCORE)],
    ),
    NodeType(
        type="chemprop_clintox", label="Chemprop ClinTox", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="Chemprop ClinTox",
        description="Clinical-toxicity likelihood from SMILES.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("clintox", PortType.SCORE)],
    ),
    NodeType(
        type="diffdock", label="DiffDock", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="DiffDock",
        description="Molecular docking of a ligand into a protein structure.",
        inputs=[Port("pdb", PortType.PDB), Port("smiles", PortType.SMILES)],
        outputs=[Port("poses", PortType.JSON)],
    ),
    # Single Cell
    NodeType(
        type="teddy", label="TEDDY Annotation", category=NodeCategory.ENDPOINT,
        module="single_cell", endpoint_display_name="TEDDY Annotation",
        description="Joint cell-type + disease annotation.",
        inputs=[Port("data", PortType.TABLE)],
        outputs=[Port("annotations", PortType.TABLE)],
    ),
    NodeType(
        type="scgpt_perturbation", label="scGPT Perturbation", category=NodeCategory.ENDPOINT,
        module="single_cell", endpoint_display_name="scGPT Perturbation",
        description="Predict gene knockout / overexpression effects.",
        inputs=[Port("data", PortType.TABLE)],
        outputs=[Port("predictions", PortType.TABLE)],
    ),
]


# ─── Curated batch-job nodes ─────────────────────────────────────────────────
# `job_name` is matched against the live Jobs API at dispatch time.

_BATCH_NODES: list[NodeType] = [
    NodeType(
        type="alphafold2", label="AlphaFold2", category=NodeCategory.BATCH,
        module="large_molecule", job_name="run_alphafold",
        description="High-accuracy structure prediction (batch job).",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("pdb", PortType.PDB)],
    ),
    NodeType(
        type="enzyme_optimization", label="Guided Enzyme Optimization",
        category=NodeCategory.BATCH, module="small_molecule",
        job_name="run_enzyme_optimization_gwb",
        description="Reward-weighted enzyme design loop (batch job).",
        inputs=[Port("pdb", PortType.PDB)],
        outputs=[Port("candidates", PortType.JSON)],
    ),
    NodeType(
        type="variant_annotation", label="Variant Annotation",
        category=NodeCategory.BATCH, module="genomics",
        job_name="run_variant_annotation",
        description="Annotate variants (ClinVar + gene filtering).",
        inputs=[Port("data", PortType.TABLE)],
        outputs=[Port("annotations", PortType.TABLE)],
    ),
]


CURATED_NODES: list[NodeType] = _IO_NODES + _ENDPOINT_NODES + _BATCH_NODES

# Fast lookups used by the catalog builder + executors.
CURATED_BY_TYPE: dict[str, NodeType] = {n.type: n for n in CURATED_NODES}
CURATED_BY_ENDPOINT: dict[str, NodeType] = {
    n.endpoint_display_name: n for n in _ENDPOINT_NODES if n.endpoint_display_name
}
CURATED_BY_JOB: dict[str, NodeType] = {
    n.job_name: n for n in _BATCH_NODES if n.job_name
}
