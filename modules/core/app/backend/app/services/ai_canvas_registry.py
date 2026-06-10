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
    BATCH = "batch"         # "Prebuilt Workflows" — job OR endpoint-chain OR (future) MCP
    IO = "io"               # data input / output (UC Volume, Delta table)
    TRANSFORM = "transform" # reshape/parse/map one node's output to the next node's input


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
    # A "Prebuilt Workflow" (category BATCH) is a higher-level capability whose
    # execution kind is NOT necessarily a Databricks job. `kind` distinguishes:
    #   "databricks_job"  → dispatched as a Jobs run (job_name)
    #   "endpoint_chain"  → an app-orchestrated chain of real-time endpoints
    #                       (e.g. Protein Design, ADMET Screen); chain id in `chain`
    #   "mcp"             → (future) a tool exposed by the MCP server app
    # ENDPOINT/IO nodes leave this empty (their category implies execution).
    kind: str = ""
    chain: str | None = None                   # endpoint_chain → app chain handler id
    requires_endpoints: list[str] = field(default_factory=list)  # chain availability gate (DISPLAY_TO_UC keys)
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


# ─── Transform nodes ─────────────────────────────────────────────────────────
# Reshape/parse/map the output of one node into the input shape of the next.
# Deterministic, no model/job calls. `kind="transform"`; the node `type` is the
# op the orchestrator switches on. (Execution wiring lands with the run path.)

_KIND_TRANSFORM = "transform"

_TRANSFORM_NODES: list[NodeType] = [
    NodeType(
        type="read_text_file", label="Read Text File", category=NodeCategory.TRANSFORM,
        kind=_KIND_TRANSFORM,
        description="Read a UC Volume text file into a string (e.g. a sequence or SMILES).",
        inputs=[Port("file", PortType.PATH, "File path")],
        outputs=[Port("text", PortType.ANY, "Text")],
    ),
    NodeType(
        type="parse_fasta", label="Parse FASTA", category=NodeCategory.TRANSFORM,
        kind=_KIND_TRANSFORM,
        description="Parse a FASTA file into a list of sequences.",
        inputs=[Port("file", PortType.PATH, "FASTA path")],
        outputs=[Port("sequences", PortType.SEQUENCES, "Sequences")],
    ),
    NodeType(
        type="csv_column", label="CSV Column", category=NodeCategory.TRANSFORM,
        kind=_KIND_TRANSFORM,
        description="Extract one column from a CSV / table as a list of values.",
        inputs=[Port("table", PortType.ANY, "Table or CSV")],
        outputs=[Port("values", PortType.ANY, "Values")],
        params=[ParamField("column", "Column name", "string", required=True)],
    ),
    NodeType(
        type="extract_field", label="Extract Field", category=NodeCategory.TRANSFORM,
        kind=_KIND_TRANSFORM,
        description="Pull a single value out of a JSON object by key / dotted path "
                    "(e.g. predictions.0.smiles).",
        inputs=[Port("data", PortType.JSON, "JSON")],
        outputs=[Port("value", PortType.ANY, "Value")],
        params=[ParamField("path", "Field path", "string", required=True,
                           help="Dotted path, e.g. results.0.pdb")],
    ),
    NodeType(
        type="field_mapper", label="Field Mapper", category=NodeCategory.TRANSFORM,
        kind=_KIND_TRANSFORM,
        description="Map named output fields to the named input fields the next node "
                    "expects (declarative {target: source-path} mapping).",
        inputs=[Port("data", PortType.JSON, "JSON")],
        outputs=[Port("mapped", PortType.JSON, "Mapped JSON")],
        params=[ParamField("mappings", "Mappings (JSON)", "text", default="{}",
                           help='e.g. {"sequence": "predictions.0.seq"}')],
    ),
    NodeType(
        type="select_top_k", label="Select / Top-K", category=NodeCategory.TRANSFORM,
        kind=_KIND_TRANSFORM,
        description="Keep the top K items of a JSON list, ranked by a field.",
        inputs=[Port("items", PortType.JSON, "Items")],
        outputs=[Port("top", PortType.JSON, "Top items")],
        params=[
            ParamField("k", "K", "int", default=5),
            ParamField("by", "Rank by field", "string", default=""),
            ParamField("order", "Order", "select", default="desc", options=["desc", "asc"]),
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


# ─── Prebuilt Workflows ──────────────────────────────────────────────────────
# Category BATCH groups all higher-level capabilities under the "Prebuilt
# Workflows" palette section — regardless of how they execute. `kind` records
# the execution model (see NodeType). Job-backed entries set `job_name` (matched
# against the live Jobs API); endpoint-chain composites set `chain` +
# `requires_endpoints` (availability gated on those endpoints being deployed).

_KIND_JOB = "databricks_job"
_KIND_CHAIN = "endpoint_chain"

_WORKFLOW_NODES: list[NodeType] = [
    # ── Genomics suite ──
    NodeType(
        type="variant_calling", label="Variant Calling", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="genomics", job_name="gwas_parabricks_alignment",
        description="Align FASTQ reads to a reference genome and call variants (Parabricks).",
        inputs=[Port("fastq_r1", PortType.PATH, "FASTQ R1"),
                Port("fastq_r2", PortType.PATH, "FASTQ R2")],
        outputs=[Port("vcf", PortType.PATH, "VCF")],
        params=[
            ParamField("reference_genome_path", "Reference genome path", "string"),
            ParamField("output_volume_path", "Output volume path", "string"),
        ],
    ),
    NodeType(
        type="vcf_ingestion", label="VCF Ingestion", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="genomics", job_name="vcf_ingestion_glow",
        description="Ingest a VCF file into a Delta table for downstream analysis.",
        inputs=[Port("vcf", PortType.PATH, "VCF path")],
        outputs=[Port("table", PortType.TABLE, "Variants table")],
        params=[ParamField("output_table_name", "Output table", "string", required=True,
                           help="catalog.schema.table")],
    ),
    NodeType(
        type="variant_annotation", label="Variant Annotation", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="genomics", job_name="variant_annotation_clinical",
        description="Annotate variants against ClinVar with gene-panel filtering.",
        inputs=[Port("table", PortType.TABLE, "Variants table")],
        outputs=[Port("annotations", PortType.TABLE, "Annotated variants")],
        params=[
            ParamField("gene_panel_mode", "Gene panel", "select", default="custom",
                       options=["custom", "acmg"]),
            ParamField("gene_regions", "Gene regions", "string", default="",
                       help="Used when panel = custom (e.g. BRCA1,BRCA2)."),
        ],
    ),
    NodeType(
        type="gwas", label="GWAS", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="genomics", job_name="gwas_glow_analysis",
        description="Genome-wide association study over called variants + phenotype.",
        inputs=[Port("vcf", PortType.PATH, "VCF path"),
                Port("phenotype", PortType.PATH, "Phenotype path")],
        outputs=[Port("results", PortType.TABLE, "GWAS results")],
        params=[
            ParamField("phenotype_column", "Phenotype column", "string", default="phenotype"),
            ParamField("contigs", "Contigs", "string", default="6"),
            ParamField("pvalue_threshold", "p-value threshold", "float", default=0.01),
        ],
    ),
    # ── Structure prediction ──
    NodeType(
        type="alphafold2", label="AlphaFold Structure Prediction", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="large_molecule", job_name="run_alphafold",
        description="High-accuracy 3D structure prediction from a protein sequence.",
        inputs=[Port("sequence", PortType.SEQUENCE)],
        outputs=[Port("pdb", PortType.PDB)],
    ),
    # ── Guided optimization loops ──
    NodeType(
        type="enzyme_optimization", label="Guided Enzyme Optimization",
        category=NodeCategory.BATCH, kind=_KIND_JOB, module="large_molecule",
        job_name="run_enzyme_optimization_gwb",
        description="Reward-weighted enzyme design loop (GenMol → score → reseed).",
        inputs=[Port("motif_pdb", PortType.PDB, "Motif PDB"),
                Port("substrate_smiles", PortType.SMILES, "Substrate")],
        outputs=[Port("candidates", PortType.JSON)],
        params=[
            ParamField("motif_residues_csv", "Motif residues", "string"),
            ParamField("num_samples", "Samples / iter", "int", default=8),
            ParamField("num_iterations", "Iterations", "int", default=10),
        ],
    ),
    NodeType(
        type="molecule_optimization", label="Guided Molecule Optimization",
        category=NodeCategory.BATCH, kind=_KIND_JOB, module="small_molecule",
        job_name="run_molecule_optimization_gwb",
        description="Reward-weighted small-molecule design loop (GenMol → score → reseed).",
        inputs=[Port("seed_smiles", PortType.SMILES, "Seed SMILES"),
                Port("target_sequence", PortType.SEQUENCE, "Target (optional)")],
        outputs=[Port("top_k", PortType.JSON, "Top candidates")],
        params=[
            ParamField("num_iterations", "Iterations", "int", default=5),
            ParamField("num_samples", "Samples / iter", "int", default=24),
            ParamField("qed_min", "QED min", "float", default=0.0),
            ParamField("tox_max", "Tox max", "float", default=1.0),
        ],
    ),
    # ── Fine-tuning ──
    NodeType(
        type="esm2_finetune", label="Fine-Tune ESM2", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="large_molecule", job_name="bionemo_esm_finetune_job",
        description="Fine-tune ESM2 on a labeled sequence dataset (BioNeMo).",
        inputs=[Port("train_data", PortType.PATH, "Train CSV"),
                Port("validation_data", PortType.PATH, "Validation CSV")],
        outputs=[Port("weights", PortType.PATH, "Fine-tuned weights")],
        params=[
            ParamField("esm_variant", "ESM2 variant", "select", default="650M",
                       options=["8M", "35M", "150M", "650M"]),
            ParamField("task_type", "Task", "select", default="regression",
                       options=["regression", "classification"]),
            ParamField("num_steps", "Steps", "int", default=50),
        ],
    ),
    NodeType(
        type="kermt_finetune", label="Fine-Tune KERMT", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="small_molecule", job_name="kermt_finetune_job",
        description="Fine-tune KERMT (GROVER) on a toxicity / ADMET dataset.",
        inputs=[Port("train_data", PortType.PATH, "Train CSV"),
                Port("validation_data", PortType.PATH, "Validation CSV"),
                Port("test_data", PortType.PATH, "Test CSV")],
        outputs=[Port("ft_id", PortType.JSON, "Fine-tune id")],
        params=[
            ParamField("target_names", "Target column(s)", "string", default="toxicity"),
            ParamField("dataset_type", "Dataset type", "select", default="classification",
                       options=["classification", "regression"]),
            ParamField("epochs", "Epochs", "int", default=20),
        ],
    ),
    NodeType(
        type="kermt_deploy", label="Deploy KERMT", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="small_molecule", job_name="kermt_deploy_job",
        description="Register a fine-tuned KERMT model as a real-time ADMET endpoint.",
        inputs=[Port("ft_id", PortType.JSON, "Fine-tune id")],
        outputs=[Port("endpoint", PortType.JSON, "Endpoint")],
        params=[ParamField("model_name", "Model name", "string", default="kermt_admet")],
    ),
]

# ── Endpoint-chain composites (kind = endpoint_chain) ──
# Not Databricks jobs — app-orchestrated chains of real-time endpoints. They sit
# in the same "Prebuilt Workflows" group and (eventually) become MCP tools.
# Availability is gated on `requires_endpoints` being deployed (DISPLAY_TO_UC keys).

_CHAIN_NODES: list[NodeType] = [
    NodeType(
        type="protein_design", label="Protein Design", category=NodeCategory.BATCH,
        kind=_KIND_CHAIN, chain="protein_design", module="large_molecule",
        requires_endpoints=["RFDiffusion", "ProteinMPNN", "ESMFold"],
        description="Chain: RFDiffusion → ProteinMPNN → ESMFold to design + validate "
                    "binders around a marked region.",
        inputs=[Port("sequence", PortType.SEQUENCE, "Sequence ([region] marked)")],
        outputs=[Port("designs", PortType.JSON, "Designs")],
        params=[ParamField("n_rfdiffusion_hits", "RFDiffusion designs", "int", default=4)],
    ),
    NodeType(
        type="admet_screen", label="ADMET Screen", category=NodeCategory.BATCH,
        kind=_KIND_CHAIN, chain="admet_screen", module="small_molecule",
        requires_endpoints=["Chemprop ADMET"],
        description="Chain: run ADMET / toxicity predictors (Chemprop ADMET, BBBP, "
                    "ClinTox, KERMT) over a SMILES set and combine the profile.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("profile", PortType.JSON, "ADMET profile")],
    ),
]


CURATED_NODES: list[NodeType] = (
    _IO_NODES + _TRANSFORM_NODES + _ENDPOINT_NODES + _WORKFLOW_NODES + _CHAIN_NODES
)

# Fast lookups used by the catalog builder + executors.
CURATED_BY_TYPE: dict[str, NodeType] = {n.type: n for n in CURATED_NODES}
CURATED_BY_ENDPOINT: dict[str, NodeType] = {
    n.endpoint_display_name: n for n in _ENDPOINT_NODES if n.endpoint_display_name
}
CURATED_BY_JOB: dict[str, NodeType] = {
    n.job_name: n for n in _WORKFLOW_NODES if n.job_name
}
