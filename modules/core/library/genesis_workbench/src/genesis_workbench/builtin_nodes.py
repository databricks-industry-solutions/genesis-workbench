"""Built-in Vortex node catalog — the hand-authored CURATED_NODES.

This is the single authoring home for the platform's node definitions. It lives
in the wheel (shared core) so a deploy notebook can publish it to the
node_catalog table, and so the app, the executor, and the MCP server all derive
from one source. The app's `services.ai_canvas_registry` re-exports these names.

Each NodeType declares its label, category, typed I/O ports, and editable param
fields (with valid values + ranges) plus how to invoke it (endpoint display name,
batch-job name, endpoint-chain id, or a built-in IO handler).
"""
from __future__ import annotations

from .node_catalog import NodeCategory, NodeType, ParamField, Port, PortType

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
    NodeType(
        type="smiles_to_pdb", label="SMILES → PDB", category=NodeCategory.TRANSFORM,
        kind=_KIND_TRANSFORM,
        description="Convert a SMILES string into a 3D-embedded PDB block "
                    "(RDKit ETKDGv3 → MMFF94) — feeds ligand-PDB inputs like "
                    "Ligand Binder Design.",
        inputs=[Port("smiles", PortType.SMILES, "SMILES")],
        outputs=[Port("pdb", PortType.PDB, "Ligand PDB")],
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
        module="small_molecule", endpoint_display_name="Chemprop ADMET", invoke_style="inputs",
        description="ADMET property profile from a SMILES string.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("admet", PortType.JSON)],
    ),
    NodeType(
        type="chemprop_bbbp", label="Chemprop BBBP", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="Chemprop BBBP", invoke_style="inputs",
        description="Blood-brain-barrier penetration from SMILES.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("bbbp", PortType.SCORE)],
    ),
    NodeType(
        type="chemprop_clintox", label="Chemprop ClinTox", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="Chemprop ClinTox", invoke_style="inputs",
        description="Clinical-toxicity likelihood from SMILES.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("clintox", PortType.SCORE)],
    ),
    NodeType(
        type="kermt_admet", label="KERMT ADMET", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="KERMT ADMET", invoke_style="inputs",
        description="KERMT (GROVER multi-task) ADMET / toxicity profile from a SMILES string.",
        inputs=[Port("smiles", PortType.SMILES)],
        outputs=[Port("admet", PortType.JSON)],
    ),
    NodeType(
        type="diffdock", label="DiffDock", category=NodeCategory.ENDPOINT,
        module="small_molecule", endpoint_display_name="DiffDock",
        description="Molecular docking of a ligand into a protein structure.",
        inputs=[Port("pdb", PortType.PDB), Port("smiles", PortType.SMILES)],
        outputs=[Port("poses", PortType.JSON)],
    ),
    # Single Cell — these endpoints take an AnnData-style JSON payload (cells +
    # obs/var), so the input port is a JSON blob (invoke_style="inputs" passes it
    # through as inputs=[<payload>], matching the live services). Not typically
    # produced by a single upstream node — wire from a Text/Delta input or an
    # upstream single-cell step.
    NodeType(
        type="teddy", label="TEDDY-G Cell Embeddings", category=NodeCategory.ENDPOINT,
        module="single_cell", endpoint_display_name="TEDDY Annotation", invoke_style="inputs",
        description="TEDDY-G 400M cell embeddings from an AnnData payload "
                    "(adata_sparsematrix + obs/var).",
        inputs=[Port("cells", PortType.JSON, "AnnData payload")],
        outputs=[Port("embedding", PortType.EMBEDDING)],
    ),
    NodeType(
        type="scgpt_perturbation", label="scGPT Perturbation", category=NodeCategory.ENDPOINT,
        module="single_cell", endpoint_display_name="scGPT Perturbation", invoke_style="inputs",
        description="Predict gene knockout / overexpression effects (expression + "
                    "gene_names + perturbation spec).",
        inputs=[Port("cells", PortType.JSON, "Expression + perturbation payload")],
        outputs=[Port("predictions", PortType.JSON)],
    ),
    NodeType(
        type="scgpt_embeddings", label="scGPT Cell Embeddings", category=NodeCategory.ENDPOINT,
        module="single_cell", endpoint_display_name="scGPT Embeddings", invoke_style="inputs",
        description="scGPT cell embeddings from an AnnData payload.",
        inputs=[Port("cells", PortType.JSON, "AnnData payload")],
        outputs=[Port("embedding", PortType.EMBEDDING)],
    ),
    NodeType(
        type="scimilarity_get_embedding", label="SCimilarity Embeddings",
        category=NodeCategory.ENDPOINT, module="single_cell",
        endpoint_display_name="SCimilarity Get Embedding", invoke_style="inputs",
        description="SCimilarity cell embeddings from an AnnData payload "
                    "(celltype_sample + obs).",
        inputs=[Port("cells", PortType.JSON, "AnnData payload")],
        outputs=[Port("embedding", PortType.EMBEDDING)],
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
        params=[ParamField("output_table_name", "Output table", "string",
                           default="vortex_vcf_ingested", help="catalog.schema.table")],
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
            ParamField("pathogenic_vcf_path", "Pathogenic VCF path", "string", default="",
                       help="Optional override (legacy/testing)."),
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
            ParamField("contigs", "Contigs", "string", default="",
                       help="Comma-separated; empty = all contigs."),
            ParamField("hwe_cutoff", "HWE cutoff", "string", default="1e-6"),
            ParamField("pvalue_threshold", "p-value threshold", "string", default="5e-8"),
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
            ParamField("motif_residues_csv", "Motif residues", "string",
                       help="Catalytic residues to preserve, e.g. 26,11,20"),
            ParamField("target_chain", "Motif chain", "string", default="B"),
            ParamField("scaffold_length_min", "Scaffold length min", "int", default=80, minimum=1, maximum=2000),
            ParamField("scaffold_length_max", "Scaffold length max", "int", default=120, minimum=1, maximum=2000),
            ParamField("num_samples", "Samples / iter", "int", default=8, minimum=1, maximum=64),
            ParamField("num_iterations", "Iterations", "int", default=10, minimum=1, maximum=50),
            # Reward-axis weights — the heart of *guided* optimization. 0 drops an
            # axis. Mirror the UI's AXIS_LABELS + DEFAULT_AXIS_WEIGHTS.
            ParamField("weight_motif_rmsd", "Weight · Motif RMSD", "float", default=1.0, minimum=0.0,
                       help="Lower RMSD is better — catalytic-site drift after redesign."),
            ParamField("weight_plddt", "Weight · ESMFold pLDDT", "float", default=1.3, minimum=0.0,
                       help="Higher is better — global fold confidence."),
            ParamField("weight_boltz", "Weight · Boltz substrate conf.", "float", default=0.5, minimum=0.0,
                       help="Only contributes if a substrate SMILES is supplied."),
            ParamField("weight_solubility", "Weight · NetSolP solubility", "float", default=1.0, minimum=0.0,
                       help="Higher is better — E. coli solubility probability."),
            ParamField("weight_half_life", "Weight · PLTNUM half-life", "float", default=2.6, minimum=0.0,
                       help="Higher is better (vs references). Set 0 to drop."),
            ParamField("weight_thermostab", "Weight · DeepSTABp Tm", "float", default=1.0, minimum=0.0,
                       help="Higher is better — predicted melting temperature."),
            ParamField("weight_immuno", "Weight · MHCflurry immunogenicity", "float", default=1.5, minimum=0.0,
                       help="Lower is better — strong-presenter density."),
            ParamField("half_life_margin", "Half-life margin", "float", default=0.05, minimum=0.0, maximum=1.0),
            ParamField("resampling_temperature", "Resampling temperature", "float", default=0.1, minimum=0.0, maximum=2.0),
            ParamField("strategy", "Reseed strategy", "select", default="resample",
                       options=["resample", "noop"]),
            ParamField("run_proteinmpnn", "Run ProteinMPNN", "bool", default=True),
            ParamField("use_inprocess_ame", "Accurate mode (in-process AME)", "bool", default=False,
                       help="Slower, higher-fidelity scoring (uses the AME job variant)."),
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
            ParamField("num_iterations", "Iterations", "int", default=5, minimum=1, maximum=50),
            ParamField("num_samples", "Samples / iter", "int", default=24, minimum=1, maximum=256),
            ParamField("select_top", "Select top", "int", default=3, minimum=1, maximum=50),
            ParamField("dock_top_k", "Dock top-K", "int", default=5, minimum=1, maximum=50),
            ParamField("qed_min", "QED min (hard filter)", "float", default=0.5, minimum=0.0, maximum=1.0),
            ParamField("tox_max", "ClinTox max (hard filter)", "float", default=0.3, minimum=0.0, maximum=1.0),
            ParamField("temperature", "Sampling temperature", "float", default=1.2, minimum=0.0, maximum=2.0),
            ParamField("randomness", "Randomness", "float", default=2.0, minimum=0.0, maximum=5.0),
            ParamField("target_label", "Target label (gene)", "string", default="",
                       help="Docking target gene symbol, for MLflow logging."),
            ParamField("dock_per_iter", "Dock per iter", "int", default=8, minimum=0, maximum=50),
            ParamField("dock_samples", "Dock samples", "int", default=3, minimum=1, maximum=50),
        ],
    ),
    # ── Fine-tuning ──
    NodeType(
        type="esm2_finetune", label="Fine-Tune ESM2", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="large_molecule", job_name="bionemo_esm_finetune_job",
        description="Fine-tune ESM2 on a labeled sequence dataset (BioNeMo).",
        inputs=[Port("train_data", PortType.PATH, "Train CSV"),
                Port("evaluation_data", PortType.PATH, "Evaluation CSV")],
        outputs=[Port("weights", PortType.PATH, "Fine-tuned weights")],
        params=[
            ParamField("finetune_label", "Fine-tune label", "string", default="vortex_finetune",
                       help="Name for this fine-tune run."),
            ParamField("esm_variant", "ESM2 variant", "select", default="650M",
                       options=["8M", "35M", "150M", "650M"]),
            ParamField("task_type", "Task", "select", default="regression",
                       options=["regression", "classification"]),
            ParamField("num_steps", "Steps", "int", default=50),
            ParamField("should_use_lora", "Use LoRA", "bool", default=False),
            ParamField("micro_batch_size", "Micro batch size", "int", default=2),
            ParamField("precision", "Precision", "string", default="bf16-mixed"),
            ParamField("mlp_ft_dropout", "MLP dropout", "float", default=0.25),
            ParamField("mlp_hidden_size", "MLP hidden size", "int", default=256),
            ParamField("mlp_target_size", "MLP target size", "int", default=1),
            ParamField("mlp_lr", "MLP learning rate", "float", default=5e-3),
            ParamField("mlp_lr_multiplier", "MLP LR multiplier", "float", default=1e2),
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
            ParamField("finetune_label", "Fine-tune label", "string", default="vortex_finetune",
                       help="Name for this fine-tune run."),
            ParamField("target_names", "Target column(s)", "string", default="toxicity"),
            ParamField("dataset_type", "Dataset type", "select", default="classification",
                       options=["classification", "regression"]),
            ParamField("epochs", "Epochs", "int", default=20),
            ParamField("batch_size", "Batch size", "int", default=16),
            ParamField("ffn_hidden_size", "FFN hidden size", "int", default=700),
        ],
    ),
    NodeType(
        type="kermt_deploy", label="Deploy KERMT", category=NodeCategory.BATCH,
        kind=_KIND_JOB, module="small_molecule", job_name="kermt_deploy_job",
        description="Register a fine-tuned KERMT model as a real-time ADMET endpoint.",
        inputs=[Port("ft_id", PortType.JSON, "Fine-tune id")],
        outputs=[Port("endpoint", PortType.JSON, "Endpoint")],
        params=[
            ParamField("model_name", "Model name", "string", default="kermt_admet"),
            ParamField("workload_type", "Workload type", "string", default="",
                       help="Serving workload size override (optional)."),
        ],
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
        outputs=[Port("designs", PortType.JSON, "Designs (structures)"),
                 Port("sequences", PortType.SEQUENCES, "Designed sequences")],
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
        params=[
            ParamField("run_admet", "Chemprop ADMET", "bool", default=True),
            ParamField("run_bbbp", "BBBP", "bool", default=True),
            ParamField("run_clintox", "ClinTox", "bool", default=True),
            ParamField("run_kermt", "KERMT", "bool", default=False),
        ],
    ),
    NodeType(
        type="protein_binder_design", label="Protein Binder Design",
        category=NodeCategory.BATCH, kind=_KIND_CHAIN, chain="protein_binder_design",
        module="large_molecule",
        requires_endpoints=["Proteina-Complexa Binder", "ESMFold"],
        description="Chain: Proteina-Complexa binder design for a target protein "
                    "(folds the target first if given a sequence; optional ESMFold "
                    "validation of each design).",
        inputs=[Port("target_pdb", PortType.PDB, "Target PDB"),
                Port("target_sequence", PortType.SEQUENCE, "Target sequence (folded if no PDB)")],
        outputs=[Port("designs", PortType.JSON, "Binder designs")],
        params=[
            ParamField("target_chain", "Target chain", "string", default="A"),
            ParamField("hotspot_residues", "Hotspot residues", "string", default="",
                       help="e.g. 45,46,89"),
            ParamField("binder_length_min", "Binder length min", "int", default=50),
            ParamField("binder_length_max", "Binder length max", "int", default=80),
            ParamField("num_samples", "Samples", "int", default=2),
            ParamField("validate_esmfold", "Validate (ESMFold)", "bool", default=False),
        ],
    ),
    NodeType(
        type="ligand_binder_design", label="Ligand Binder Design",
        category=NodeCategory.BATCH, kind=_KIND_CHAIN, chain="ligand_binder_design",
        module="small_molecule",
        requires_endpoints=["Proteina-Complexa Ligand"],
        description="Chain: Proteina-Complexa-Ligand protein binders for a ligand "
                    "(expects a ligand PDB; optional ESMFold + DiffDock validation).",
        inputs=[Port("ligand_pdb", PortType.PDB, "Ligand PDB")],
        outputs=[Port("designs", PortType.JSON, "Protein binder designs")],
        params=[
            ParamField("binder_length_min", "Binder length min", "int", default=50),
            ParamField("binder_length_max", "Binder length max", "int", default=80),
            ParamField("num_samples", "Samples", "int", default=2),
            ParamField("validate_esmfold", "Validate (ESMFold)", "bool", default=False),
            ParamField("validate_diffdock", "Validate (DiffDock)", "bool", default=False),
            ParamField("ligand_smiles", "Ligand SMILES", "string", default="",
                       help="Needed for DiffDock validation."),
        ],
    ),
    NodeType(
        type="motif_scaffolding", label="Motif Scaffolding",
        category=NodeCategory.BATCH, kind=_KIND_CHAIN, chain="motif_scaffolding",
        module="small_molecule",
        requires_endpoints=["Proteina-Complexa AME"],
        description="Chain: Proteina-Complexa-AME scaffolds preserving a functional "
                    "motif (optional ProteinMPNN optimisation + ESMFold validation).",
        inputs=[Port("motif_pdb", PortType.PDB, "Motif PDB")],
        outputs=[Port("scaffolds", PortType.JSON, "Scaffolds")],
        params=[
            ParamField("target_chain", "Motif chain", "string", default="B"),
            ParamField("scaffold_length_min", "Scaffold length min", "int", default=50),
            ParamField("scaffold_length_max", "Scaffold length max", "int", default=80),
            ParamField("num_samples", "Samples", "int", default=2),
            ParamField("optimize_mpnn", "Optimise (ProteinMPNN)", "bool", default=False),
            ParamField("validate_esmfold", "Validate (ESMFold)", "bool", default=False),
        ],
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
