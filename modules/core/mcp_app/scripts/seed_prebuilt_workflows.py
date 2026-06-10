"""Create + seed the `prebuilt_workflows` Delta registry (P1).

This is the workflows counterpart to the endpoints registry (`model_deployments`
⋈ `models`): the single source of truth for the prebuilt workflows that both the
MCP server (workflow tools) and Vortex (prebuilt-workflow nodes) consume.

Each row declares a capability: stable key, label, kind (databricks_job |
endpoint_chain), module, the Jobs-API `job_name` (for jobs), typed input/output
ports, params, and a description. Job availability is resolved live (Jobs API),
so this table only carries the *contract*, not deploy state.

Seeded from the curated workflow set (genomics, structure prediction, guided
optimization, fine-tunes). Endpoint-chains (Protein Design / ADMET Screen) are
listed with kind=endpoint_chain and no job_name — their executor lands with the
shared-executor work.

Run locally:  DATABRICKS_CONFIG_PROFILE=ci-demo python seed_prebuilt_workflows.py \
                  --catalog <cat> --schema <schema> --warehouse <id>
Idempotent: CREATE OR REPLACE TABLE + INSERT (fully re-seeds).
"""
from __future__ import annotations

import argparse
import json

from databricks.sdk import WorkspaceClient

# (key, label, kind, module, job_name, inputs, outputs, params, description)
# ports: [{"name","dtype","label"}]; params: [{"name","label","type","default","options"}]
WORKFLOWS = [
    ("variant_calling", "Variant Calling", "databricks_job", "genomics",
     "gwas_parabricks_alignment",
     [{"name": "fastq_r1", "dtype": "path"}, {"name": "fastq_r2", "dtype": "path"}],
     [{"name": "vcf", "dtype": "path"}],
     [{"name": "reference_genome_path", "type": "string"},
      {"name": "output_volume_path", "type": "string"}],
     "Align FASTQ reads to a reference genome and call variants (Parabricks)."),
    ("vcf_ingestion", "VCF Ingestion", "databricks_job", "genomics",
     "vcf_ingestion_glow",
     [{"name": "vcf", "dtype": "path"}], [{"name": "table", "dtype": "table"}],
     [{"name": "output_table_name", "type": "string"}],
     "Ingest a VCF file into a Delta table."),
    ("variant_annotation", "Variant Annotation", "databricks_job", "genomics",
     "variant_annotation_clinical",
     [{"name": "table", "dtype": "table"}], [{"name": "annotations", "dtype": "table"}],
     [{"name": "gene_panel_mode", "type": "select", "options": ["custom", "acmg"]},
      {"name": "gene_regions", "type": "string"}],
     "Annotate variants against ClinVar with gene-panel filtering."),
    ("gwas", "GWAS", "databricks_job", "genomics", "gwas_glow_analysis",
     [{"name": "vcf", "dtype": "path"}, {"name": "phenotype", "dtype": "path"}],
     [{"name": "results", "dtype": "table"}],
     [{"name": "phenotype_column", "type": "string"},
      {"name": "contigs", "type": "string"}, {"name": "pvalue_threshold", "type": "float"}],
     "Genome-wide association study over called variants + phenotype."),
    ("alphafold2", "AlphaFold Structure Prediction", "databricks_job", "large_molecule",
     "run_alphafold",
     [{"name": "sequence", "dtype": "sequence"}], [{"name": "pdb", "dtype": "pdb"}],
     [], "High-accuracy 3D structure prediction from a protein sequence."),
    ("enzyme_optimization", "Guided Enzyme Optimization", "databricks_job", "large_molecule",
     "run_enzyme_optimization_gwb",
     [{"name": "motif_pdb", "dtype": "pdb"}, {"name": "substrate_smiles", "dtype": "smiles"}],
     [{"name": "candidates", "dtype": "json"}],
     [{"name": "motif_residues_csv", "type": "string"}, {"name": "target_chain", "type": "string"},
      {"name": "scaffold_length_min", "type": "int"}, {"name": "scaffold_length_max", "type": "int"},
      {"name": "num_samples", "type": "int"}, {"name": "num_iterations", "type": "int"},
      {"name": "weight_motif_rmsd", "type": "float"}, {"name": "weight_plddt", "type": "float"},
      {"name": "weight_boltz", "type": "float"}, {"name": "weight_solubility", "type": "float"},
      {"name": "weight_half_life", "type": "float"}, {"name": "weight_thermostab", "type": "float"},
      {"name": "weight_immuno", "type": "float"}, {"name": "half_life_margin", "type": "float"},
      {"name": "resampling_temperature", "type": "float"}, {"name": "strategy", "type": "string"},
      {"name": "run_proteinmpnn", "type": "bool"}, {"name": "use_inprocess_ame", "type": "bool"}],
     "Reward-weighted enzyme design loop (GenMol -> score -> reseed) across solubility, "
     "half-life, thermostability, immunogenicity, fold confidence + motif RMSD axes."),
    ("molecule_optimization", "Guided Molecule Optimization", "databricks_job", "small_molecule",
     "run_molecule_optimization_gwb",
     [{"name": "seed_smiles", "dtype": "smiles"}, {"name": "target_sequence", "dtype": "sequence"}],
     [{"name": "top_k", "dtype": "json"}],
     [{"name": "num_iterations", "type": "int"}, {"name": "num_samples", "type": "int"},
      {"name": "qed_min", "type": "float"}, {"name": "tox_max", "type": "float"}],
     "Reward-weighted small-molecule design loop (GenMol -> score -> reseed)."),
    ("esm2_finetune", "Fine-Tune ESM2", "databricks_job", "large_molecule",
     "bionemo_esm_finetune_job",
     [{"name": "train_data", "dtype": "path"}, {"name": "validation_data", "dtype": "path"}],
     [{"name": "weights", "dtype": "path"}],
     [{"name": "esm_variant", "type": "select", "options": ["8M", "35M", "150M", "650M"]},
      {"name": "task_type", "type": "select", "options": ["regression", "classification"]},
      {"name": "num_steps", "type": "int"}],
     "Fine-tune ESM2 on a labeled sequence dataset (BioNeMo)."),
    ("kermt_finetune", "Fine-Tune KERMT", "databricks_job", "small_molecule",
     "kermt_finetune_job",
     [{"name": "train_data", "dtype": "path"}, {"name": "validation_data", "dtype": "path"},
      {"name": "test_data", "dtype": "path"}],
     [{"name": "ft_id", "dtype": "json"}],
     [{"name": "target_names", "type": "string"},
      {"name": "dataset_type", "type": "select", "options": ["classification", "regression"]},
      {"name": "epochs", "type": "int"}],
     "Fine-tune KERMT (GROVER) on a toxicity / ADMET dataset."),
    ("kermt_deploy", "Deploy KERMT", "databricks_job", "small_molecule",
     "kermt_deploy_job",
     [{"name": "ft_id", "dtype": "json"}], [{"name": "endpoint", "dtype": "json"}],
     [{"name": "model_name", "type": "string"}],
     "Register a fine-tuned KERMT model as a real-time ADMET endpoint."),
    # Endpoint-chains (kind=endpoint_chain): no job_name; executor lands with the
    # shared-executor work. Listed so the registry is complete.
    ("protein_design", "Protein Design", "endpoint_chain", "large_molecule", None,
     [{"name": "sequence", "dtype": "sequence"}], [{"name": "designs", "dtype": "json"}],
     [{"name": "n_rfdiffusion_hits", "type": "int"}],
     "Chain: RFDiffusion -> ProteinMPNN -> ESMFold to design + validate binders."),
    ("admet_screen", "ADMET Screen", "endpoint_chain", "small_molecule", None,
     [{"name": "smiles", "dtype": "smiles"}], [{"name": "profile", "dtype": "json"}],
     [], "Chain: run ADMET / toxicity predictors over a SMILES set."),
    ("protein_binder_design", "Protein Binder Design", "endpoint_chain", "large_molecule", None,
     [{"name": "target_pdb", "dtype": "pdb"}, {"name": "target_sequence", "dtype": "sequence"}],
     [{"name": "designs", "dtype": "json"}],
     [{"name": "target_chain", "type": "string"}, {"name": "hotspot_residues", "type": "string"},
      {"name": "binder_length_min", "type": "int"}, {"name": "binder_length_max", "type": "int"},
      {"name": "num_samples", "type": "int"}, {"name": "validate_esmfold", "type": "bool"}],
     "Chain: Proteina-Complexa binder design for a target protein (+ optional ESMFold validation)."),
    ("ligand_binder_design", "Ligand Binder Design", "endpoint_chain", "small_molecule", None,
     [{"name": "ligand_pdb", "dtype": "pdb"}], [{"name": "designs", "dtype": "json"}],
     [{"name": "binder_length_min", "type": "int"}, {"name": "binder_length_max", "type": "int"},
      {"name": "num_samples", "type": "int"}, {"name": "validate_esmfold", "type": "bool"},
      {"name": "validate_diffdock", "type": "bool"}, {"name": "ligand_smiles", "type": "string"}],
     "Chain: Proteina-Complexa-Ligand protein binders for a ligand (+ optional ESMFold / DiffDock)."),
    ("motif_scaffolding", "Motif Scaffolding", "endpoint_chain", "small_molecule", None,
     [{"name": "motif_pdb", "dtype": "pdb"}], [{"name": "scaffolds", "dtype": "json"}],
     [{"name": "target_chain", "type": "string"}, {"name": "scaffold_length_min", "type": "int"},
      {"name": "scaffold_length_max", "type": "int"}, {"name": "num_samples", "type": "int"},
      {"name": "optimize_mpnn", "type": "bool"}, {"name": "validate_esmfold", "type": "bool"}],
     "Chain: Proteina-Complexa-AME scaffolds preserving a functional motif (+ MPNN / ESMFold)."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--warehouse", required=True)
    args = ap.parse_args()

    w = WorkspaceClient()
    table = f"{args.catalog}.{args.schema}.prebuilt_workflows"

    def run(stmt: str) -> None:
        resp = w.statement_execution.execute_statement(
            warehouse_id=args.warehouse, statement=stmt,
            catalog=args.catalog, schema=args.schema, wait_timeout="50s",
        )
        state = resp.status.state.value if resp.status and resp.status.state else "?"
        if state != "SUCCEEDED":
            raise RuntimeError(f"statement {state}: {resp.status.error}")

    run(f"""CREATE OR REPLACE TABLE {table} (
        workflow_key STRING, label STRING, kind STRING, module STRING,
        job_name STRING, inputs_json STRING, outputs_json STRING,
        params_json STRING, description STRING, is_active BOOLEAN)""")

    def sql_str(v):
        return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"

    rows = []
    for key, label, kind, module, job, ins, outs, params, desc in WORKFLOWS:
        rows.append(
            "(" + ", ".join([
                sql_str(key), sql_str(label), sql_str(kind), sql_str(module),
                sql_str(job), sql_str(json.dumps(ins)), sql_str(json.dumps(outs)),
                sql_str(json.dumps(params)), sql_str(desc), "true",
            ]) + ")"
        )
    run(f"INSERT INTO {table} VALUES\n" + ",\n".join(rows))
    print(f"Seeded {len(rows)} workflows into {table}")


if __name__ == "__main__":
    main()
