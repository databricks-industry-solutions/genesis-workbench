# Genesis Workbench Documentation

## Disease Biology

- [Variant Calling](variant_calling.md) — GPU-accelerated germline variant calling from FASTQ files using Parabricks
- [GWAS Analysis](gwas_analysis.md) — Genome-wide association testing between genetic variants and phenotypes using Glow
- [VCF Ingestion](vcf_ingestion.md) — Convert VCF files to Delta tables for querying and analysis
- [Variant Annotation](variant_annotation.md) — Annotate variants with ClinVar clinical significance data

## Protein Studies

- [Protein Structure Prediction](protein_structure_prediction.md) — Predict 3D protein structures with ESMFold (fast) or AlphaFold2 (high-accuracy)
- [Protein Design](protein_design.md) — Design novel proteins by redesigning specified regions using RFDiffusion + ProteinMPNN
- [Sequence Similarity Search](sequence_search.md) — Fast BLAST-like search across 150M+ sequences using ESM-2 embeddings

## Small Molecules

- [Molecular Docking](molecular_docking.md) — Predict protein-ligand binding poses using DiffDock
- [Protein Binder Design](protein_binder_design.md) — Design proteins that bind a target protein using Proteina-Complexa
- [Ligand Binder Design](ligand_binder_design.md) — Design proteins that bind a small molecule using Proteina-Complexa-Ligand
- [Motif Scaffolding](motif_scaffolding.md) — Transplant functional motifs into new protein scaffolds
- [Guided Enzyme Optimization](enzyme_optimization.md) — Reward-weighted optimization loop on top of motif scaffolding; scores each candidate on motif fidelity, fold confidence, optional substrate complex, and four developability axes (solubility, anchor-relative half-life, melting temperature, immunogenic burden). Two generation modes: **Fast** (default, ~30 min, no GPU cost — endpoint-based AME with parent resampling between iterations) and **Accurate** (~30-60 min, ~$22 GPU — in-process AME with Feynman-Kac steering biasing diffusion toward developability)
- [ADMET & Safety](admet_safety.md) — Profile molecules for drug-like properties and toxicity using ChemProp

## Single Cell

- [Single Cell Analysis](single_cell_analysis.md) — End-to-end scRNA-seq processing with Scanpy or RAPIDS, plus interactive results viewer with differential expression, pathway enrichment, and trajectory analysis
- [Cell Type Annotation](cell_type_annotation.md) — Predict cell types per cluster using SCimilarity's 23M-cell reference database
- [Cell Similarity Search](cell_similarity.md) — Find matching cells across published studies using SCimilarity embeddings
- [Gene Perturbation Prediction](perturbation_prediction.md) — Predict effects of gene knockouts or overexpression using scGPT

## NVIDIA BioNeMo

- [ESM2 Fine-tuning & Inference](bionemo_esm2.md) — Fine-tune and run inference with NVIDIA's ESM-2 protein language model
