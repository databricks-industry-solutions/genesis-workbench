---
name: genesis-workbench
description: Deploy and manage the Genesis Workbench for life sciences generative AI applications on Databricks, including protein modeling, single-cell genomics, and BioNeMo integration.
---

# Genesis Workbench Skill

Deploy, configure, and manage [Genesis Workbench](https://github.com/databricks-industry-solutions/genesis-workbench) — an open-source platform for life sciences generative AI on Databricks.

## Overview

Genesis Workbench simplifies deploying GPU-accelerated biological AI models on Databricks infrastructure. It targets life science researchers who need advanced ML capabilities (protein folding, single-cell analysis, drug discovery) without deep infrastructure expertise.

## Core Modules

### Single Cell Module
- **scGPT** — Single-cell transformer for gene embeddings and perturbation prediction
- **scGPT Perturbation** — Zero-shot gene knockout/overexpression effect prediction using scGPT's transformer
- **SCimilarity** — Cell similarity search and cell type annotation against a 23M-cell reference database (3 endpoints: GeneOrder, GetEmbedding, SearchNearest)
- **Scanpy** — CPU-based single-cell QC, clustering, UMAP, marker gene detection, optional diffusion pseudotime
- **Rapids-SingleCell** — GPU-accelerated single-cell analysis (CUDA-optimized version of Scanpy pipeline)

**UI Workflows:**
- **Raw Processing** — QC → normalize → HVG → PCA → cluster → UMAP → markers (Scanpy or Rapids)
- **Cell Type Annotation** — Automatic cluster annotation via SCimilarity reference search with UMAP visualization
- **Cell Similarity Search** — Search 23M-cell reference for cells similar to a cluster, with disease/study breakdowns
- **Differential Expression** — Pairwise DE between clusters with volcano plot (Mann-Whitney U + BH correction)
- **Pathway Enrichment** — GO/KEGG/Reactome enrichment of cluster markers via Enrichr (gseapy)
- **Trajectory Analysis** — Diffusion pseudotime with UMAP coloring and gene expression along pseudotime
- **Perturbation Prediction** — Predict gene knockout/overexpression effects with gene selector ranked by cluster expression

### Protein Studies Module
- **ESMFold** — Fast protein structure prediction from sequence
- **AlphaFold2** — High-accuracy protein structure prediction (batch job)
- **Boltz-1** — Multi-chain protein structure prediction (protein-protein, protein-ligand complexes)
- **ProteinMPNN** — Protein sequence design from backbone structure
- **RFDiffusion** — Protein backbone generation via diffusion (inpainting)
- **ESM2 Embeddings** — 1280-D protein sequence embeddings for similarity search

**UI Workflows:**
- **Structure Prediction** — ESMFold (real-time), AlphaFold2 (batch), and Boltz (real-time, multi-chain)
- **Protein Design** — ESMFold → RFDiffusion inpainting → ProteinMPNN → ESMFold validation pipeline
- **Inverse Folding** — Standalone ProteinMPNN: paste PDB backbone → get designed sequences → auto-fold with ESMFold
- **Sequence Search** — ESM2 embeddings → vector search → Smith-Waterman alignment → ranked results

### Small Molecule Module
- **Chemprop** — Molecular property prediction (BBBP, ClinTox, ADMET)
- **DiffDock** — Molecular docking via diffusion
- **Proteina-Complexa** — Protein binder design (protein-protein, ligand, motif scaffolding)
- **NetSolP-1.0** — Protein solubility prediction in *E. coli* (BSD-3-Clause, ONNX Runtime, CPU endpoint)
- **PLTNUM-ESM2** — Protein half-life relative stability ranker (MIT, ESM-2 650M backbone, GPU_SMALL endpoint)
- **DeepSTABp** — Protein melting temperature regression in °C (MIT, ProtT5-XL backbone, GPU_SMALL endpoint)
- **MHCflurry 2.x** — MHC-I immunogenic burden prediction (Apache-2.0, peptide-MHC presentation, CPU endpoint)

**UI Workflows:**
- **Binder Design** — Proteina-Complexa protein binder generation with ESMFold validation
- **Ligand Binder Design** — Small-molecule binder design with DiffDock docking validation
- **Motif Scaffolding** — Scaffold generation with ProteinMPNN sequence optimization
- **Guided Enzyme Optimization** — Reward-weighted optimization loop around Proteina-Complexa-AME + ProteinMPNN + ESMFold. Scores each candidate on motif RMSD, pLDDT, optional Boltz substrate confidence, and four developability axes (solubility, half-life anchored vs reference enzyme, thermostability, immunogenicity). Form has a **Generation mode** toggle: **Fast** (default, ~30 min) — endpoint-based AME with parent resampling between iterations; **Accurate** (~30-60 min, ~$22 GPU) — in-process AME on an A10 with Feynman-Kac steering during diffusion (reward biases sampling, not just selection).
- **ADMET & Safety** — Multi-model property profiling (BBB penetration, toxicity, ADMET)

### Disease Biology Module
- **VCF Ingestion** — VCF-to-Delta via Glow
- **Variant Annotation** — ClinVar annotation with gene filtering
- **GWAS Analysis** — Genome-wide association studies pipeline

### BioNeMo Integration
- NVIDIA BioNeMo container infrastructure
- Pre-trained models optimized for enterprise workloads (ESM2 fine-tuning)

### Parabricks Module
- NVIDIA Parabricks GPU-accelerated genomics pipelines (alignment, variant calling)

### Access Management & Monitoring
- Security controls, endpoint management, and observability dashboards
- Start All Endpoints keep-alive feature

## Key Dependencies
- Streamlit (UI), Databricks SDK, MLflow, BioPython, PyTorch, gseapy, scipy, Plotly, parasail

## Deployment

### Prerequisites
1. A Databricks workspace with GPU cluster support
2. Cloud provider environment configured (AWS, Azure, or GCP)
3. NVIDIA EULA acceptance for GPU/BioNeMo resources

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/databricks-industry-solutions/genesis-workbench.git
   cd genesis-workbench
   ```

2. Configure your cloud environment by editing the appropriate env file (`aws.env`, `azure.env`, or `gcp.env`).

3. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

4. To tear down:
   ```bash
   ./destroy.sh
   ```

5. For cleanup of residual resources:
   ```bash
   ./cleanup.sh
   ```

**IMPORTANT:** Review the `CHANGELOG.md` for known issues and configuration notes before deploying.

**IMPORTANT:** Always use the `databricks-authentication` skill to authenticate before any Databricks operations.

**IMPORTANT:** For workspace provisioning, use the `databricks-fe-vm-workspace-deployment` skill if the demo requires GPU clusters or app integrations.

## Instructions

1. When asked to set up or demo Genesis Workbench, first authenticate using the `databricks-authentication` skill.
2. Ensure a workspace with GPU support is available. Use `databricks-fe-vm-workspace-deployment` if needed.
3. Clone the repo and configure the cloud-specific env file for the target environment.
4. Run `deploy.sh` to deploy all modules, or deploy individual modules as needed.
5. The Streamlit-based UI is served as a Databricks App — use the `databricks-apps` skill for app management if needed.
6. Use `databricks-resource-deployment` for deploying additional infrastructure (clusters, jobs, notebooks).
7. For data generation needs (e.g., synthetic biological data for demos), reference the `databricks-data-generation` skill.
8. Check `Installation.md` in the repo for detailed setup guidance and `CHANGELOG.md` for known issues.

## When to Use This Skill

- User asks about deploying biological/life sciences AI models on Databricks
- User mentions Genesis Workbench, protein folding, single-cell analysis, scGPT, AlphaFold, ESMFold, BioNeMo, or drug discovery workflows on Databricks
- User wants to demo life sciences capabilities on Databricks
- User needs to set up GPU-accelerated biological model inference
- User asks about cell type annotation, differential expression, pathway enrichment, pseudotime analysis, or gene perturbation prediction
- User asks about molecular docking, protein binder design, ADMET profiling, or inverse folding
- User asks about variant annotation, GWAS, or VCF ingestion on Databricks
- User asks about sequence search or protein similarity search
