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
- **scGPT** — Single-cell transformer for genomics analysis
- **SCimilarity** — Cell similarity and annotation

### Protein Studies Module
- **ESMFold** — Protein structure prediction
- **AlphaFold2** — Protein structure modeling
- **ProteinMPNN** — Protein sequence design
- **RFDiffusion** — Protein structure generation via diffusion
- **Boltz-1** — Multi-chain protein structure prediction

### BioNeMo Integration
- NVIDIA BioNeMo container infrastructure
- Pre-trained models optimized for enterprise workloads

### Access Management & Monitoring
- Security controls and observability dashboards

## Key Dependencies
- Streamlit (UI), Databricks SDK, MLflow, BioPython, PyTorch ecosystem

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
