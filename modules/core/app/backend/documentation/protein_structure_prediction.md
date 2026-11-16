# Protein Structure Prediction

## Introduction

Protein Structure Prediction determines the 3D folded structure of a protein from its amino acid sequence. Genesis Workbench offers two approaches: **ESMFold** for fast, interactive predictions and **OpenFold3** for high-accuracy predictions (monomers and complexes) with optional multiple sequence alignment (MSA).

## What It Achieves

- Predicts 3D atomic coordinates of protein structures from sequence alone
- Provides per-residue confidence scores (pLDDT) indicating prediction reliability
- Outputs PDB format files viewable in the integrated Mol* 3D structure viewer
- OpenFold3 additionally provides complex-aware confidence metrics (pTM, ipTM, binder pTM, and minimum interface PAE) for multi-chain assemblies

## How to Use

### ESMFold (Fast, Interactive)

1. Navigate to **Large Molecule > Protein Structure Prediction** tab
2. Select **ESMFold** as the prediction method
3. Paste an amino acid sequence or upload a FASTA file
4. Click **Predict Structure**
5. View the 3D structure in the integrated Mol* viewer (colored by pLDDT confidence)

### OpenFold3 (High-Accuracy, Job-Based)

1. Navigate to **Large Molecule > Protein Structure Prediction** tab
2. Select **OpenFold3** as the prediction method
3. Enter the protein sequence — fold a **complex** by separating chains with a colon (`SEQA:SEQB`)
4. Leave **Use MSA** on (recommended, ColabFold server) for best accuracy, or turn it off for a faster single-sequence fold
5. Configure MLflow experiment and run names
6. Click **Start Job** — this submits a serverless-GPU Databricks job (typically a few minutes)
7. Search past runs to retrieve completed predictions and view them in the Mol* viewer

### Inputs

| Field | Description | Notes |
|-------|-------------|-------|
| Protein sequence | Amino acid sequence (single-letter codes); `:`-separated chains for a complex | Max ~1024 residues for ESMFold |
| Method | ESMFold or OpenFold3 | ESMFold is seconds; OpenFold3 is a few minutes |
| Use MSA | OpenFold3 only — fetch an MSA from the ColabFold server | Default on; off = faster single-sequence fold |

### Outputs

- **PDB structure**: 3D atomic coordinates (`ranked_0.pdb`)
- **pLDDT scores**: Per-residue confidence (0-100, higher is better)
- **Confidence metrics** (OpenFold3): pTM, ipTM, binder pTM, average pLDDT, and minimum interface PAE for complexes

## How It's Implemented

### ESMFold

- **Model**: Meta's ESMFold v1 deployed as a Databricks Model Serving endpoint
- **Backend** (`modules/core/app/backend/app/services/`): sends sequence to endpoint, receives PDB
- **Speed**: ~1-5 seconds per sequence
- **Registration**: `modules/large_molecule/esmfold/esmfold_v1/notebooks/01_register_esmfold.py`

### OpenFold3

- **Model**: OpenFold3 (AF3-style predictor, Apache-2.0) with the OpenBind-0 checkpoint, run on serverless GPU (H100)
- **MSA**: fetched on demand from the remote **ColabFold MSA server** (`use_msa` toggle, default on) — no multi-terabyte genetic databases required
- **Job**: `modules/large_molecule/openfold3/openfold3_v0.5.1/resources/run_openfold3.yml`
- **Notebook**: `run_openfold3_fold.py` (GPU fold → CIF → `ranked_0.pdb` + confidence JSON)
- **Supports**: Monomer and complex (colon-separated chain sequences)
- **Results**: `/Volumes/{catalog}/{schema}/openfold3/results/{run_id}/ranked_0.pdb`
- **Settings key**: `run_openfold3_job_id`

### Boltz-1 (Multi-chain with RNA support)

- **Model**: Boltz-1 for protein-RNA and protein-protein complexes
- **Registration**: `modules/large_molecule/boltz/boltz_1/notebooks/01_register_boltz.py`
- **Unique capability**: Supports RNA chains alongside protein chains

### Key Files

- `modules/core/app/backend/app/services/alphafold.py` — OpenFold3 job dispatch (`start_run_alphafold_job()`), run search, and result retrieval
- `modules/large_molecule/esmfold/esmfold_v1/` — ESMFold registration
- `modules/large_molecule/openfold3/openfold3_v0.5.1/` — OpenFold3 pipeline
- `modules/large_molecule/boltz/boltz_1/` — Boltz-1 registration
