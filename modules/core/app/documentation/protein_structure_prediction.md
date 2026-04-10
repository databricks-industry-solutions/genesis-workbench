# Protein Structure Prediction

## Introduction

Protein Structure Prediction determines the 3D folded structure of a protein from its amino acid sequence. Genesis Workbench offers two approaches: **ESMFold** for fast predictions and **AlphaFold2** for high-accuracy predictions with multiple sequence alignment (MSA).

## What It Achieves

- Predicts 3D atomic coordinates of protein structures from sequence alone
- Provides per-residue confidence scores (pLDDT) indicating prediction reliability
- Outputs PDB format files viewable in the integrated Mol* 3D structure viewer
- AlphaFold2 additionally provides predicted aligned error (PAE) matrices for domain-level confidence

## How to Use

### ESMFold (Fast, Interactive)

1. Navigate to **Protein Studies > Protein Structure Prediction** tab
2. Select **ESMFold** as the prediction method
3. Paste an amino acid sequence or upload a FASTA file
4. Click **Predict Structure**
5. View the 3D structure in the integrated Mol* viewer (colored by pLDDT confidence)

### AlphaFold2 (High-Accuracy, Job-Based)

1. Navigate to **Protein Studies > Protein Structure Prediction** tab
2. Select **AlphaFold2** as the prediction method
3. Enter the protein sequence
4. Configure MLflow experiment and run names
5. Click **Start AlphaFold Job** — this submits a Databricks job (takes minutes to hours depending on sequence length)
6. Search past runs to retrieve completed predictions

### Inputs

| Field | Description | Notes |
|-------|-------------|-------|
| Protein sequence | Amino acid sequence (single-letter codes) | Max ~1024 residues for ESMFold |
| Method | ESMFold or AlphaFold2 | ESMFold is seconds; AlphaFold2 is minutes-hours |

### Outputs

- **PDB structure**: 3D atomic coordinates
- **pLDDT scores**: Per-residue confidence (0-100, higher is better)
- **PAE matrix** (AlphaFold2 only): Predicted aligned error between residue pairs

## How It's Implemented

### ESMFold

- **Model**: Meta's ESMFold v1 deployed as a Databricks Model Serving endpoint
- **Backend** (`modules/core/app/utils/protein_structure.py`): `hit_esmfold()` sends sequence to endpoint, receives PDB
- **Speed**: ~1-5 seconds per sequence
- **Registration**: `modules/protein_studies/esmfold/esmfold_v1/notebooks/01_register_esmfold.py`

### AlphaFold2

- **Pipeline**: Two-stage (CPU featurization + GPU folding)
  1. **Featurize**: MSA search against UniRef90, BFD, MGnify databases
  2. **Fold**: Neural network inference on GPU
- **Job**: `modules/protein_studies/alphafold/alphafold_v2.3.2/resources/run_alphafold.yml`
- **Notebooks**: `run_alphafold_featurize.py` (CPU) → `run_alphafold_fold.py` (GPU)
- **Supports**: Monomer and multimer (colon-separated chain sequences)

### Boltz-1 (Multi-chain with RNA support)

- **Model**: Boltz-1 for protein-RNA and protein-protein complexes
- **Registration**: `modules/protein_studies/boltz/boltz_1/notebooks/01_register_boltz.py`
- **Unique capability**: Supports RNA chains alongside protein chains

### Key Files

- `modules/core/app/views/protein_studies.py` — UI
- `modules/core/app/utils/protein_structure.py` — `hit_esmfold()`, `start_run_alphafold_job()`
- `modules/protein_studies/esmfold/esmfold_v1/` — ESMFold registration
- `modules/protein_studies/alphafold/alphafold_v2.3.2/` — AlphaFold2 pipeline
- `modules/protein_studies/boltz/boltz_1/` — Boltz-1 registration
