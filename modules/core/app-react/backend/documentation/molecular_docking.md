# Molecular Docking with DiffDock

## Introduction

Molecular Docking predicts how a small molecule (ligand) binds to a target protein using DiffDock, a diffusion-based generative model. It generates multiple candidate binding poses ranked by confidence score, enabling virtual screening and structure-based drug design.

## What It Achieves

- Predicts 3D binding poses for protein-ligand complexes
- Generates multiple candidate poses (1-20) ranked by confidence
- Provides interactive 3D visualization of docked complexes
- Does not require pre-defined binding site — DiffDock predicts the binding location

## How to Use

1. Navigate to **Small Molecules > Molecular Docking** tab
2. Enter the ligand as a SMILES string (e.g., `CC(C)Cc1ccc(cc1)C(C)C(=O)O` for ibuprofen)
3. Provide the target protein as PDB content
4. Set the number of poses to generate (1-20, default 5)
5. Configure MLflow experiment and run names
6. Click **Run DiffDock**
7. View ranked poses in the 3D viewer, select individual poses to inspect

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Ligand SMILES | Small molecule in SMILES notation | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |
| Protein PDB | Target protein structure (PDB format) | Paste or upload PDB content |
| Number of poses | Poses to generate per complex | 5 |

### Outputs

- **Ranked poses**: Each with a confidence score (0-1, higher is better)
- **SDF files**: Ligand coordinates for each pose
- **3D visualization**: Interactive Mol* viewer showing protein-ligand complex

## How It's Implemented

### Pipeline (3 Steps)

```
Step 1: ESM Embeddings (~200ms)
  └→ Compute protein embeddings via ESM endpoint

Step 2: DiffDock Inference (~1.5s)
  └→ Generate and rank binding poses via diffusion model

Step 3: Results Processing (~2s)
  └→ Parse SDF outputs, create visualization, log to MLflow
```

### Model Architecture

DiffDock uses a diffusion generative model that:
1. Starts with random ligand placement
2. Iteratively denoises translation, rotation, and torsion angles
3. Produces multiple candidate poses per complex
4. Scores each pose with a learned confidence model

### Key Files

- `modules/core/app/views/small_molecules.py` — UI (Molecular Docking tab)
- `modules/core/app/utils/small_molecule_tools.py` — `hit_diffdock()`, `hit_esm_embeddings()`
- `modules/small_molecule/diffdock/diffdock_v1/notebooks/01_register_diffdock.py` — Model registration

### Dependencies

- DiffDock Model Serving endpoint (GPU)
- ESM Embeddings endpoint (for protein context)
