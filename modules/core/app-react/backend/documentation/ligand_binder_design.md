# Ligand Binder Design

## Introduction

Ligand Binder Design generates novel protein sequences that bind a specific small molecule (ligand) using Proteina-Complexa-Ligand. This enables design of enzymes, biosensors, and receptors for target molecules.

## What It Achieves

- Designs de novo proteins that bind a specified small molecule
- Accepts ligands as SMILES strings or PDB files with HETATM records
- Optionally validates designs with ESMFold (structure prediction) and DiffDock (docking confirmation)
- Produces multiple candidate designs ranked by reward score

## How to Use

1. Navigate to **Small Molecules > Ligand Binder Design** tab
2. Provide the ligand:
   - Enter a SMILES string, or
   - Upload a PDB file containing HETATM records
3. Set protein length range and number of designs
4. Optionally enable:
   - **ESMFold validation**: Predict structure of each designed protein
   - **DiffDock validation**: Confirm the designed protein actually binds the ligand
5. Configure MLflow tracking
6. Click **Design Ligand Binders**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Ligand | SMILES string or PDB with HETATM | `CC(=O)Oc1ccccc1C(=O)O` (aspirin) |
| Protein length | Min-max residue range | 80-150 |
| Number of designs | Candidates to generate | 5 |
| ESMFold validation | Predict designed structures | Checkbox |
| DiffDock validation | Confirm binding with docking | Checkbox |

### Outputs

- **Designed protein sequences**: Novel sequences with reward scores
- **Predicted structures** (if ESMFold enabled): 3D structures
- **Docking poses** (if DiffDock enabled): Predicted protein-ligand binding poses with confidence scores

## How It's Implemented

### Pipeline

```
Ligand (SMILES or PDB)
  ↓
Proteina-Complexa-Ligand → generate protein designs
  ↓
[Optional: ESMFold → fold each design]
  ↓
[Optional: DiffDock → dock ligand against each design]
  ↓
Ranked designs with sequences, structures, and docking results
```

### Key Files

- `modules/core/app/views/small_molecule_workflows/ligand_binder_design.py` — UI
- `modules/core/app/utils/small_molecule_tools.py` — `hit_proteina_complexa_ligand()`, `hit_esmfold()`, `hit_diffdock()`
