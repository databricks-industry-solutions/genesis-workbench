# Functional Motif Scaffolding

## Introduction

Functional Motif Scaffolding transplants a functional motif (such as an enzyme active site or binding loop) into a new protein scaffold using Proteina-Complexa-AME. This enables creating novel proteins that carry a desired functional region in a stable, well-folded scaffold.

## What It Achieves

- Takes a functional motif (from a PDB structure) and generates new protein scaffolds around it
- Preserves the 3D geometry of the motif while creating a completely new surrounding structure
- Optionally optimizes the scaffold sequence with ProteinMPNN
- Optionally validates the final design with ESMFold

## How to Use

1. Navigate to **Small Molecules > Motif Scaffolding** tab
2. Upload a motif PDB file (with optional ligand as HETATM records)
3. Specify the motif chain ID
4. Set scaffold length range and number of designs
5. Optionally enable:
   - **ProteinMPNN optimization**: Optimize scaffold sequence for stability
   - **ESMFold validation**: Predict structure of final design
6. Configure MLflow tracking
7. Click **Generate Scaffolds**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Motif PDB | PDB with functional motif (optional HETATM ligand) | Upload PDB file |
| Motif chain | Chain ID containing the motif | `A` |
| Scaffold length | Min-max residue range | 100-200 |
| Number of scaffolds | Candidates to generate | 5 |
| ProteinMPNN optimization | Optimize sequence for stability | Checkbox |
| ESMFold validation | Predict final structure | Checkbox |

### Outputs

- **Scaffold designs**: Novel proteins containing the transplanted motif
- **Optimized sequences** (if ProteinMPNN enabled): Sequences optimized for foldability
- **Predicted structures** (if ESMFold enabled): 3D structures of final designs

## How It's Implemented

### Pipeline

```
Motif PDB
  ↓
Proteina-Complexa-AME → generate scaffolds with transplanted motif
  ↓
[Optional: ProteinMPNN → optimize scaffold sequence]
  ↓
[Optional: ESMFold → validate structure]
  ↓
Final scaffold designs
```

### Key Files

- `modules/core/app/views/small_molecules.py` — UI (Motif Scaffolding tab)
- `modules/core/app/utils/small_molecule_tools.py` — `hit_proteina_complexa_ame()`, `_hit_proteinmpnn()`, `hit_esmfold()`
