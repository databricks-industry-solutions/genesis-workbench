# Protein Binder Design

## Introduction

Protein Binder Design generates novel protein sequences that bind to a target protein using Proteina-Complexa. This enables computational design of therapeutic antibodies, enzyme inhibitors, and other protein-protein interaction modulators.

## What It Achieves

- Designs de novo protein sequences that bind a specified target protein
- Supports targeting specific residues (hotspots) on the target surface
- Optionally validates designs with ESMFold structure prediction
- Produces multiple candidate binders ranked by reward score

## How to Use

1. Navigate to **Small Molecules > Protein Binder Design** tab
2. Provide the target protein:
   - Upload a PDB file, or
   - Enter a sequence (will be folded with ESMFold first)
3. Specify the target chain ID
4. Optionally define hotspot residues (surface residues to target)
5. Set binder length range and number of designs
6. Enable ESMFold validation to predict designed structures
7. Configure MLflow tracking
8. Click **Design Binders**

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Target protein | PDB structure or amino acid sequence | PDB file or sequence string |
| Target chain | Chain ID to target | `A` |
| Hotspot residues | Optional: specific residues to bind | `R45,E78,K112` |
| Binder length | Min-max residue range | 50-100 |
| Number of designs | Candidate binders to generate | 5 |
| ESMFold validation | Predict structure of each design | Checkbox |

### Outputs

- **Designed sequences**: Novel protein sequences with reward scores
- **Predicted structures** (if ESMFold enabled): 3D structures of designed binders
- **Reward scores**: Model confidence in binding capability

## How It's Implemented

### Pipeline

```
Target protein (PDB or sequence)
  ↓
[If sequence: ESMFold → fold to PDB]
  ↓
Proteina-Complexa → generate binder designs
  ↓
[Optional: ESMFold → validate each design]
  ↓
Ranked designs with sequences and structures
```

### Key Files

- `modules/core/app/views/small_molecules.py` — UI (Protein Binder Design tab)
- `modules/core/app/utils/small_molecule_tools.py` — `hit_proteina_complexa()`, `hit_esmfold()`
- `modules/small_molecule/proteina_complexa/proteina_complexa_v1/` — Model registration
