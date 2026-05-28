# Protein Design

## Introduction

Protein Design generates novel protein sequences and structures by redesigning specified regions of existing proteins. It combines three AI models in a pipeline: RFDiffusion for backbone generation, ProteinMPNN for sequence design, and ESMFold for validation.

## What It Achieves

- Takes a protein sequence with a marked region to redesign (inpainting)
- Generates a new backbone geometry for the specified region using diffusion modeling
- Designs an optimal amino acid sequence for the new backbone
- Validates the final design by predicting its 3D structure
- Produces aligned visualizations comparing original and redesigned structures

## How to Use

1. Navigate to **Protein Studies > Protein Design** tab
2. Enter a protein sequence with the region to redesign marked as `[REGION_TO_REPLACE]`
   - Example: `MKTAYIAK[REGION_TO_REPLACE]FLEEHPGG`
3. Configure MLflow experiment and run names
4. Click **Start Design**
5. View the designed sequence and aligned 3D structures (original vs redesigned)

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Sequence | Amino acid sequence with `[REGION_TO_REPLACE]` marker | `MKTAYIAK[REGION_TO_REPLACE]FLEEHPGG` |
| Region length | Automatically determined from marker position | Typically 5-30 residues |

### Outputs

- **Designed sequence**: Novel amino acid sequence for the replaced region
- **3D structures**: Original and redesigned structures aligned for comparison
- **Confidence scores**: pLDDT from ESMFold validation

## How It's Implemented

### Pipeline (4 Steps)

```
Input sequence with [REGION_TO_REPLACE]
  ↓
Step 1: ESMFold — fold original sequence to get starting structure
  ↓
Step 2: RFDiffusion — generate new backbone for the marked region (inpainting)
  ↓
Step 3: ProteinMPNN — design optimal sequence for the new backbone
  ↓
Step 4: ESMFold — validate the final designed sequence
  ↓
Output: Designed sequence + aligned structures
```

### Models Used

| Model | Role | Technology |
|-------|------|------------|
| ESMFold | Fold original + validate design | Meta ESMFold v1 |
| RFDiffusion | Generate new backbone geometry | Baker Lab diffusion model |
| ProteinMPNN | Inverse design: backbone → sequence | Baker Lab MPNN |

### RFDiffusion Inpainting

- Takes a PDB structure and residue range to redesign
- Uses contig statements to define fixed vs. redesigned regions (e.g., `A1-11/10-10/A22-50`)
- Generates new backbone coordinates via iterative denoising (20 diffusion steps)

### ProteinMPNN Inverse Design

- Takes backbone-only PDB (N, CA, C, O atoms)
- Generates 3 candidate sequences per backbone
- Sampling temperature controls diversity vs. conservatism

### Key Files

- `modules/core/app/views/protein_studies.py` — UI (Protein Design tab)
- `modules/protein_studies/rfdiffusion/rfdiffusion_v1.1.0/notebooks/01_register_rfdiffusion.py` — RFDiffusion registration
- `modules/protein_studies/protein_mpnn/protein_mpnn_v0.1.0/notebooks/01_register_proteinmpnn.py` — ProteinMPNN registration
- `modules/protein_studies/esmfold/esmfold_v1/notebooks/01_register_esmfold.py` — ESMFold registration
