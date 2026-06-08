# Inverse Folding with ProteinMPNN

## Introduction

Inverse Folding is the reverse of structure prediction: given a protein **backbone** (a 3D structure with no sequence, or whose sequence you want to redesign), it generates new amino-acid sequences predicted to fold into that backbone. Genesis Workbench uses **ProteinMPNN** for the design step, then **validates** each design by folding it back with ESMFold so you can immediately see whether the proposed sequence reproduces the target shape.

## What It Achieves

- Designs multiple candidate sequences for a fixed protein backbone (PDB)
- Validates each candidate by re-folding it with ESMFold (round-trip check)
- Lets you inspect every design's predicted structure in an interactive 3D viewer
- Captures the designed sequences to an MLflow run for provenance and later retrieval

## How to Use

1. Navigate to **Large Molecule > Inverse Folding**
2. Paste a **backbone PDB** (ATOM records). A default example is pre-filled.
3. Set the MLflow **Experiment** and **Run name** (defaults are pre-populated like other workflows).
4. Click **Design Sequences** — ProteinMPNN returns a set of candidate sequences.
5. Select any design from the dropdown; it is auto-folded by ESMFold and rendered in the 3D viewer.
6. Open the linked **MLflow run** to retrieve the full set of designs later.

### Inputs

| Field | Description | Default |
|-------|-------------|---------|
| PDB content | Backbone structure (ATOM records) to redesign | example backbone |
| MLflow Experiment | Experiment the designs are logged to | `gwb_inverse_folding` |
| Run name | Run name for this design batch | `inverse_folding_<timestamp>` |

### Outputs

- **Designed sequences** — multiple ProteinMPNN candidates for the backbone
- **Validated structures** — each selected design folded by ESMFold, shown in the Mol* viewer
- **MLflow artifacts** — `designs.fasta`, `designs.json`, and `backbone.pdb`, logged under the run (tag `feature='inverse_folding'`)

## How It's Implemented

```
Backbone PDB ──► ProteinMPNN ──► N candidate sequences ──► (select one)
                                                              │
                                                              ▼
                                                          ESMFold ──► predicted structure (3D viewer)
                                                              │
                                              designs.fasta / designs.json / backbone.pdb → MLflow
```

ProteinMPNN is a message-passing neural network that conditions on backbone geometry to sample sequences likely to fold into it. ESMFold then provides a fast single-sequence structure prediction (no MSA) as a round-trip sanity check on each design.

### Key Files

- `modules/core/app/backend/app/routers/large_molecule.py` — `InverseFoldingRequest`/`InverseFoldingResponse`, the route, and `_log_inverse_folding` (MLflow capture)
- `modules/core/app/frontend/src/components/InverseFoldingTab.tsx` — the form, design selector, and auto-fold viewer
- `modules/large_molecule/protein_mpnn/protein_mpnn_v0.1.0` — ProteinMPNN model
- `modules/large_molecule/esmfold/esmfold_v1` — ESMFold validation model

### Dependencies

- ProteinMPNN Model Serving endpoint
- ESMFold Model Serving endpoint (validation fold; runs on a GPU_MEDIUM / A10 endpoint)
