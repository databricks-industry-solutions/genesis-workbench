# Antibody Design (VHH)

## Introduction

Antibody Design generates and optimizes single-domain antibodies (**VHH / nanobodies**) against a
target antigen epitope using **RFD4-Proteina** (the NVIDIA×Baker flow-matching design model) loaded
in-process on an H100. It runs as a long batch job: RFD4 proposes VHH candidates conditioned on the
epitope, then a reward-weighted loop ranks them on binding, fold confidence, and developability and
reseeds toward the best — the same shape as Guided Enzyme Optimization.

## What It Achieves

- Designs de-novo VHH backbones + sequences conditioned on a specified antigen epitope (hotspots)
- Preserves the binding CDR loops RFD4 designs while (optionally) letting ProteinMPNN re-pick a more
  natural, developable **framework**
- Scores every candidate on an interpretable composite reward and surfaces a ranked shortlist with
  3D structures you can inspect and download

## How to Use

1. Open **Large Molecule → Antibody Design**.
2. Paste the **antigen structure** (PDB text) and set the **antigen chain**.
3. Enter the **epitope residues** (comma-separated residue numbers on the antigen) you want the VHH
   to bind.
4. Set the **VHH length window** (default 110–130), **candidates per iteration (K)**, and
   **iterations (N)**.
5. Leave **ProteinMPNN framework redesign** and **Boltz co-fold** on for the full reward (or turn
   them off for a faster, binding-agnostic pass).
6. Set the MLflow experiment + run name and click **Design VHH antibodies**.

The job runs for ~1–4 h on an H100. The run appears immediately under **Search Past Runs** with a
progressive stage (`submitted` → `iter_N_*` → `complete`); open it with **View** once complete to see
the ranked candidates, per-axis scores, a 3D viewer, and a PDB download.

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Antigen structure | Antigen PDB text (ATOM records for the target chain) | PDB string |
| Antigen chain | Chain id to design against | `A` |
| Epitope residues | Target residues the VHH should bind (CSV ints) | `31,52,99` |
| VHH length min/max | Designed-chain length window | 110–130 |
| Candidates / iteration (K) | VHH designs generated each iteration | 8 |
| Iterations (N) | Optimization-loop ceiling (convergence usually exits earlier) | 6 |
| ProteinMPNN framework redesign | Fix the CDRs, re-pick the framework | checkbox |
| Boltz co-fold | Co-fold antigen + VHH to score binding (ipTM) | checkbox |

### Outputs

- **MLflow run** tagged `feature=antibody_design` with per-candidate metrics and iteration summaries
  (`iter_max_reward`, `iter_mean_reward`).
- **`results/reward_trajectory.csv`** — every candidate with its composite reward + per-axis scores.
- **`results/topK_pdbs/*.pdb`** — the top-ranked VHH structures (shown in the result dialog + downloadable).
- Reward axes: binding (Boltz ipTM), fold confidence (ESMFold pLDDT), solubility (NetSolP), half-life
  (PLTNUM, anchored), Tm (DeepSTABp), and immunogenic burden (MHCflurry, minimized).

## How It's Implemented

### Pipeline

```
Antigen PDB + epitope hotspots
  ↓
RFD4-Proteina (in-process, H100) → K VHH candidates (condition_spec: epitope C_HOT)
  ↓
anarcii numbering → identify CDR loops
  ↓
[optional] ProteinMPNN → redesign the FRAMEWORK (fix CDRs) → ESMFold re-fold
  ↓
Score: Boltz ipTM (binding) · pLDDT · NetSolP · PLTNUM · DeepSTABp · MHCflurry
  ↓
Composite reward (z-score→min-max per axis, weighted) → resample → next iteration
  ↓
Ranked shortlist + top-K PDBs
```

### Key Files

- `modules/large_molecule/antibody_design/antibody_design_v1/notebooks/01_run_antibody_design.py` — orchestrator loop
- `modules/large_molecule/antibody_design/antibody_design_v1/notebooks/utils.py` — RFD4 in-process generator, anarcii, endpoint helpers, reward composer
- `modules/core/app/backend/app/services/antibody_design.py` — dispatcher + search + result loaders
- `modules/core/app/backend/app/routers/antibody_design.py` — `/api/antibody_design/*`
- `modules/core/app/frontend/src/components/AntibodyDesignTab.tsx` — the UI tab

### Underlying models / endpoints

- **Generation:** RFD4-Proteina loaded in-process on the orchestrator's `GPU_1xH100` job (requires the
  `rfd4_proteina` submodule deployed first — the orchestrator loads its staged flow + AE checkpoints).
- **Validation / scoring endpoints:** ProteinMPNN, ESMFold, Boltz, NetSolP, PLTNUM, DeepSTABp, MHCflurry
  (the same serving endpoints Guided Enzyme Optimization uses).

## Limitations and known issues

- **The VHH condition_spec is first-draft.** It generates a VHH-length binder conditioned on the epitope;
  it does not yet scaffold a true Ig framework (keep framework, design only CDR loops). `anarcii` annotates
  how antibody-like each design is (`is_vhh`), and ProteinMPNN framework redesign only runs when a design
  numbers as a V-domain. Expect a deploy-time iteration or two on the conditioning.
- **VHH / nanobody only** (single domain). scFv (paired VH+VL) is a future extension.
- Requires the `rfd4_proteina` submodule + the ProteinMPNN/ESMFold/Boltz/developability endpoints to be
  deployed and reachable; a cold GPU endpoint can add minutes to the first scoring round.
