# Vaccine Immunogen Design

## Introduction

A vaccine teaches the immune system to recognize a small patch on a pathogen — the **epitope**. That
patch, lifted out of its parent protein, is usually floppy, unstable, and hard to manufacture.
**Vaccine Immunogen Design** uses **RFD4-Proteina** (the NVIDIA×Baker flow-matching design model) loaded
in-process on an H100 to design a brand-new, stable scaffold protein that holds the epitope locked in its
native 3D conformation — *epitope-focused immunogen design* ("keep the gem fixed, design the ring around
it"). It runs as a long batch job: RFD4 **motif-scaffolds** candidate carriers that present the epitope,
then a reward-weighted loop ranks them on presentation fidelity, fold confidence, and manufacturability
and reseeds toward the best — the same shape as Antibody Design and Guided Enzyme Optimization.

## What It Achieves

- Grafts a conserved neutralizing epitope motif onto de-novo stable scaffolds (motif scaffolding —
  the epitope backbone *and* sequence are held fixed; the flanks are generated)
- Keeps the presented epitope sequence untouched while (optionally) letting ProteinMPNN re-pick a more
  expressible, manufacturable **scaffold**
- Scores every candidate on an interpretable composite reward and surfaces a ranked shortlist with 3D
  structures you can inspect and download — immunogens that present a conserved, low-mutation region
  are the basis of broader, longer-lasting vaccines (RSV / influenza / pan-coronavirus)

## How to Use

1. Open **Large Molecule → Vaccine Immunogen Design**.
2. Paste the **epitope structure** (PDB text) and set the **motif chain**.
3. Enter the **motif residues** (comma-separated residue numbers of the epitope) — or leave blank to use
   the whole chain as the motif.
4. Set the total **scaffold length window** (default 80–120; must exceed the motif length),
   **candidates per iteration (K)**, and **iterations (N)**.
5. Leave **ProteinMPNN scaffold redesign** on for a more expressible carrier (the epitope stays fixed).
6. Set the MLflow experiment + run name and click **Design immunogens**.

The job runs for ~1–4 h on an H100. The run appears immediately under **Search Past Runs** with a
progressive stage (`submitted` → `iter_N_*` → `complete`); open it with **View** once complete to see the
ranked candidates, per-axis scores, a 3D viewer, and a PDB download.

### Inputs

| Field | Description | Example |
|-------|-------------|---------|
| Epitope structure | Epitope-motif PDB text (ATOM records for the motif chain) | PDB string |
| Motif chain | Chain id the epitope motif is on | `A` |
| Motif residues | Epitope residues to preserve (CSV ints; empty = whole chain) | `254,…,277` |
| Scaffold length min/max | Total designed-scaffold length window (incl. the motif) | 80–120 |
| Candidates / iteration (K) | Scaffolds generated each iteration | 8 |
| Iterations (N) | Optimization-loop ceiling (convergence usually exits earlier) | 6 |
| ProteinMPNN scaffold redesign | Fix the epitope, re-pick the scaffold | checkbox |

The bundled demo prefills the **RSV F protein antigenic site II** peptide (RCSB `3IXT` chain P, residues
254–277), the canonical epitope-scaffolding target from Correia et al. 2014 (*Nature*, "Proof of principle
for epitope-focused vaccine design").

### Outputs

- **MLflow run** tagged `feature=vaccine_immunogen` with per-candidate metrics and iteration summaries
  (`iter_max_reward`, `iter_mean_reward`).
- **`results/reward_trajectory.csv`** — every candidate with its composite reward + per-axis scores.
- **`results/topK_pdbs/*.pdb`** — the top-ranked immunogen scaffolds (shown in the result dialog + downloadable).
- Reward axes: **epitope-presentation fidelity** (motif backbone RMSD of the folded design vs. the input
  epitope — the headline axis, minimized), scaffold fold confidence (ESMFold pLDDT), solubility (NetSolP),
  Tm (DeepSTABp), a rule-based **sequence-liability scan** (manufacturability — a weighted motif count,
  minimized; per-candidate breakdown shown as `liability_detail`), and **optional scaffold self-reactivity**
  (MHCflurry MHC-I / HLAIIPred MHC-II burden of the carrier, both off by default — a vaccine is *meant* to
  be immunogenic; opt in only to trim unwanted T-cell epitopes in the scaffold).

## How It's Implemented

### Pipeline

```
Epitope motif PDB + motif residues
  ↓
RFD4-Proteina (in-process, H100) → K scaffolds (motif-scaffolding condition_spec: C_CRD + C_SEQ on the motif)
  ↓
[optional] ProteinMPNN → redesign the SCAFFOLD (fix the epitope) → ESMFold re-fold
  ↓
Score: motif RMSD (presentation fidelity) · pLDDT · NetSolP · DeepSTABp · liability scan · [optional] MHCflurry / HLAIIPred
  ↓
Composite reward (z-score→min-max per axis, weighted) → resample → next iteration
  ↓
Ranked shortlist + top-K PDBs
```

### Key Files

- `modules/large_molecule/vaccine_immunogen/vaccine_immunogen_v1/notebooks/01_run_vaccine_immunogen.py` — orchestrator loop
- `modules/large_molecule/vaccine_immunogen/vaccine_immunogen_v1/notebooks/utils.py` — RFD4 in-process motif-scaffolding generator, sequence-located motif-RMSD, endpoint helpers, reward composer
- `modules/core/app/backend/app/services/vaccine_immunogen.py` — dispatcher + search + result loaders
- `modules/core/app/backend/app/routers/vaccine_immunogen.py` — `/api/vaccine_immunogen/*`
- `modules/core/app/frontend/src/components/VaccineImmunogenTab.tsx` — the UI tab

### Underlying models / endpoints

- **Generation:** RFD4-Proteina loaded in-process on the orchestrator's `GPU_1xH100` job (requires the
  `rfd4_proteina` submodule deployed first — the orchestrator loads its staged flow + AE checkpoints).
- **Validation / scoring endpoints:** ProteinMPNN, ESMFold, NetSolP, DeepSTABp, and (optional) MHCflurry
  (MHC-I), HLAIIPred (MHC-II). Shared with Antibody Design + Guided Enzyme Optimization. Epitope-presentation
  fidelity (motif RMSD) needs no endpoint — it is a biotite backbone superposition of the ESMFold-folded
  motif region against the input epitope (the design is folded from sequence, so the RMSD is a true
  self-consistency check).

## Limitations and known issues

- **The motif-scaffolding condition_spec is first-draft.** The epitope is centered with symmetric flexible
  flanks (`utils._flank_segments`). A terminal placement (0-length flank) or a **discontinuous**
  (multi-segment) epitope is a future refinement — a discontinuous epitope is currently preserved as a
  single contiguous range (min→max residue). Expect a deploy-time iteration or two on the conditioning,
  same caveat as the antibody_design + RFD4 register notebooks.
- Motif RMSD is computed only when the epitope sequence is found verbatim in the folded design (it is held
  fixed by `C_SEQ` and the ProteinMPNN fix); candidates where it isn't located score the axis as skipped
  (NaN) rather than crashing the iteration, and are flagged `epitope_presented=false` in the trajectory.
- **Scaffold length must exceed the motif length** (there has to be room for a scaffold around the epitope);
  the orchestrator fails fast otherwise.
- Requires the `rfd4_proteina` submodule + the ProteinMPNN/ESMFold/developability endpoints to be deployed
  and reachable; a cold GPU endpoint can add minutes to the first scoring round.
