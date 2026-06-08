# Guided Molecule Design with GenMol

## Introduction

Guided Molecule Design is a closed-loop small-molecule generator. It answers the question *"where do candidate ligands come from?"* by growing a seed scaffold (typically a binding motif) into novel drug-like molecules with NVIDIA's **GenMol** (`NV-GenMol-89M-v2`), scoring every candidate, and **reseeding** the next round from the best survivors. The loop is steered by **hard constraints** — you state the drug-likeness and toxicity you require, and only molecules that meet them are kept and optimized.

Optionally, when you supply a target protein, each candidate is folded (ESMFold) and docked (DiffDock) so that **predicted binding** joins the score that drives the loop.

## What It Achieves

- Generates novel, synthesizable SMILES from a seed scaffold (or a binding motif resolved from a gene/target)
- Enforces **hard constraints** — keep only molecules with `QED ≥ Min QED` and `ClinTox ≤ Max ClinTox`
- Iteratively improves candidates: generate → score → keep feasible → reseed from the best
- Optionally folds + docks each candidate against a target so binding affinity drives selection
- Captures the full trajectory, every explored molecule, and the final shortlist to an MLflow run
- Degrades gracefully: if no molecule satisfies the constraints, the run still completes and reports what it explored

## How to Use

1. Navigate to **Small Molecule > Guided Molecule Design**
2. Provide a **seed scaffold** as one or more SMILES (one per line). You can paste from the clipboard, or use **Find binding motif from target** (enter a gene such as `PARP1`) to auto-fill the binding motif as the seed.
3. Set the **loop budget**: number of iterations and candidates generated per iteration.
4. Set the **hard constraints**: **Min QED** (drug-likeness floor) and **Max ClinTox** (clinical-toxicity ceiling).
5. *(Optional)* Under **Dock into reward**, resolve a target gene or paste a protein sequence. It is folded with ESMFold and DiffDock binding is added to the reward. Leave empty for a QED + ADMET-only loop.
6. Set the MLflow **Experiment** and **Run name**.
7. Click **Run** — the run appears immediately in **Search Past Runs** and advances through `submitted → running → complete`.
8. Open **View** to see the reward trajectory, the valid (constraint-satisfying) candidates, and every molecule explored.

### Inputs

| Field | Description | Default |
|-------|-------------|---------|
| Seed scaffold SMILES | One or more seed fragments (the binding motif), one per line | — |
| Iterations | Generate→score→reseed rounds (1–50) | 25 |
| Candidates / iteration (K) | Molecules GenMol generates each round | 24 |
| Select top | Survivors carried into the next round as new seeds | 3 |
| Min QED | Hard constraint — keep molecules with QED ≥ this | 0.50 |
| Max ClinTox | Hard constraint — keep molecules with ClinTox ≤ this | 0.30 |
| Dock into reward (optional) | Target gene or protein sequence; folded (ESMFold) + docked (DiffDock) so binding joins the reward | empty |
| Dock / iter | Top candidates docked each iteration when a target is set | 8 |
| MLflow Experiment / Run name | Where the run is logged | `gwb_molecule_optimization` / `mol_opt_<timestamp>` |

### Outputs

- **Valid candidates** — molecules that satisfy both hard constraints, ranked by reward (shown first in the View)
- **Explored molecules** — every molecule generated across all iterations, with its QED, ClinTox, and (if docked) binding score
- **Reward trajectory** — a graph of best reward per iteration (same style as Guided Enzyme Optimization)
- **MLflow artifacts** — `top_k.json` (`{top_k, explored}`) and `trajectory.json`, plus per-iteration metrics and all input parameters

> **No candidates found?** The job does **not** fail. The View shows the reward trajectory, the full explored-molecule table, and a bold **"No candidates could be found"** message. The loop also reseeds from the *least-violating* molecules so later iterations keep searching near the constraint boundary.

## How It's Implemented

### The generate → score → reseed loop

```
Seed scaffold(s)  ──►  ┌─────────────────────────────────────────────┐
                       │  1. GENERATE  — GenMol grows each seed into  │
                       │     K candidate molecules                    │
                       │  2. SCORE     — QED (RDKit) + ClinTox/ADMET  │
                       │     (Chemprop); optional ESMFold+DiffDock    │
                       │     binding joins the reward                 │
                       │  3. FILTER    — keep "feasible" molecules     │
                       │     (QED ≥ Min QED, ClinTox ≤ Max ClinTox)   │
                       │  4. RESEED    — next round's seeds = top      │
                       │     survivors (or least-violating if none)   │
                       └───────────────┬─────────────────────────────┘
                                       │ repeat × Iterations
                                       ▼
                       top_k (feasible) + explored (all) + trajectory → MLflow
```

### Models used

| Role | Model | Endpoint / library |
|------|-------|--------------------|
| Generator | GenMol `NV-GenMol-89M-v2` (SAFE masked-diffusion) | `gwb_*_genmol_endpoint` (GPU) |
| Drug-likeness | QED | RDKit (in orchestrator) |
| Toxicity / ADMET | Chemprop ClinTox + ADMET | `gwb_*_chemprop_*` endpoints |
| Binding (optional) | ESMFold (fold target) + DiffDock (dock) | `gwb_*_esmfold_endpoint`, `gwb_*_diffdock_endpoint` |

### MLflow tracking

The run is pre-created at submit so it shows up in **Search Past Runs** instantly, tagged `feature='molecule_optimization'`. The orchestrator advances `job_status` (`submitted → running → complete`, or `failed` on error), logs per-iteration reward metrics, the full input parameters (including `qed_min`, `tox_max`, `target_gene`), and writes `top_k.json` + `trajectory.json` artifacts. Search and View read MLflow by the feature tag.

### Key Files

- `modules/core/app/backend/app/services/molecule_optimization.py` — dispatch (`start_molecule_optimization_job`), status/search, result loaders (`load_top_k` returns `{top_k, explored}`)
- `modules/core/app/backend/app/routers/small_molecule.py` — REST routes
- `modules/core/app/frontend/src/components/GuidedMoleculeOptimizationTab.tsx` — the form, reward chart, valid/explored tables, motif finder, dock-into-reward
- `modules/small_molecule/genmol/genmol_v1/notebooks/03_run_molecule_optimization.py` — the orchestrator notebook (the `run_molecule_optimization_gwb` job)
- `modules/small_molecule/genmol/genmol_v1/notebooks/01_register_genmol.py` — GenMol model registration + endpoint

### Dependencies

- GenMol Model Serving endpoint (GPU) — deploy with `./deploy.sh small_molecule <cloud> --only-submodule genmol/genmol_v1`
- Chemprop ClinTox/ADMET endpoints (scoring)
- ESMFold + DiffDock endpoints (only when **Dock into reward** is used)
