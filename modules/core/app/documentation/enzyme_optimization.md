# Guided Enzyme Optimization

## Introduction

Guided Enzyme Optimization wraps the Motif Scaffolding stack (Proteina-Complexa-AME + ProteinMPNN + ESMFold) in a **reward-weighted optimization loop**. Instead of accepting whatever a single AME generation produces, each iteration scores K candidates on physical fidelity *and* developability, then biases generation toward the high-reward candidates.

This is meant for designing a real, useful enzyme — one that holds its catalytic motif, folds confidently, optionally binds a substrate, *and* survives wet-lab developability checks (solubility, half-life vs a known reference, melting point, immunogenic burden).

The form has a **Generation mode** toggle that picks where the reward signal is applied:

- **Fast** *(default, ~30 min, no GPU cost)* — AME runs as the deployed Model Serving endpoint. The loop scores K candidates after generation and resamples parents by reward for the next iteration's seed. **Reward signal applies between iterations only.**
- **Accurate** *(~30-60 min, ~$22 GPU cost)* — AME loads on an A10 GPU cluster and uses Feynman-Kac steering during sampling: at intermediate denoising steps, partial structures are scored and trajectories are importance-sampled so losing branches get pruned early. **Reward signal applies during diffusion**, not just at iteration boundaries.

Both modes share the same scoring axes, the same UI form, the same MLflow output shape, and the same predictor endpoints. The toggle only changes how the reward signal influences generation.

## What It Achieves

- Generates novel enzyme scaffolds carrying a transplanted catalytic motif
- Scores every candidate on up to **seven axes** simultaneously, with user-tunable weights
- **Anchors half-life prediction against a known reference enzyme** so the half-life axis enforces a real, defensible threshold instead of a vague ranking
- Logs the full trajectory to MLflow so you can re-rank later or feed the top-K straight into a wet-lab queue
- Lets you trade off cost vs quality at submission time (Fast for iteration / hypothesis-screening, Accurate for production runs)

## Optimization Axes

| Axis | Source | Direction | Notes |
|---|---|---|---|
| Motif backbone RMSD | Bio.PDB superposition of motif residues vs the input | Lower better | Penalizes catalytic-site drift after redesign |
| ESMFold pLDDT | Mean CA pLDDT from the post-redesign ESMFold | Higher better | Global fold confidence |
| Boltz substrate confidence | Boltz iLDDT/ipTM with input protein + substrate SMILES | Higher better | Only contributes if substrate SMILES is supplied |
| NetSolP solubility | NetSolP-1.0 (BSD-3-Clause) | Higher better | *E. coli* solubility prob in [0, 1] |
| PLTNUM half-life (anchor-relative) | PLTNUM-ESM2 + sigmoid against `min(reference) + margin` | Higher better | Set weight 0 to drop. Without a reference, falls back to a flat 0.5 |
| DeepSTABp Tm | DeepSTABp ProtT5 + MLP | Higher better | Predicted melting temperature in °C |
| MHCflurry immunogenic burden | Sliding 9-mer scan × 6-allele HLA panel; strong-presenter density | Lower better | Default panel covers ~95% of global population |

## How to Use

1. Navigate to **Small Molecules → Guided Enzyme Optimization**.
2. Paste your motif PDB (catalytic site + optional ligand HETATMs); set the chain id and motif residue numbers.
3. **Pick the Generation mode** — Fast for iteration / sanity-checking, Accurate for production runs where top-K quality matters.
4. Set scaffold length range, K (candidates per iteration), N (iterations).
5. (Optional) Enter a substrate SMILES — gates the Boltz axis.
6. (Optional but recommended) Add ≥ 1 reference enzyme in the **Half-life anchor** section: paste the sequence, the measured half-life in hours, and the cell system the half-life was measured in. Adjust the margin (default 0.05).
7. Open the **Per-axis reward weights** panel and tune sliders. Weight 0 disables that axis silently.
8. Click **Launch optimization job**. The page switches to a polling view while the orchestrator job runs.

### Inputs

| Field | Description | Default |
|-------|-------------|---------|
| Motif PDB | Catalytic motif + optional ligand | provided template |
| Motif chain | Chain id of the motif residues in the input PDB | `B` |
| Motif residues (CSV) | Residue numbers used for backbone-RMSD scoring | `1,2,3` |
| Generation mode | Fast (endpoint-based) / Accurate (in-process FK steering) | Fast |
| Scaffold length min/max | Range for AME generator | 80 / 120 |
| K (candidates / iteration) | AME samples per round | 8 |
| N (iterations) | Loop length (ceiling — convergence-stop usually exits earlier) | 10 |
| Substrate SMILES | Optional — gates Boltz | empty |
| Reference enzymes | sequence + half-life (h) + cell system | T4 lysozyme prefilled |
| Half-life margin | β for the anchor sigmoid | 0.05 |
| Per-axis weights | 7 sliders, 0–5 | 1.0 each (Boltz 0.5) |
| Strategy | `resample` / `noop` | `resample` |
| Resampling temperature | Softmax τ for parent selection (Fast only) | 0.10 |
| Run ProteinMPNN | Redesign each scaffold's sequence post-AME | true |
| Stopping criteria | Convergence (default ON), Reward threshold (opt-in), Best-K cap (opt-in) | Convergence threshold 0.01 over a 2-iter window |

### Outputs

- **Live `iter_max_reward` / `iter_mean_reward` line chart** as the loop runs.
- **Top-25 candidates table** with composite reward + every per-axis score.
- **Mol\* viewer** overlaying the input motif and the selected designed scaffold; per-candidate PDB download button.
- **MLflow run** with full trajectory CSV (`results/reward_trajectory.csv`), top-K PDBs (`results/topK_pdbs/*.pdb`), per-iteration / per-candidate metrics, and the `generation_mode` (Fast / Accurate) + `use_inprocess_ame` (bool) params so two runs can be diffed side-by-side.

## Generation Modes — Detailed

### Fast mode (default)

Runs as the **`run_enzyme_optimization_gwb`** Databricks job on a CPU cluster. AME stays on its serving endpoint (`gwb_*_proteina_complexa_ame_endpoint`); the orchestrator notebook calls it via the SDK each iteration.

```
Per iteration (Fast):
  AME endpoint  → K scaffolds (no reward awareness)
       ↓
  ProteinMPNN endpoint  → redesign each scaffold's sequence
       ↓
  ESMFold endpoint  → predicted PDB + per-residue pLDDT
       ↓
  Score: motif_rmsd, plddt, boltz?, solubility, half_life, thermostab, immuno
       ↓
  Composite reward = Σ wᵢ · zscore_minmax(scoreᵢ)
       ↓
  Resample parents ~ softmax(reward / temperature)  ← reward signal
       ↓                                              applies HERE
  next iteration's AME generation seeded from parents
```

The selection pressure shows up *between* iterations: the next round's AME draws from the resampled parents, but the AME generation itself is not aware of developability. With K=8 and N=3, the implicit hypothesis is "given enough K samples, a good one will randomly turn up; the reward bias keeps re-rolling the dice toward the good neighborhood."

**Cost:** $0 GPU on the orchestrator (the cluster is CPU; all compute is on already-deployed endpoints). Dispatch is free.
**Wall-clock:** ~7-30 min depending on K, N, and endpoint warm/cold state.

### Accurate mode

Runs as the **`run_enzyme_optimization_gwb_inprocess_ame`** Databricks job on an A10 GPU cluster. AME is **loaded into the orchestrator's own Python process** rather than called as an endpoint. This unlocks Proteina-Complexa's **Feynman-Kac steering** search algorithm, which biases the diffusion process *during* generation.

```
At job start (one-time):
  Install proteinfoundation + transitive deps (~3-5 min)
       ↓
  dbutils.library.restartPython()  (so the new torch ABI takes over)
       ↓
  Pull AME .ckpt files from the registered UC model
  ({catalog}.{schema}.proteina_complexa_ame, version from GWB models table)
       ↓
  Proteina.load_from_checkpoint(...)  → AME in memory
       ↓
  Warmup all 8 endpoints (developability + ESMFold + ProteinMPNN)

Per iteration (Accurate):
  FK-steering search on AME, with DevelopabilityCompositeReward attached:
       beam_width × n_branch trajectories at each checkpoint
                     ↓
       At checkpoint t (e.g. step 25, 50, 75 of 100):
         partial structure → quick fold → reward score (NetSolP, PLTNUM,
                                          DeepSTABp, MHCflurry)
                     ↓
         importance-weighted multinomial: kill losing trajectories,
                                          replicate winners
       ↑                                            ← reward signal
       └────── continue to next checkpoint                applies HERE
                     ↓
  Final K samples (post-FK-steering)
       ↓
  ProteinMPNN endpoint  → redesign each scaffold's sequence
       ↓
  ESMFold endpoint  → predicted PDB + pLDDT
       ↓
  Score (full registry, same as Fast)
       ↓
  Log to MLflow; resample parents for next iteration
```

The reward signal influences AME's diffusion *during* sampling, not just selection between iterations. The hypothesis: "kill losing trajectories at step 25 instead of running them through to step 100, allocating compute to developability-promising branches."

**Why it costs more:** the A10 GPU cluster runs the entire job (~$3.60/hour). FK-steering with `beam_width=4, n_branch=4, step_checkpoints=[0,25,50,75,100]` runs ~16 partial-rollout trajectories per AME call — each with 4 reward calls (one per developability endpoint).

**Wall-clock:** ~30-60 min for K=4, N=2 — verified at 46 min on the catalytic-triad smoke motif with all axes at weight 1.0.

**Verified delta vs Fast** (K=4, N=2, same motif, same anchor): Fast `iter_mean=0.449` → Accurate `iter_mean=0.506`. iter_max is similar across both modes; what changes is that the *whole batch* gets pulled toward higher reward in Accurate, instead of mostly relying on the top-1 lucky draw.

### When to pick which

| Use case | Mode |
|---|---|
| First exploration of a new motif / weight stack | Fast |
| Iterating on per-axis weights to find a good balance | Fast |
| "Test predictors on T4 lysozyme" smoke run | Fast — N/A on Accurate (predictor smoke is endpoint-only) |
| Production run where you'll feed top-K to wet-lab | **Accurate** |
| You only care about the single top candidate, not the batch | Either — iter_max is similar |
| You want the *batch* to skew toward developability | **Accurate** |
| You don't have a reference enzyme for the half-life anchor | Either — the half-life axis falls back to a flat 0.5 in both |

## Honest Caveats

- **Half-life axis is anchor-relative, not in hours.** Top candidates are predicted to be at least as long-lived as your reference enzyme + margin. There is no permissively-licensed open ML model in 2026 that returns half-life in hours.
- **The Boltz axis only activates** when you supply a substrate SMILES; otherwise it's silently dropped from the composite reward.
- **Fast mode's first iteration is uniform** — the reward signal can't influence sampling until iteration 2. Run at least N=2 iterations to see any optimization effect. **Accurate mode applies the reward inside iteration 1's AME generation** via FK-steering.
- **Accurate mode requires `proteina_complexa/proteina_complexa_v1` to be deployed first** — its UC model is the only source for the AME checkpoints. If not deployed, the orchestrator raises with a clear "deploy proteina_complexa first" message (no NGC fallback).
- **Predictions are biased ranking signals, not guarantees.** Always validate top candidates wet-lab.

## How It's Implemented

### Architecture

Two Databricks jobs share the same orchestrator notebook (`01_run_optimization.py`); the toggle drives the dispatcher to pick the right job by name:

| Toggle | Job name | Cluster |
|---|---|---|
| Fast (default) | `run_enzyme_optimization_gwb` | CPU |
| Accurate | `run_enzyme_optimization_gwb_inprocess_ame` | A10 GPU |

Both jobs are configured for **on-demand** compute on every cloud — `aws_attributes.availability: ON_DEMAND` / `azure_attributes.availability: ON_DEMAND_AZURE` / `gcp_attributes.availability: ON_DEMAND_GCP`. This matches the GWB pattern (see `boltz/boltz_1/databricks.yml`); spot reclamation is unsuitable for the multi-hour Accurate runs.

### Anchor Mechanism for Half-Life

PLTNUM is a relative ranker — its raw output is a probability, not hours. To turn it into a real signal:

1. At job start, score every reference enzyme with PLTNUM → `S_threshold = min(reference scores) + margin`.
2. Per candidate, `half_life_reward = sigmoid((PLTNUM_candidate − S_threshold) / β)`.
3. Soft-prior, not hard-prune — early iterations don't starve when no candidate beats the anchor yet.

The anchor lets you make a defensible probabilistic claim: *top candidates are predicted to be at least as long-lived as your reference enzyme, with margin*. That's honest and useful, instead of pretending we predict hours.

### Endpoint Warmup at Job Start

Every developability endpoint has scale-to-zero enabled — the first call after an idle period hits a 5-20 min cold start. Without warming up at job start, the first mid-loop call to a cold endpoint busts the request timeout. The orchestrator pre-warms all eight endpoints (NetSolP, PLTNUM, DeepSTABp, MHCflurry, AME, ESMFold, ProteinMPNN, Boltz when relevant) with one dummy call each before the loop runs. This applies to **both** Fast and Accurate paths.

### FK-Steering Reward Hook (Accurate path)

The custom reward class `DevelopabilityCompositeReward` inherits from upstream Proteina-Complexa's `BaseRewardModel` (`proteinfoundation.rewards.base_reward.BaseRewardModel`). Its `score(pdb_path, requires_grad=False, **kwargs)`:

1. Extracts the AA sequence from the candidate PDB the search algorithm passes in.
2. Calls NetSolP / PLTNUM / DeepSTABp / MHCflurry serving endpoints in series.
3. Applies the half-life anchor sigmoid to PLTNUM's raw score.
4. Returns the standardized `{reward: dict[str, Tensor], grad: dict, total_reward: Tensor}` dict.
5. Per-axis fallback: any failed endpoint contributes 0 instead of crashing the search.

The reward instance is pre-built once per job from the user's weights + pre-computed anchor threshold, then attached to the loaded model via `model.reward_model = reward` directly (Proteina's `configure_inference()` only takes `(inf_cfg, nn_ag)`, not a reward kwarg).

### Stopping Criteria

The N parameter is a *ceiling*, not a fixed budget. Three opt-in stop modes:

- **Convergence stop (ON by default)** — exits early when `iter_max_reward` improvement over the last `convergence_window` iterations is below `convergence_threshold` (defaults: 0.01 over 2 iters).
- **Reward threshold stop (opt-in)** — exits the moment any candidate's composite reward exceeds `target_reward`.
- **Best-K cap (opt-in)** — exits once `best_k_target` candidates with reward ≥ `best_k_threshold` have been found.

A `stop_reason` MLflow tag records which one fired. Setting `convergence_threshold` to a negative number disables convergence-stop; leaving the other two empty disables them.

### Strategy Hook

The "next-iteration parent generator" is a strategy interface so Phase 2 can swap in mutation operators without touching the orchestrator core. Phase 1 ships:

- `ResampleFromAMEStrategy` (default) — softmax-weighted resampling of high-reward parents, then re-call AME with unchanged length bounds.
- `NoOpStrategy` (verification) — returns `None` so the loop can be smoke-tested without re-running AME.

### Key Files

- `modules/core/app/views/small_molecule_workflows/enzyme_optimization.py` — UI (the form + Generation-mode toggle + polling results view)
- `modules/core/app/utils/enzyme_optimization_tools.py` — `start_enzyme_optimization_job` (dispatches to Fast or Accurate job by toggle), `predict_enzyme_properties` (smoke test), `load_optimization_*` result loaders
- `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/notebooks/01_run_optimization.py` — the orchestrator notebook, branches on `use_inprocess_ame`
- `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/notebooks/utils.py` — predictor registry, reward composer, anchor mechanism, strategy interface, **and** all the Accurate-path helpers (`DevelopabilityCompositeReward`, `load_ame_model`, `_fetch_ame_checkpoints_from_uc`, `_resolve_ame_uc_version`, `install_proteinfoundation_if_needed`, `run_ame_with_rewards`, `warmup_*`)
- `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/notebooks/proteinfoundation_requirements.txt` — exact-pinned deps for the Accurate path's in-process AME load
- `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/resources/run_optimization.yml` — both job specs (CPU + A10 GPU)
- `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/databricks.yml` — per-cloud on-demand overlays for both jobs
- Predictor submodules under `modules/small_molecule/{netsolp,pltnum,deepstabp,mhcflurry}/`
- `modules/core/app/utils/structure_utils.py:motif_backbone_rmsd` — Bio.PDB-based motif RMSD helper
- `modules/core/app/utils/protein_design.py:hit_boltz` — long-timeout Boltz client (the SDK's 60s default was killing cold-start GPU calls)
