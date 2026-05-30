# Guided Enzyme Optimization (Large Molecule)

**Module:** small_molecule
**Status:** GA
**Added:** 2026-05-07

## What it does

Iteratively designs enzyme candidates that hold a fixed catalytic motif and score well against a configurable bundle of developability axes (fold confidence, substrate-complex confidence, solubility, half-life, thermostability, immunogenicity). The user supplies a motif PDB + ligand and (optionally) a substrate SMILES; the loop generates K scaffolds per iteration via Proteina-Complexa's AME, optionally redesigns each with ProteinMPNN, structure-predicts with ESMFold, scores all axes, and resamples parents weighted by the multi-axis composite reward for the next iteration. Two flavors:

- **Fast** (~30 min, no extra GPU cost) — AME runs through its existing serving endpoint; reward signal is applied *between* iterations only.
- **Accurate** (~6 hours, ~$22 GPU) — AME loads on an A10 cluster and uses Feynman-Kac steering: at intermediate denoising steps, partial structures are scored and trajectories are importance-sampled so losing branches get pruned during sampling. Better top-K candidates per dollar in theory; not yet validated head-to-head.

## How to use (UI walkthrough)

1. Open the GWB app → **Small Molecule** → **Enzyme Optimization** tab.
2. Paste a motif PDB (ATOM = catalytic residues, optional HETATM = ligand). Set the motif chain and the residue numbers (CSV) you want preserved.
3. Pick **Generation mode**: `Fast` or `Accurate`.
4. Set `K` (candidates per iteration, default 8), `N` (iteration ceiling, default 10), scaffold-length min/max, and the **ProteinMPNN** redesign checkbox.
5. (Optional) Paste a **Substrate SMILES** to enable the Boltz substrate-complex axis.
6. In the right column, adjust **per-axis reward weights** (sliders; weight 0 disables an axis). Edit the **Half-life anchor** reference enzymes if you want a non-default anchor.
7. Set **MLflow experiment + run name**. Click **Run**.
8. The dispatcher pre-creates the MLflow run (tagged `job_status=submitted`) and submits the orchestrator job. The form clears and a **Search Past Runs** row appears immediately — refresh the search to see status progression: `submitted → iter_<N>_scoring → complete` (or `failed`).
9. Once status is `complete`, click **View** on the row. The result dialog shows: live composite-reward chart, top-K candidates table, per-candidate Mol\* viewer, eight-metric grid (one per axis + composite reward), and a PDB download.

## Inputs

- **Motif PDB**: must include at least one ATOM record for the motif chain. Ligand HETATM is optional but recommended when a substrate is supplied.
- **Motif residues (CSV)**: residue numbers (within the motif chain) used for the post-redesign motif-backbone RMSD axis.
- **Scaffold length range**: AME pads the motif to a target length in `[min, max]`; each iteration samples uniformly within the range.
- **K (candidates per iteration)** and **N (iteration ceiling)**: K typically 8-16; N is a ceiling (the early-stop usually exits sooner, see *Stopping criteria* below).
- **Reference enzymes (half-life anchor)**: one or more rows of `(sequence, half_life_hours, cell_system)`. Defaults provided are T4 lysozyme (stable, ~24 h) + N-end-rule destabilized T4 lysozyme (~30 min). The loop scores candidates' PLTNUM relative-stability against these references; candidates above `min(reference) + margin` get positive reward.
- **Substrate SMILES** (optional): unlocks the Boltz axis. Without it, Boltz weight is forced to 0 even if the slider is non-zero.
- **HLA panel**: defaults to the Sette-style 6-allele panel `HLA-A*02:01, HLA-A*01:01, HLA-B*07:02, HLA-B*44:02, HLA-C*07:01, HLA-C*04:01` (~95 % global coverage). Centralized in `enzyme_optimization_tools._DEFAULT_MHC_ALLELES`.

## Outputs

For each run, logged under the MLflow run's artifacts directory:

- `iter_<i>/candidates.parquet` — every candidate's sequence + per-axis raw scores + per-axis normalized scores + composite reward.
- `iter_<i>/structures/*.pdb` — ESMFold structures for each candidate.
- `iter_<i>/redesigned/*.pdb` — ProteinMPNN redesigns (when enabled).
- `top_k.parquet` — global top-K across all iterations, ranked by composite reward.
- `metrics`: `iter_<i>_max_reward`, `iter_<i>_mean_reward`, `final_top_reward`, `elapsed_steps`, `total_candidates_scored`.
- `tags`: `origin=genesis_workbench`, `feature=enzyme_optimization`, `created_by=<user>`, `generation_mode=fast|accurate`, `job_status=<stage>`, `job_run_id=<int>`.

The result dialog reads `top_k.parquet` for the table and selects one candidate at a time to render in Mol\* + the eight-metric grid.

## Reward axes (sliders → composite reward)

| Axis | Direction | Endpoint / source |
|---|---|---|
| Motif backbone RMSD | lower is better | local Kabsch alignment after ProteinMPNN redesign |
| ESMFold pLDDT | higher is better | `ESMFold` endpoint |
| Boltz substrate-complex confidence | higher is better | `Boltz` endpoint; only contributes if substrate SMILES supplied |
| NetSolP solubility (E. coli) | higher is better | `NetSolP-1.0` endpoint |
| PLTNUM half-life (anchor-relative) | higher is better | `PLTNUM-ESM2` endpoint, scored against the reference enzyme(s) + margin |
| DeepSTABp Tm (°C) | higher is better | `DeepSTABp` endpoint |
| MHCflurry immunogenic burden | lower is better | `MHCflurry 2.x` endpoint over the HLA panel |

Each axis is z-score-then-min-max normalized within the iteration's batch before weighted sum (except half-life which is pre-normalized via the anchor sigmoid). The composite reward drives parent resampling between iterations.

## Stopping criteria

The loop is an N-iteration ceiling but normally exits earlier via the convergence stop:

- **Plateau**: composite reward's rolling improvement falls below a tolerance for `plateau_patience` consecutive iterations.
- **Strict ceiling**: `N` iterations completed.
- **Strategy `noop`** (Advanced): verification mode — skips re-generation after iter 1 so the strategy hook can be smoke-tested.

## Underlying models / endpoints

Every endpoint is resolved at call time via `genesis_workbench.models.get_endpoint_name_for_uc_model(uc_name)` from the `model_deployments` table. None are constructed client-side from env vars.

- **Proteina-Complexa** — scaffold generation (AME). UC name `proteina_complexa`. Endpoint or A10 cluster (Fast vs Accurate). Source pinned at SHA `f95f2d4` on NVIDIA's `remove_openbabel` branch (drops the GPL-2.0 Open Babel transitive dep). See `CHANGELOG.md` "proteina_no_openbabel (2026-05-22)" for the full rationale; the SHA is in `01_register_proteina_complexa.py`.
- **ProteinMPNN** — per-scaffold redesign (when enabled). UC name `protein_mpnn`.
- **ESMFold** — fold prediction. UC name `esmfold`.
- **Boltz** — substrate-complex prediction. UC name `boltz`. Only invoked when a substrate SMILES is supplied.
- **NetSolP-1.0** — E. coli solubility probability. UC name `netsolp`.
- **PLTNUM-ESM2** — protein half-life relative-stability ranking. UC name `pltnum`.
- **DeepSTABp** — melting temperature regression. UC name `deepstabp`.
- **MHCflurry 2.x** — MHC-I immunogenic burden over the HLA panel. UC name `mhcflurry`.

Detailed module reference: [`modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/README.md`](../../../small_molecule/enzyme_optimization/enzyme_optimization_v1/README.md).

## Limitations and known issues

- **Boltz axis silently no-ops without a substrate SMILES.** Even if the slider is > 0, the axis is dropped from the composite reward when no SMILES is supplied. Set the slider to 0 explicitly to make the intent clearer.
- **MHCflurry binding affinity ≠ immunogenicity.** The "immunogenic burden" axis is a strong-presenter density proxy across the default HLA panel; it does not account for T-cell receptor diversity, regulatory T-cell tolerance, or population-level HLA frequency weighting. Treat low burden as a necessary, not sufficient, condition for low immunogenicity.
- **PLTNUM is mammalian-cell-line trained.** Half-life predictions are most meaningful when the reference enzymes' `cell_system` matches the deployment target. The default references are NIH3T3 (mammalian).
- **DeepSTABp Tm is a regression, not a measurement.** Expect typical RMSE of ~7-9 °C on held-out enzymes; rank-order is more reliable than absolute values.
- **Fast vs Accurate has not been head-to-head benchmarked on this reward stack.** Accurate is wired and runs; whether it actually produces better top-K per dollar than Fast at K=8, N=10 is an open question.
- **Endpoint cold-start**: the Accurate path's first iteration adds ~3-5 min for cluster start + pip install. The dispatcher pre-creates the MLflow run so *Search Past Runs* shows the row immediately — without it, the cold-start interval looked like "the job didn't start".
- **No retry on transient endpoint failures**: a single endpoint 5xx during scoring drops that candidate from the iteration. The loop continues with K-1; the dropped candidate is logged but not retried.
- **App service principal needs `WRITE VOLUME`** on the catalog/schema to log the per-iteration artifacts. Without it, the orchestrator falls back to logging to the workspace `/tmp` and artifacts vanish on cluster shutdown.
