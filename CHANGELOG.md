# Genesis Workbench — Changelog

## guided_enzyme_creation (2026-05-07) — Phase 1.6: Batch-workflow pattern + Variant Annotation per-run tables

### Batch-workflow pattern, formalized

The "form → job → MLflow → search past runs → result dialog" shape that AlphaFold, Disease Biology (GWAS / Variant Annotation), and Scanpy already used is now an explicit, named pattern — and Guided Enzyme Optimization has been retrofitted onto it. New skill file `claude_skills/SKILL_GENESIS_WORKBENCH_BATCH_WORKFLOW_PATTERN.md` codifies the five layers (orchestrator job, registration, dispatcher, search past runs, result dialog) with the canonical reference being the enzyme-optimization stack. `CLAUDE.md` adds a row pointing to it.

This is the third pass on this pattern. Each retrofit fixed a real bug from the prior shape:

- **MLflow run is pre-created from the dispatcher** (Disease-Biology pattern, now adopted by enzyme_optimization). Without it, cluster cold-start + `pip install` (3–5 min on the Accurate path) leaves *Search Past Runs* empty even though the job is in flight. The dispatcher tags `origin=genesis_workbench`, `feature=enzyme_optimization`, `created_by=<user>`, `job_status=submitted`, logs `generation_mode` + scaffold/sample/iteration params, then passes the new `mlflow_run_id` job parameter so the orchestrator attaches via `mlflow.start_run(run_id=...)` instead of creating its own.
- **Dispatcher returns `(job_id, job_run_id)` not `(job_id, run_id)`.** They are different things — the int job-run id versus the hex MLflow run id — and conflating them is what broke MLflow artifact retrieval in earlier passes. The launch view now shows the `job_run_id` and a "View Run" button; *Search Past Runs* discovers the MLflow run via tags.
- **Inline auto-polling result pane is gone.** Replaced with the AlphaFold-style success-message + *Search Past Runs* dialog. The dialog renders a live composite-reward chart, top-K candidates table, per-candidate Mol* viewer, eight-metric grid, and PDB download. New module-level helpers (`_is_viewable_status`, `_set_selected_enzyme_opt_run_status`, `_display_enzyme_opt_result_dialog`) keep the dialog decorator usable; the search helpers (`search_enzyme_optimization_runs_by_experiment_name`, `search_enzyme_optimization_runs_by_run_name`) live in `enzyme_optimization_tools.py`.
- **New `register_enzyme_optimization_job` notebook + DAB resource.** Without it the app SP doesn't have `CAN_MANAGE_RUN` and `WorkspaceClient().jobs.list(name=...)` returns empty from the app context — the dispatcher errored with "Orchestrator job '...' not found." Pattern matches the existing `register_*` notebooks.

### Apps-sandbox-safe UC volume writes

`_write_motif_pdb_to_volume` (the helper that persists the user's motif PDB before dispatching the job) used to do `os.makedirs(...) ; open(path, "w")`. That works in a Databricks notebook, but Databricks Apps run in a sandboxed container that does **not** have POSIX access to `/Volumes/...` — `open(...)` raises `PermissionError: [Errno 13] Permission denied: '/Volumes'`. Switched to the SDK's Files API (`WorkspaceClient().files.upload(...)`) which uploads via the workspace's UC backend with the app SP's auth and auto-creates intermediate directories. Same pattern is now the canonical "write to a volume from a Databricks App" recipe.

`grant_app_permissions.py` correspondingly grants `WRITE VOLUME` (in addition to `READ VOLUME`) on the schema to the app SP — without it the SDK upload still 403s.

### `dataframe_records` everywhere for serving endpoints

All four developability endpoints (NetSolP / PLTNUM / DeepSTABp / MHCflurry) and ProteinMPNN V8 require the `dataframe_records=[{...}]` payload shape. The legacy `inputs=...` shape gets rejected by MLflow's schema enforcement as either "extra inputs: ['index', 'columns', 'data']" (when split-orient) or as a tensor mismatch. Updates:

- `enzyme_optimization_tools.py:_query_predictor` switched to `dataframe_records` and now passes a list of named-column records explicitly.
- `protein_design.py:hit_proteinmpnn` rewritten to V8's two-column schema (`pdb` + `fixed_positions`, JSON-encoded). The orchestrator's broad `except` was masking the schema-mismatch as "ProteinMPNN optimization failed for N/N scaffold(s)."
- `motif_scaffolding.py:_hit_proteinmpnn` updated for the same V8 schema with `fixed_positions=""` (no motif preservation — that tab redesigns every residue).

### Default 6-allele HLA panel centralized in app

`enzyme_optimization_tools.py` now exports `_DEFAULT_MHC_ALLELES = "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"` (the Sette-style 6-allele panel covering ~95% of the global population), matching the default in the orchestrator's `utils.py:call_mhcflurry`. Single source of truth.

### Variant Annotation — per-run tables refactor

The variant_annotation flow used to write to global tables (`*_variants_with_pathogenic`, `*_pathogenic`, `*_annotated`) with a `run_name` discriminator column, deduped via `MERGE INTO ... WHERE run_name = ...`. Two problems: (a) Glow's VCF-INFO struct shape can drift between runs (different variants → different INFO fields), causing `MERGE INTO` schema-evolution failures, and (b) the `run_name`-only key doesn't disambiguate concurrent runs with the same name from different users.

Notebooks `01_spike_pathogenic_variants.py`, `02_filter_and_annotate.py`, and `03_save_results.py` rewritten to write to **deterministic per-run table names**: `<base>__<sanitized_run_name>_<mlflow_run_id_prefix>`. The sanitizer lowercases + collapses non-alphanumeric to `_`, caps at 40 chars; the run-id prefix is the first 8 chars of the MLflow run id. Each run owns its own three tables — Glow's struct drift can't collide across runs, concurrent runs are disjoint, and `DROP TABLE` is the cleanup story instead of `DELETE WHERE`.

App-side, `pull_annotation_results(run_id, run_name)` in `disease_biology.py` reads the new `pathogenic_table` MLflow tag/param set by `03_save_results.py` and falls back to reconstructing the name from `(run_name, run_id)` if the tag is absent (older runs).

`deploy.sh` updated; new `data/brca_pathogenic_corrected.vcf` checked in for spike-step testing.

### `deploy_model.py` — apostrophes in deployment_description

`MERGE INTO {catalog}.{schema}.model_deployments` was breaking with `PARSE_SYNTAX_ERROR` whenever a model's `deployment_description` contained an apostrophe — DeepSTABp's `"mt_mode ('Cell' or 'Lysate', default 'Cell')"` was the trigger. Added a local `_sql_escape()` helper that doubles internal single quotes; mirrors the `_sql_val` helper in `genesis_workbench/models.py`. Copy-pasted rather than imported because `deploy_model.py` runs before the `genesis_workbench` library is on the cluster's path. Numeric / identifier interpolations stay raw (trusted bundle vars or Python ints/bools).

### Single-cell processing — annotation cache UX

`processing.py:_display_results_viewer`: when loading a saved SCimilarity annotation from MLflow, real exceptions (auth, network, malformed JSON) are now stashed in `annotation_load_error_<run_id>` and surfaced as a yellow `st.warning(...)` so the user knows whether to retry or annotate fresh. The "artifact absent" case stays silent — the *Annotate Clusters* button is the right action.

Bigger fix: when overlaying annotation labels onto the UMAP, the original cluster column is no longer mutated. Mutating it fed back into a subsequent `annotate_clusters` call (which reads `df[cluster_col].unique()`) and compounded the label, e.g. `"0 - classical monocyte"` → `"0 - classical monocyte - classical monocyte"` on the second annotation pass. The annotated labels now live in a sibling column `<cluster_col> (annotated)`.

### Workflow diagrams

New rendered workflow images for the in-app `documentation/enzyme_optimization.md` page: `enzyme_optimization_workflow_fast.{mmd,png,svg}` and `enzyme_optimization_workflow_accurate.{mmd,png,svg}` (plus a generic `enzyme_optimization_workflow.{mmd,png,svg}`). The .md doc embeds the PNGs at the top of each mode's section so the help-page reader can see the topology before reading the prose.

### Files added / modified

| Path | Change |
|---|---|
| `claude_skills/SKILL_GENESIS_WORKBENCH_BATCH_WORKFLOW_PATTERN.md` *(new)* | Five-layer batch-workflow pattern with anti-patterns from real bugs and copy-paste-ready references. |
| `CLAUDE.md` | New row pointing to the batch-workflow-pattern skill. |
| `modules/core/app/utils/enzyme_optimization_tools.py` | Pre-create MLflow run from dispatcher; SDK Files API for volume writes; `dataframe_records` payload; return `(job_id, job_run_id)`; default HLA panel; `search_*_runs_by_*` helpers. |
| `modules/core/app/views/small_molecule_workflows/enzyme_optimization.py` | Replaced inline auto-polling pane with success-message + Search-Past-Runs dialog; live progress chart + top-K table + per-candidate Mol* viewer + 8-metric grid + PDB download. |
| `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/notebooks/register_enzyme_optimization_job.py` *(new)* | Registration notebook so the app SP gets `CAN_MANAGE_RUN` on the orchestrator. |
| `…/resources/register_enzyme_optimization_job.yml` *(new)* | DAB resource for the registration job. |
| `…/notebooks/01_run_optimization.py` | Reads new `mlflow_run_id` widget; attaches via `start_run(run_id=...)` when set, else creates new; emits progressive `job_status` tags. |
| `…/resources/run_optimization.yml`, `…/variables.yml`, `…/deploy.sh` | New `mlflow_run_id` job parameter; supporting bundle wiring. |
| `modules/core/notebooks/grant_app_permissions.py` | Grants `WRITE VOLUME` on the schema to the app SP (was only `READ VOLUME`). |
| `modules/core/notebooks/deploy_model.py` | New `_sql_escape()` for apostrophe-safe `MERGE INTO model_deployments`. |
| `modules/core/app/utils/disease_biology.py` | `pull_annotation_results` reads `pathogenic_table` tag, reconstructs deterministic per-run name as fallback. |
| `modules/core/app/utils/protein_design.py` | `hit_proteinmpnn` rewritten to ProteinMPNN V8's `dataframe_records` two-column schema. |
| `modules/core/app/utils/single_cell_analysis.py` | Annotation save/load tightened with explicit error path. |
| `modules/core/app/views/single_cell_workflows/processing.py` | Annotation load error surfaced; annotated cluster labels written to sibling column to prevent compound-on-rerun. |
| `modules/core/app/views/small_molecule_workflows/motif_scaffolding.py` | ProteinMPNN call updated to V8 `dataframe_records` schema. |
| `modules/core/app/views/disease_biology.py` | Variant-annotation results dialog reads from per-run table. |
| `modules/disease_biology/variant_annotation/variant_annotation_v1/notebooks/{01_spike_pathogenic_variants,02_filter_and_annotate,03_save_results}.py` | Per-run tables; `pathogenic_table` tag set by `03`. |
| `…/variant_annotation_v1/deploy.sh` | Per-run-tables bundle wiring. |
| `…/variant_annotation_v1/data/brca_pathogenic_corrected.vcf` *(new)* | Spike-step test fixture. |
| `modules/core/app/documentation/enzyme_optimization.md` | Embeds the new Fast / Accurate workflow PNGs at the top of each mode's section. |
| `modules/core/app/images/enzyme_optimization_workflow{,_fast,_accurate}.{mmd,png,svg}` *(new)* | Rendered workflow diagrams. |
| `docs/streamlit-to-react-migration-analysis.md` *(new)* | Feasibility analysis for a future React+FastAPI rewrite of the Streamlit app. Scoping document, not a planned deliverable. |
| `modules/single_cell/{rapidssinglecell,scanpy}/.../notebooks/*.py` | Minor in-line fixes supporting the per-run / cache-busting work above. |

---

## guided_enzyme_creation (2026-05-06) — Phase 1.5: Generation-mode toggle (Fast / Accurate)

### New: Fast vs Accurate generation modes for Guided Enzyme Optimization

The Guided Enzyme Optimization form gained a **Generation mode** radio (Fast / Accurate, default Fast). The toggle picks where the reward signal is applied: between iterations (Fast) or *during* the diffusion process (Accurate).

- **Fast (~30 min, no GPU cost)** — unchanged from Phase 1.4. AME runs as a Databricks Model Serving endpoint, the loop scores K candidates after generation, parents are resampled by reward for the next iteration. Job: `run_enzyme_optimization_gwb` on a CPU cluster.
- **Accurate (~30-60 min, ~$22 GPU cost)** — new. AME loads on an A10 GPU cluster (checkpoints pulled from the registered UC model `{catalog}.{schema}.proteina_complexa_ame`, version resolved from the GWB `models` table). Uses Proteina-Complexa's Feynman-Kac steering: at intermediate denoising steps, partial structures are scored via our `DevelopabilityCompositeReward(BaseRewardModel)` and trajectories are importance-sampled, so losing branches get pruned early and surviving compute lands on developability-promising candidates. Job: `run_enzyme_optimization_gwb_inprocess_ame` on an A10 GPU cluster.

Verified end-to-end on K=4, N=2, all-axes-weight-1.0 with the catalytic-triad smoke motif: Fast `iter_mean=0.449`, Accurate **`iter_mean=0.506`** — first signal that FK-steering's importance-weighted resampling concentrates reward more uniformly across the batch. iter_max is in the same range across both modes.

### Architecture

Both jobs share the same orchestrator notebook (`01_run_optimization.py`); the dispatcher (`enzyme_optimization_tools.py:_resolve_orchestrator_job_id(use_inprocess_ame)`) picks the right job by name based on the toggle. The notebook reads `use_inprocess_ame` at top, runs `install_proteinfoundation_if_needed()` + `dbutils.library.restartPython()` *before* any torch-touching imports (the Databricks runtime ships its own torch — installing torch 2.7.0 alongside leaves torch_scatter referencing the wrong C++ symbols and crashing on first use), then either calls `call_ame()` (Fast) or `run_ame_with_rewards()` (Accurate) per iteration.

`DevelopabilityCompositeReward` inherits from `proteinfoundation.rewards.base_reward.BaseRewardModel`. Its `score(pdb_path, requires_grad=False, **kwargs)` extracts the AA sequence from the candidate PDB and calls our four developability endpoints (NetSolP / PLTNUM / DeepSTABp / MHCflurry) in series, applying the half-life anchor sigmoid to PLTNUM. Per-axis fallback: any failed endpoint contributes 0 instead of crashing the search. The reward is attached via `model.reward_model = reward` directly (not via `inf_cfg.reward_model` / Hydra) — `Proteina.configure_inference()` only takes `(inf_cfg, nn_ag)`.

### AME checkpoints from UC, not NGC

The Accurate path pulls AME checkpoints from the registered UC model `{catalog}.{schema}.proteina_complexa_ame` (whichever version is `is_active=true` in the GWB `models` table). No NGC fallback — if the proteina_complexa submodule isn't deployed in this workspace, `_fetch_ame_checkpoints_from_uc(...)` raises with a clear "deploy proteina_complexa first" message. Re-deploys are bit-reproducible because the UC version is the same .ckpt blob the proteina_complexa registration baked.

### Endpoint warmup at job start (both paths)

The orchestrator pre-warms NetSolP / PLTNUM / DeepSTABp / MHCflurry (`warmup_developability_endpoints`) and AME / ESMFold / ProteinMPNN (`warmup_generation_endpoints`) with one dummy call each before the loop starts. Without it, scale-to-zero cold starts of 5-20 min mid-loop bust the request timeout. The Fast-path AME call's `_query` timeout was also raised: 600s → **1800s** (AME cold start observed at >20 min); default `_query` timeout for the developability endpoints raised 600s → **1200s**.

### Phase 1.4 motif chain bug, fixed

A latent bug in the Phase 1.4 ProteinMPNN call: orchestrator passed `fixed_positions={target_chain: motif_residues}` where `target_chain` was the input motif's chain (typically "B"), but AME emits scaffolds on chain **"A"**. Upstream's `tied_featurize` KeyError'd on the chain mismatch; the orchestrator's broad `except` swallowed the error; motif preservation silently no-op'd every iteration. Fixed by hardcoding `AME_OUTPUT_CHAIN="A"` and making the fall-through warning loud (`⚠️ MOTIF PRESERVATION DID NOT RUN for this candidate.`).

### Pinning + on-demand compute pattern alignment

- All proteinfoundation transitive deps are now exact-pinned in a versioned `enzyme_optimization_v1/notebooks/proteinfoundation_requirements.txt` (35 packages — torch==2.7.0 to align with PyG cu126 wheels, accelerate==0.34.2 for transformers 5.5.0 compat, biotite==1.4.0 because atomworks 2.2.0 hard-pins it, plus 30 more). The proteinfoundation upstream itself is pinned via `git clone --branch dev` with the resolved SHA logged at install time.
- Both enzyme_optimization jobs (Fast + Accurate) are configured for **on-demand** compute on every cloud — `availability: ON_DEMAND` / `ON_DEMAND_AZURE` / `ON_DEMAND_GCP` overlaid via per-cloud `targets:` in `databricks.yml`, matching the canonical pattern in `boltz/boltz_1/databricks.yml`. Spot-instance reclamation killed two consecutive verification runs at 13 and 35 min; the rule is now codified in `claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md` (new "On-demand compute (hard rule)" section).

### Files added / modified

| Path | Change |
|---|---|
| `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/resources/run_optimization.yml` | New `run_enzyme_optimization_inprocess_ame` job resource (A10 GPU). Both jobs gained `use_inprocess_ame`, `fk_n_branch`, `fk_beam_width`, `fk_temperature`, `fk_step_checkpoints` job params. |
| `…/databricks.yml` | Per-cloud on-demand overlays for both jobs across `prod_aws` / `prod_azure` / `prod_gcp`. |
| `…/notebooks/utils.py` | New: `DevelopabilityCompositeReward` (lazy factory, inherits from upstream `BaseRewardModel`), `load_ame_model()`, `_fetch_ame_checkpoints_from_uc`, `_resolve_ame_uc_version`, `install_proteinfoundation_if_needed()`, `_build_motif_atom_spec_from_pdb`, `warmup_developability_endpoints`, `warmup_generation_endpoints`, `run_ame_with_rewards`. AME `call_ame` timeout 600s → 1800s. Default `_query` timeout 600s → 1200s. |
| `…/notebooks/01_run_optimization.py` | Early-install cell at the top reads `use_inprocess_ame`; if true, installs proteinfoundation deps and `dbutils.library.restartPython()` before any torch-touching code. Job-start warmup of all 8 endpoints on both paths. `AME_OUTPUT_CHAIN="A"` for ProteinMPNN motif preservation. |
| `…/notebooks/proteinfoundation_requirements.txt` *(new)* | All 35 Accurate-path pip pins, exact-pinned per the GWB rule. |
| `modules/core/app/utils/enzyme_optimization_tools.py` | `_resolve_orchestrator_job_id(use_inprocess_ame)` picks job by name. New `use_inprocess_ame` kwarg on `start_enzyme_optimization_job`. |
| `modules/core/app/views/small_molecule_workflows/enzyme_optimization.py` | Generation-mode radio (Fast / Accurate, default Fast) with cost/time/mechanism help-text. |
| `claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md` | New "On-demand compute (hard rule)" section codifying the per-cloud overlay pattern. |
| `claude_skills/SKILL_GENESIS_WORKBENCH_WORKFLOWS.md` | Generation-mode toggle + Fast/Accurate behavior described under Guided Enzyme Optimization. |
| `claude_skills/SKILL_GENESIS_WORKBENCH.md` | "Inside Genesis Workbench" updated with the dual-mode toggle. |

---

## guided_enzyme_creation (2026-05-05)

### New: Guided Enzyme Optimization workflow

A new "Guided Enzyme Optimization" tab under Small Molecules — a reward-weighted resampling loop around the Motif Scaffolding stack (Proteina-Complexa-AME + ProteinMPNN + ESMFold). Each iteration scores K candidates on motif backbone RMSD, ESMFold pLDDT, optional Boltz substrate complex confidence, and four developability axes (solubility, half-life, thermostability, immunogenicity), composes a weighted composite reward, logs everything to MLflow, then biases the next iteration toward high-reward parents.

The half-life axis is **anchor-based**: the user supplies one or more reference enzymes with measured half-life; the loop sigmoids each candidate's PLTNUM relative-stability score against `min(reference) + margin`, turning a relative ranker into a defensible "predicted at least as long-lived as your reference, with margin" signal.

A **strategy interface** is shipped now so Phase 2 can drop in `EvolutionaryStrategy` (mutate top-K with ProteinMPNN's fixed-positions mode) without touching the orchestrator core. Phase 1 implements `ResampleFromAMEStrategy` (default) and `NoOpStrategy` (verification).

### New predictor submodules under `modules/small_molecule/`

- `netsolp/netsolp_v1/` — NetSolP-1.0 distilled, BSD-3-Clause. ONNX Runtime CPU endpoint. Tiny dep tree (`onnxruntime`, `fair-esm`). Weights bundled in `weights/` (extracted once via `weights/extract_weights.sh` from the upstream tarball; ~30 MB, BSD attribution preserved alongside).
- `pltnum/pltnum_v1/` — PLTNUM-ESM2, MIT. Half-life relative-stability ranker on ESM-2 650M. GPU_SMALL endpoint. Weights auto-pull from HuggingFace `sagawa/PLTNUM-ESM2-NIH3T3`. The `PLTNUM_PreTrainedModel` class is vendored inline in the registration notebook with attribution.
- `deepstabp/deepstabp_v1/` — DeepSTABp, MIT. Tm regression in °C. ProtT5-XL backbone (~3 GB, MIT verified at parent ProtTrans repo) auto-pulls from HuggingFace; the 80 MB MLP head pulls from the upstream GitHub raw URL. GPU_SMALL endpoint.
- `mhcflurry/mhcflurry_v2/` — MHCflurry 2.2.1, Apache-2.0. MHC-I peptide-presentation predictor wrapped as a sliding-9-mer scan with a default 6-allele HLA panel (~95% global coverage), aggregated to a per-residue "immunogenic burden" score. CPU endpoint. Weights via `mhcflurry-downloads fetch` at registration time.

### New orchestrator submodule

- `enzyme_optimization/enzyme_optimization_v1/` — Databricks job that runs the loop on a CPU cluster (all heavy work delegated to endpoints). 24h timeout cap. Parameters mirror `start_scanpy_job`. The job is dispatched on demand by the Streamlit page, not auto-run at deploy time.

### Boltz SDK timeout fix

`modules/core/app/utils/protein_design.py:hit_boltz` now uses a long-timeout `WorkspaceClient(config=Config(http_timeout_seconds=600))` and accepts a `timeout_seconds` kwarg (the optimization loop passes 900s for ligand complexes). Same incident pattern that bit SCimilarity earlier — the Databricks SDK's 60s default was killing Boltz cold starts on `GPU_SMALL`.

### Motif RMSD utility

`modules/core/app/utils/structure_utils.py` gained `motif_backbone_rmsd(input_pdb_str, designed_pdb_str, motif_residues, ...)` — thin Bio.PDB.Superimposer wrapper for the optimization loop's structural fidelity axis. Uses the existing `select_and_align` helpers; no new file.

### App side

- New `modules/core/app/utils/enzyme_optimization_tools.py` — `start_enzyme_optimization_job` (mirrors `start_scanpy_job` shape, dynamic job-ID lookup by name since `RUN_*_JOB_ID` env vars aren't actually wired up in this repo), `predict_enzyme_properties` (single-sequence smoke test across all four developability endpoints, used by the form's "Test on T4 lysozyme" button), `load_optimization_trajectory` / `load_top_k_pdbs` / `get_run_status` for the polling view.
- `modules/core/app/utils/streamlit_helper.py:_MODEL_ENDPOINT_MAP` updated with the four new display-name → UC-name mappings (`netsolp_v1`, `pltnum_v1`, `deepstabp_v1`, `mhcflurry_v2`).
- `modules/core/app/views/small_molecules.py` — new `enzyme_opt_tab` between Motif Scaffolding and ADMET; renders `enzyme_optimization.render()`.

### Documentation

- README per-module dependency table extended with NetSolP, PLTNUM, DeepSTABp, MHCflurry rows (every pin + license + upstream source); "Inside Genesis Workbench" line lists all four developability predictors.
- New in-app help: `modules/core/app/documentation/enzyme_optimization.md` with full inputs/outputs/pipeline + the honest caveats (half-life is anchor-relative, not hours; first iteration is uniform; etc.).
- `claude_skills/SKILL_GENESIS_WORKBENCH.md`, `SKILL_GENESIS_WORKBENCH_WORKFLOWS.md` updated with the new workflow.
- `claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md` documents the new per-submodule deploy order and the one-time NetSolP weight-population step.
- `claude_skills/SKILL_GENESIS_WORKBENCH_DESTROY_WIZARD.md` adds a note on the new `small_molecule` submodule list and on NetSolP's git-bundled weights surviving destroy.
- `claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md` gains a top-level **Dependency hygiene** section codifying the exact-pin rule applied to every new pip dep introduced in this branch.

---

## version_pinning (2026-04-20 → 2026-05-05)

### Version pinning across modules
- Exact-pinned every pip dependency in registration notebooks (`pkg==X.Y.Z`); removed `latest`/range specifiers. Includes torch, torchvision, tensorflow, transformers, scimilarity, and others.
- Removed hardcoded `aws_attributes` from submodule bundles; now driven entirely by per-target overrides in each module's `databricks.yml`.

### SCimilarity refactor — VS-backed similarity search
- Externalized the 23M-cell reference from the in-memory MLflow `SearchNearest` PyFunc into a Delta table (`scimilarity_cells`) + Databricks Vector Search index (`scimilarity_cell_index`). The deprecated `register_SearchNearest_task` is removed from the job graph.
- New notebooks: `06b_extract_reference_to_delta.py` (writes embeddings + metadata in 1M-row batches) and `06c_create_cell_vector_index.py` (creates VS endpoint + Delta Sync index, polls until ready).
- App-side `search_nearest_cells()` queries the VS index and joins back to the Delta table for `prediction`/`disease`/`tissue`/`study`. No per-request PyFunc endpoint in the loop.
- Bugfixes from end-to-end testing: `06b` per-batch `reset_index(drop=True)` (without it, only the first 1M rows had populated metadata); `06c` distinguishes `NotFound` vs transient API errors and tolerates `sync_index` 400 when a sync is already running.
- SCimilarity cache volume created out-of-band in `deploy.sh` (alphafold pattern); removed from `volumes.yml` so `bundle destroy` doesn't blow away the 12 GB Zenodo download.

### Deploy / destroy ergonomics
- New `--only-submodule <path>` flag on root and aggregator `deploy.sh` / `destroy.sh` for surgical per-submodule operations (e.g. redeploy only `scimilarity` without touching `scanpy`/`scgpt`/`rapidssinglecell`). Atomic modules reject the flag with a clear error.
- New skill files: `claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md` and `SKILL_GENESIS_WORKBENCH_DESTROY_WIZARD.md`. Destroy wizard codifies "core last" ordering and the new policy: do **not** delete VS endpoints/indexes by default — preserving them turns a 5+ hour re-sync into a fast incremental on the next deploy.
- `update.sh` now bootstraps the `genesis_workbench_secret_scope` and the two app-required secrets (`core_catalog_name`, `core_schema_name`) the same way `deploy.sh` does, so app deploys don't fail with `Invalid secret resource ...`.
- `destroy_module.py` gained an optional VS cleanup block (currently commented out, for opt-in only when a fully clean slate is wanted).

### Reliability
- `register_boltz` and `register_esmfold` raised from 1 h → 2 h on both the task `timeout_seconds` and the inner `wait_for_job_run_completion(...)` poll, so weight downloads + endpoint provisioning don't false-fail.

### UI
- Removed page-level CSS injection in `single_cell_workflows/processing.py` that bumped *all* tabs on the Single Cell page to `1.1rem / 500` — the parent tab strip now matches the other modules' default Streamlit tab styling.

---

## development (2026-04-11/13)

### Protein Studies — New workflows

- **Boltz Structure Prediction**: Added Boltz as a third structure prediction model alongside ESMFold and AlphaFold2 in the Structure Prediction tab. Supports multi-chain complexes (protein-protein, protein-ligand with SMILES input). Uses the already-deployed Boltz endpoint with proper input formatting (`protein_A:SEQUENCE`).
- **Inverse Folding with ProteinMPNN**: New standalone tab where users paste a PDB backbone and ProteinMPNN designs multiple sequences that fold into that structure. Selectbox to browse designs, auto-validates each with ESMFold and displays the predicted structure. Includes a default PDB (ubiquitin fragment) for immediate testing.

### Single Cell — New workflows

- **Cell Type Annotation via SCimilarity**: Automatic cluster annotation using the 3 SCimilarity endpoints (GeneOrder, GetEmbedding, SearchNearest) against a 23M-cell reference database. Generates cell embeddings in batches, searches nearest neighbors, majority-vote prediction per cluster. Displays annotation table + UMAP colored by predicted cell type. Includes progress bar with spinner.
- **Cell Similarity Search**: Standalone tab to search the 23M-cell reference for cells similar to a selected cluster. Shows neighbor cell type distribution (bar chart), disease distribution (bar chart), study sources, and full results table.
- **Differential Expression Viewer**: Pairwise DE between any two user-selected clusters in the results viewer. Mann-Whitney U test with Benjamini-Hochberg correction. Interactive volcano plot with labeled significant genes and sortable results table.
- **Pathway Enrichment Analysis**: Enrichr-based GO/KEGG/Reactome enrichment of cluster marker genes via gseapy. Bar chart of top enriched pathways and full results table with overlap, p-values, and gene lists.
- **Trajectory / Pseudotime Analysis**: Optional diffusion pseudotime computation added to both Scanpy and Rapids-SingleCell processing pipelines (new "Compute Pseudotime" checkbox). Results viewer shows UMAP colored by pseudotime and gene expression along pseudotime with LOWESS trendline.
- **Gene Perturbation Prediction (scGPT)**: New scGPT perturbation model for zero-shot prediction of gene knockout/overexpression effects. New registration notebook (`03_register_scgpt_perturbation.py`), deployment notebook (`04_import_perturbation_gwb.py`), and job YAML tasks. UI tab with gene selector ranked by cluster expression, perturbation type radio, bar chart of top affected genes, scatter plot (original vs predicted), and summary metrics.

### Deployment optimizations

- **SCimilarity**: Smart skip when model files already exist in Volume (saves 60+ min on re-deploy). Parallel model + sample data downloads via ThreadPoolExecutor. Quiet wget/tar output. CPU nodes for wget and GeneOrder job clusters instead of GPU (cost savings).
- **scGPT**: Skip model/data downloads when files already exist. Pre-load model weights in `load_context()` instead of re-reading 2-3 GB from disk on every `predict()` call. Smaller input_example (10x1500 instead of 1000x1000) for faster model logging. Force float32 dtype in perturbation model to avoid float16/float32 mismatches.

### UI consistency

- **Mol* viewer standardization**: Added dark theme CSS (`MOLSTAR_DARK_CSS`) to `molstar_tools.py` matching the small molecule viewer style. Standardized viewer dimensions to `width: 100%; height: 500px` (was hardcoded `1200px x 400px`). Switched to `loadStructureFromData` API. Uniform `components.html` height (540px) across all protein studies and small molecule views.

### Infrastructure

- **destroy.sh safety**: Changed `rm .deployed` to `rm -f .deployed` in all 7 module destroy scripts to avoid errors when file doesn't exist.
- **App permissions**: Automated `CAN_QUERY` grants on serving endpoints and `CAN_MANAGE_RUN` on jobs during deployment via `grant_app_permissions.py`.

---

## add_small_molecule_studies (2026-04-09/10)

### Home page overhaul
- **AI Assistant tab** ("What do you want to do?"): Natural language interface powered by Claude Sonnet via Databricks Foundation Model endpoint. Users describe what they want to accomplish and get guided to the right workflow with step-by-step instructions.
- **Documentation Search tab**: Full-text search across all workflow documentation (markdown files indexed at app startup)
- **Example pills**: Pre-built example questions for quick discovery (e.g., "How do I predict protein structure?", "Run a GWAS analysis")
- **Documentation**: Added markdown docs for all workflows — molecular docking, protein design, GWAS analysis, variant annotation, sequence search, single cell analysis, etc.
- **Bigger/bolder tabs**: Global CSS for larger tab labels (`1.1rem`, `font-weight: 600`)
- **User settings caching**: Cached in session state to avoid re-fetching on every rerun
- **LLM endpoint**: Configurable via `LLM_ENDPOINT_NAME` env var, added to `app.yml` resources with `CAN_QUERY` permission

### DiffDock — Confidence model fix
- Fixed `'HeteroDataBatch' has no attribute 'complex_t'` error by building a separate confidence dataset using `InferenceDataset` with confidence model parameters — matches the pattern in `01_register_diffdock_wo_esm.py`
- Updated example PDB in UI from 3-residue stub to 50-residue excerpt from PDB 6agt (fetched at runtime)
- Added step-by-step progress bar (ESM embeddings → DiffDock scoring → results) replacing single spinner

### Disease Biology — New module
- **GWAS Pipeline**: Parabricks `fq2bam` + `haplotypecaller` (split from single `pbrun germline`), `--low-memory` flag, real-time log streaming, input file validation, BWA index download
- **VCF Ingestion**: VCF-to-Delta via Glow, auto-generated output table names (`vcf_ingested_{timestamp}`), output table logged as MLflow tag for downstream lookup
- **Variant Annotation**: ClinVar annotation with gene panel filtering (BRCA1/BRCA2, ACMG SF v3.2 81-gene panel via BED file, or custom regions), Lakeview dashboard with auto-updated catalog/schema references (fixed `lakeview.update` SDK API to use `Dashboard` object)
- **Parabricks**: Fixed Python 3.12 pip incompatibility (`python -m ensurepip`), separate `%sh`/`%pip` cells, reference genome download switched from FTP to HTTPS, downloads to local disk then copies to Volume
- **GWAS Analysis cluster**: Changed from single-node to 4-worker multi-node for parallel processing
- All disease biology init workflows use MERGE INTO for idempotent settings inserts
- Sample VCF download fixed: FTP→HTTPS, Python-based download with validation instead of `%sh wget`
- GWAS results: Graceful handling of all-NULL pvalues with clear user message

### MLflow job status tracking
- Created `modules/core/notebooks/update_mlflow_status.py` — lightweight serverless notebook for setting `job_status` tag
- Added `mark_success` / `mark_failure` tasks (using `depends_on` with `outcome: SUCCEEDED/FAILED`) to all 5 job workflows: parabricks_alignment, gwas_analysis, vcf_ingestion, variant_annotation, alphafold
- Search results now derive status from `job_status` tag which is reliably set by completion tasks
- Added visual progress column to search results with blinking orange dots for in-progress, green for complete, red for failed

### Sequence Search — New workflow
- **Download**: UniRef90 FASTA download with skip-if-exists
- **Delta tables**: Batch FASTA parsing (500K records/batch), skip if table has >100 rows
- **Batch embedding**: Switched from GPU `pandas_udf` to `ai_query()` via deployed ESM2 serving endpoint — eliminates GPU cluster dependency. GPU notebook preserved as `03_batch_embed_sequences_gpu.py` backup. Limited to 1M sequences.
- **Vector index**: Fixed SDK API (`EndpointType.STANDARD`, `EmbeddingVectorColumn`, `endpoint_status`), enabled Change Data Feed on source table
- UI: Moved Sequence Search to second tab position (after Settings)
- Added `parasail` to app requirements for Smith-Waterman alignment

### Settings page improvements
- **Deployed Endpoints section**: Shows real-time status using served entity deployment state (🟢 Ready, 🟡 Starting, ⚪ Scaled to zero, 🔴 Failed). Cached in session state with manual refresh button.
- **Deployed Modules section**: Shows all non-job settings from settings table
- **Start All Endpoints**: Added `CAN_MANAGE` permission in `app.yml`, REST API for run status check
- **Registered Workflows**: Spinner during loading

### UI / UX improvements
- Mol* viewer: Fixed SDF loading by converting to HETATM PDB records, switched to `loadStructureFromData` API with correct 3-parameter signature
- Ligand Binder Design: Docked ligand view options only shown when valid SDF exists
- Disease Biology search: Default search text initialized to "gwas"
- VCF Ingestion → Variant Annotation auto-populate: "From VCF Ingestion" pill selector
- Protein Design: Auto-generated run name with timestamp
- All inline imports moved to page top

### Infrastructure fixes
- `databricks-sdk==0.50.0` and `databricks-sql-connector==4.0.3` pinned in sequence search notebooks
- `cloudpickle==3.0.0` pinned to match ESM2 model registration environment
- `app.yml`: Added `start_all_endpoints_job` with `CAN_MANAGE` permission, LLM endpoint with `CAN_QUERY`
- Removed `ServingModelWorkloadType` dependency (class removed in newer SDK)

---

## add_small_molecule_studies (2026-04-05)

### Start All Endpoints feature
Added ability to start all deployed model serving endpoints and keep them alive for a configurable duration (1–12 hours). Useful for demos where endpoints must not scale to zero.

- **New notebook**: `modules/core/notebooks/start_all_endpoints.py` — queries `model_deployments` for active endpoints, retrieves `input_example` from MLflow model registry, starts endpoints via REST API, and pings them every 15 minutes using parallel requests
- **New job**: `modules/core/resources/jobs/start_all_endpoints.yml` — DAB job definition (max 1 concurrent run)
- **Settings UI**: Added "Endpoint Management" tab in Settings with a duration picker and "Start All Endpoints" button. Detects if a keep-alive job is already running and shows estimated end time instead of allowing a duplicate launch
- **Wiring**: Job ID stored in `settings` table as `start_all_endpoints_job_id`, loaded into env by `workbench.initialize()`, passed through `initialize_core.yml` → `initialize_core.py`

### Small Molecule — new model sub-modules

#### DiffDock
Added [DiffDock](https://github.com/gcorso/DiffDock) molecular docking model as a new sub-module (`diffdock/diffdock_v1`). DiffDock uses diffusion generative modeling to predict 3D binding poses for protein–ligand complexes, with a score model (reverse diffusion) and a confidence model to rank predicted poses.

- **New sub-module**: `modules/small_molecule/diffdock/diffdock_v1/` — full DAB bundle with `databricks.yml`, `variables.yml`, `deploy.sh`, `destroy.sh`
- **Job resource**: `register_diffdock.yml` — dedicated GPU cluster (DBR 14.3 LTS ML GPU, A10G single node) for checkpoint download and model registration
- **Volume**: `volumes.yml` — managed UC volume for DiffDock artifact caching
- **Notebook**: `01_register_diffdock.py` — installs DiffDock + PyG extensions, clones DiffDock repo, computes ESM embeddings, runs inference to pre-download model weights, defines `DiffDockModel` PyFunc wrapper with lazy loading (to avoid 300s serving timeout), registers to UC via `mlflow.pyfunc.log_model`, imports into GWB and deploys serving endpoint
- **Key design**: Lazy model loading in `DiffDockModel.predict()` — defers heavy ESM2 and DiffDock score/confidence model loading until first prediction call to circumvent model serving endpoint startup timeouts
- **Artifacts bundled**: DiffDock repo, ESM2 weights (~2.5 GB), score model, confidence model — all packaged as MLflow artifacts to prevent re-downloading during serving
- Updated parent `modules/small_molecule/deploy.sh` and `destroy.sh` to include `diffdock/diffdock_v1` in the module loop

#### Proteina-Complexa
Added NVIDIA [Proteina-Complexa](https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa) protein binder design models as a new sub-module (`proteina_complexa/proteina_complexa_v1`). Proteina-Complexa is a generative flow-matching model that designs novel protein binders for protein targets, small-molecule ligands, and scaffolds functional motifs — all in a fully atomistic manner.

- **New sub-module**: `modules/small_molecule/proteina_complexa/proteina_complexa_v1/` — full DAB bundle with `databricks.yml`, `variables.yml`, `deploy.sh`, `destroy.sh`
- **Job resource**: `register_proteina_complexa.yml` — dedicated GPU cluster (DBR 15.4 ML GPU, A10G single node) for NGC checkpoint download and multi-model registration
- **Volume**: `volumes.yml` — managed UC volume for checkpoint caching
- **Notebook**: `01_register_proteina_complexa.py` — installs proteinfoundation + PyG extensions + transitive deps (atomworks, graphein, openfold, etc.), downloads 3 sets of checkpoints from NGC, defines base `_ProteinaComplexaBase` PyFunc class with 3 variant subclasses, registers all 3 models to UC, imports each into GWB and deploys serving endpoints
- **3 model variants registered**:
  | Model | UC Name | Use Case |
  |-------|---------|----------|
  | `proteina-complexa` | `proteina_complexa` | Protein-protein binder design |
  | `proteina-complexa-ligand` | `proteina_complexa_ligand` | Small-molecule ligand binder design |
  | `proteina-complexa-ame` | `proteina_complexa_ame` | Motif scaffolding with ligand context |
- **Code bundling**: Stages `proteinfoundation`, `openfold`, and `graphein` as `code_paths` — avoids broken build from graphein's invalid PEP 440 specifier and openfold's missing Python 3.12 support
- **NGC checkpoint patch**: Patches `concat_pair_feature_factory.py` to match NGC checkpoint dims (272→271) — NGC checkpoints were trained without `CrossSequenceChainIndexPairFeat`
- Updated parent `modules/small_molecule/deploy.sh` and `destroy.sh` to include `proteina_complexa/proteina_complexa_v1` in the module loop

---

### Protein Studies — deployment fixes

#### ESMFold
- Reverted registration job from serverless GPU back to dedicated GPU cluster (`15.4.x-gpu-ml-scala2.12`) — serverless env installed CPU-only torch causing CUDA driver mismatch on serving endpoint
- Added `aws_attributes: availability: ON_DEMAND` to prevent spot preemption during long registration jobs
- Reverted `databricks.yml` CLI version to `>=0.236.*`

#### Boltz
- Reverted all files to match known working commit (`8348954`): dedicated GPU cluster, `flash_attn==1.0.9`, `torch==2.3.1+cu121`, `mlflow==2.15.1`, `cloudpickle==2.2.1`
- `flash_attn==2.8.3` was incompatible with `boltz==0.4.0` (removed `flash_attn_unpadded_kvpacked_func` API)
- Added `aws_attributes: availability: ON_DEMAND`

---

## deploy/fe-vm-hls-amer (2026-03-18)

Deployment to `fe-vm-hls-amer` (AWS) — all modules verified working (51/51 checks passed).

---

### Infrastructure

#### ON_DEMAND enforcement
- 12 of 21 jobs use dedicated clusters (`job_clusters` with `new_cluster`). All 12 have `aws_attributes: availability: ON_DEMAND` in their YAMLs.
- The remaining 10 jobs run on **serverless compute** (no cluster definition) — `aws_attributes` does not apply to them.
- **Known DAB issue**: On the initial deploy, 6 of the 12 cluster-based jobs were created with `SPOT_WITH_FALLBACK` despite the YAML specifying `ON_DEMAND`. This appears to be a DAB bug where cluster config changes aren't propagated to existing jobs.
- Fixed via `databricks jobs reset` API for: `run_scanpy_gwb`, `run_rapidssinglecell_gwb`, `register_scgpt`, `register_scimilarity`, `register_proteinmpnn`, `register_rfdiffusion`
- **On fresh deploy**: Verify all cluster-based jobs have ON_DEMAND after `deploy.sh` completes. If DAB doesn't apply the setting, use `databricks jobs get <job_id>` to check and `databricks jobs reset` to fix.

#### Shared catalog GRANT
- `initialize_core.py`: wrapped `GRANT USE CATALOG` in try/except — user may not own the target catalog but `account users` may already have `ALL_PRIVILEGES`. **Fixed in code** — graceful fallback on grant failure.

#### Wheel deployment
- `deploy.sh` copies wheels to UC Volume after `databricks bundle deploy`. If the script fails mid-run, the wheel copy step may not execute. **Fixed in code** — wheel copy is now in the deploy flow with `set -e`.

#### Job tags
- `download_gene_references_gwb` was missing standard GWB tags on initial deploy. **Fixed in code** — both scanpy and rapids-singlecell `download_gene_references.yml` now include `tags: application: genesis_workbench, module: single_cell`.

#### application.env
- `deploy.sh` uses `paste -sd,` which breaks on comments/blank lines. **Fixed in code** — env file cleaned of all comments and blanks. Deployers should keep env files comment-free.

#### Docker credentials
- `bionemo_docker_token` and `parabricks_docker_token` are passed as plaintext DAB variables — visible in workspace job definitions via API. **Not yet fixed.** To harden: store in secret scope and reference via `{{secrets/scope/key}}`.

#### Unused DAB experiment
- `modules/core/resources/experiments/module_registration.yml` created an empty, confusing experiment at `/Users/<user>/dbx_genesis_workbench_modules`. Commented out with explanation. Deleted from workspace.

---

### AlphaFold

#### 5-layer download failure
3 of 7 download tasks failed. Five distinct issues stacked:
1. **Spot preemption** → fixed by ON_DEMAND
2. **FTP/rsync blocked** on AWS VPC → `sed` patches FTP→HTTPS
3. **Heredoc quoting** in `%sh` cells → moved script creation to Python cell
4. **Path parsing + silent wget** → replaced recursive wget with explicit URL parsing (`curl` + `grep`) fed to `aria2c -j 16`
5. **HTML "href" prefix in regex** → fixed with `cut -d'"' -f2 | tr -d '/'`

Files changed: `download_setup.py`, `download_pdb_mmcif.py`

---

### Single Cell (scanpy + rapids-singlecell)

#### Dependency pins
- `scikit-learn==1.5.*` — cuml 25.10 wraps `BaseEstimator._get_default_requests` removed in newer scikit-learn
- `numpy<2` — prevents TensorFlow import failure (`numpy.core.multiarray failed to import`) on DBR 16.4 GPU ML runtime

#### App UI improvements
- **Mode-aware MLflow defaults**: Moved mode selector outside `st.form` so experiment name updates dynamically (`scanpy_genesis_workbench` vs `rapidssinglecell_genesis_workbench`)
- **Heading**: Renamed "Run Scanpy Analysis" → "Run Analysis"
- **Results viewer filter**: Default changed from `scanpy_genesis_workbench` to `genesis_workbench` — partial match shows both scanpy and rapids results

#### Gene mapping — dataset compatibility

Both `analyze_single_h5ad.py` notebooks (scanpy + rapids-singlecell) have two paths for gene names:

**Path 1: `gene_name_column` is provided**
- Uses the specified column directly as gene names
- Uppercases for consistent QC (MT-, RPS, RPL detection)
- No Ensembl reference lookup — skips entirely
- Works for any dataset where a column already has gene symbols

**Path 2: `gene_name_column` is empty (default)**
- Requires `species` parameter
- Assumes `adata.var.index` contains Ensembl IDs (e.g., ENSG00000141510)
- Loads reference CSV from `/Volumes/{catalog}/{schema}/{scanpy|rapids}_reference/ensembl_genes_{species}.csv`
- Merges on `ensembl_gene_id` to get `external_gene_name`
- Falls back to Ensembl ID if no match

**Dataset compatibility:**

| Dataset type | gene_name_column | species | Notes |
|---|---|---|---|
| Ensembl IDs as index (e.g., 10x CellRanger) | (empty) | hsapiens/mmusculus/rnorvegicus | Standard path, uses reference mapping |
| Gene symbols as index (e.g., Adams) | `row_names` | (ignored) | After reset_index, symbols land in `row_names` |
| Gene symbols in a named column | column name (e.g., `gene_name`) | (ignored) | Direct column reference |

**Potential improvements (not yet implemented):**
1. Auto-detect: Check if var index values look like Ensembl IDs (`ENSG`/`ENSMUSG` prefix). If not, auto-switch to treating them as gene symbols.
2. Dual-column support: For datasets with both gene symbols and Ensembl IDs, allow specifying both.
3. Pre-existing `mt` column: Skip recalculation if the column already exists.

---

### BioNeMo

#### Sample finetune data
- Downloaded BLAT_ECOLX_Tenaillon2013 (beta-lactamase fitness landscape, 989 sequences, 80/20 train/eval split) from [SWAT repo](https://github.com/ziul-bio/SWAT) (MIT License)
- Uploaded to `/Volumes/{catalog}/{schema}/bionemo/esm2/ft_data/`
- Added download step to `initialize.py` so future deployments auto-provision the data
- Original data: Jacquier et al., PNAS 2013

#### Placeholder paths fixed
- Job YAML defaults updated from `/Volumes/my_catalog/schema/volume/folder/...` to `${var.core_catalog_name}/${var.core_schema_name}` variable interpolation
- Notebook widget defaults cleared of hardcoded developer paths
- Files: `bionemo_finetune_esm.yml`, `bionemo_inference_esm.yml`, `bionemo_esm_finetune.py`, `bionemo_esm_inference.py`

#### Inference fixes
- **Auto-detect output type**: Checks for `classification_output` or `regression_output` in results dict instead of trusting `task_type` parameter
- **Create result directory**: Added `os.makedirs(result_location, exist_ok=True)` before writing results
- **Task type help note**: Added help text to UI selectbox: "Must match the task type used during fine-tuning. If mismatched, the notebook will auto-detect from the model output."
- **bionemo_weights typo**: Fixed `mmt_bionemo_esm2_tinetune_test` → `mmt_bionemo_esm2_finetune_test` in table

---

### Boltz

#### PyTorch version mismatch
- `conda_env.yml` had `torch==2.3.1+cu121` but `transformers>=4.41` (from `dbboltz`) pulled in a version requiring torch>=2.4
- Logs showed: `"Disabling PyTorch because PyTorch >= 2.4 is required but found 2.3.1"`
- Fixed: Updated to `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`

#### Dependency pinning
- `transformers>=4.41,<4.50` and `sentence-transformers>=2.7,<3` pinned in both `dbboltz/pyproject.toml` and `conda_env.yml` to prevent future drift

#### Workload
- Bumped to `Medium` workload size for stability
- Model re-registered (v3) with updated deps

---

### SCimilarity

#### Per-endpoint workload configuration
Replaced shared `workload_type` job parameter with per-endpoint params:

| Endpoint | Workload Type | Workload Size | Rationale |
|---|---|---|---|
| gene_order | CPU | Small | Only `setuptools` in pip_requirements — no torch, fast CPU build |
| get_embedding | MULTIGPU_MEDIUM | Small | NN inference needs GPU; scimilarity pulls torch transitively |
| search_nearest | MULTIGPU_MEDIUM | Small | scimilarity pulls torch (CPU builds slow); needs RAM with low concurrency |

**Why GPU for get_embedding and search_nearest?** All models depend on `scimilarity==0.4.0` which transitively pulls `torch` + `pytorch-lightning`. GPU serving environments have torch pre-cached in the base image → fast container builds. CPU serving works functionally but triggers a full torch install from scratch.

**Why Small concurrency for search_nearest?** Loads ~23M cell reference into RAM (~12GB per worker). Small (0-4 workers) fits in memory. Medium (0-16) OOMs.

#### Registration flow updated
```
01_wget_scimilarity (download model + sample data)
    ├── 02_register_GeneOrder
    ├── 03_register_GetEmbedding
    ├── 04_register_SearchNearest
    │       └── 05_importNserve_model_gwb (deploy endpoints)
    └── 06a_extractNsave_DiseaseCellTypeSamples (NEW)
```
- Added `extract_sample_data_task` (06a) — extracts IPF myofibroblast samples for endpoint testing
- Runs after `wget_SCimilarity_task` in parallel with register tasks

#### Notebook rename
- `05_import_model_gwb.py` → `05_importNserve_model_gwb.py` (reflects both import + serve)
- Task key: `update_model_catalog_scimilarity_models_task` → `importNserve_scimilarity_models_task`

#### Module README
- Created comprehensive `README.md` covering endpoints, workload rationale, job flow, parameters, notebooks, data structure, and dependencies

#### 06b endpoint testing notebook updates
- Replaced `databricks_instance` / REST API calls with `WorkspaceClient()` SDK-based endpoint queries
- Removed `databricks_instance` widget and markdown references
- Fixed pip install: `numcodecs[crc32c]==0.13.1`
- Dynamic arrow offsets for UMAP visualization (no hardcoded coordinates)
- Added `start_stop_check_endpoints (all endpoints).py` helper for waking all endpoints

---

### Model Serving (all endpoints)

#### AI Gateway configuration

All GWB model serving endpoints currently have no inference tables configured. To enable AI Gateway with inference tables on new endpoints, use `AiGatewayConfig`:

```python
from databricks.sdk.service.serving import AiGatewayConfig, AiGatewayInferenceTableConfig

ai_gateway_config = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name_prefix=f"{endpoint_name}_serving",
        enabled=True,
    ),
)
```

Pass `ai_gateway=ai_gateway_config` to `w.serving_endpoints.create_and_wait()`.

For **existing endpoints**, enable via:
```python
w.serving_endpoints.put_ai_gateway(
    name=endpoint_name,
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name_prefix=f"{endpoint_name}_serving",
        enabled=True,
    ),
)
```

`models.py` has been updated with `AiGatewayConfig` / `AiGatewayInferenceTableConfig` imports and commented-out config blocks ready to enable, replacing legacy `AutoCaptureConfigInput`.

---

### MLflow Experiments

#### Two-path design
| Path | Base | Purpose | When |
|---|---|---|---|
| System-level | `/Shared/dbx_genesis_workbench_models/` | Model registration artifacts | Deploy-time (admin, one-time) |
| User-level | `/Users/<email>/<mlflow_experiment_folder>/` | User workflow results | App UI (every user run) |

The folder name (default `mlflow_experiments`) is configurable in the app's Profile page.

---

### Post-Deploy Verification

Verification scripts were used to validate the deployment — checking all jobs (ON_DEMAND), endpoints (READY), volumes, app, tables, and groups. These are workspace-specific (hardcoded job IDs, endpoint names) and kept in the local deployment logs at `docs/deployments/fe-vm-hls-amer/` (gitignored). When deploying to a new workspace, create a workspace-specific copy with updated IDs and paths.

---

### Files Changed

#### New files
- `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/README.md`
- `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/notebooks/05_importNserve_model_gwb.py`
- `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/notebooks/helper_funcs/start_stop_check_endpoints (all endpoints).py`

#### Modified files
- `modules/bionemo/notebooks/bionemo_esm_finetune.py`
- `modules/bionemo/notebooks/bionemo_esm_inference.py`
- `modules/bionemo/notebooks/initialize.py`
- `modules/bionemo/resources/bionemo_finetune_esm.yml`
- `modules/bionemo/resources/bionemo_inference_esm.yml`
- `modules/core/app/views/nvidia/bionemo_esm.py`
- `modules/core/app/views/single_cell.py`
- `modules/core/library/genesis_workbench/src/genesis_workbench/models.py`
- `modules/protein_studies/boltz/boltz_1/dbboltz/pyproject.toml`
- `modules/protein_studies/boltz/boltz_1/notebooks/conda_env.yml`
- `modules/single_cell/rapidssinglecell/rapidssinglecell_v0.0.1/notebooks/analyze_single_h5ad.py`
- `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/resources/register_scimilarity.job.yml`
- `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/notebooks/06a_extractNsave_DiseaseCellTypeSamples.py`
- `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/notebooks/06b_checkNuse_SCimilarityEndpoints.ipynb`
- `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/notebooks/helper_funcs/start_stop_check_endpoints.ipynb`

#### Deleted/renamed files
- `05_import_model_gwb.py` → renamed to `05_importNserve_model_gwb.py`
