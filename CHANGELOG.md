# Genesis Workbench — Changelog

## e2fe_gwb_deploy_fixes (2026-04-29/30)

### Bug fixes — post-version_pinning regressions

- **bionemo finetune notebook**: Dropped `psutil==7.2.2` and `pynvml==11.0.0` pins from the `version_pinning` branch. Both versions conflict with what the NVIDIA bionemo Docker base image already ships (`psutil 7.0.0`, `pynvml 12.0.0`). Plain pip resolver hung 81 min retrying the proxy; `--no-index` fast-failed `ResolutionImpossible`. Main's unpinned `psutil pynvml` is the working state.
- **Boltz served-model conda_env.yml**: Bumped torch to `2.4.1+cu121` (transformers>=4.41 in dbboltz/pyproject.toml needs torch>=2.4 — without this, the served endpoint silently disabled PyTorch at startup and every inference request hung). Added `transformers<4.50` + `sentence-transformers<3` upper bounds + `setuptools<82` pin. Comment block restricted to ASCII (PyYAML reader chokes on UTF-8 multi-byte chars at MLflow register-time).
- **Boltz dbboltz/pyproject.toml**: Matching `transformers<4.50` + `sentence-transformers<3` upper bounds.
- **Boltz register notebook**: `workload_type=GPU_LARGE` + `workload_size=Large` so diffusion completes within Databricks Model Serving sync timeout.
- **App `hit_boltz` timeout**: Dedicated `_boltz_workspace_client = WorkspaceClient(config=Config(http_timeout_seconds=1800))` in `modules/core/app/utils/protein_design.py`. Databricks SDK default ~5-min timeout was tripping Boltz inference mid-prediction. Per-call timeout isn't exposed on `serving_endpoints.query()`, so a separate client with extended config is the cleanest path. Other endpoints unchanged.
- **`models.py:130` defensive default**: Switched from `user_settings['mlflow_experiment_folder']` to `user_settings.get('mlflow_experiment_folder', 'mlflow_experiments')`. Users with no settings row now get the documented default instead of an opaque `KeyError`.

### Validation

End-to-end on e2-demo-field-eng (mmt_gwb catalog):
- bionemo finetune: 2026-04-29 04:57 UTC (run 720059965576396, ESM2-650M, val_loss 4.2M → 38K, 5 checkpoints saved)
- Boltz: 2026-04-30 — direct endpoint returned multi-chain prediction (protein 215aa + RNA 19nt) AND GWB app UI rendering confirmed
- ESMFold v5: 2026-04-30 — direct endpoint returned PDB structure in 5s + app UI rendering confirmed (serving with main's `transformers==4.41.2` deps)

### Open issues — observed but not fixed in this cycle

- **SCimilarity NaN in embeddings (Stage 2 Cell Type Annotation)**: Preprocessing format mismatch — Rapids/scanpy scaling produces values with negatives, `log1p` of negatives in `modules/core/app/utils/scimilarity_tools.py:54` produces NaN. Fix is `adata.raw` preservation in scanpy/rapids `analyze_single_h5ad.py` notebooks + read `.raw.X` in scimilarity_tools.py. Deferred to a focused session — needs careful data-flow trace.
- **`a@b.com` placeholder defaults across job YAMLs** (bionemo, parabricks, gwas at minimum): Latent CLI-trigger gotcha. Only affects `databricks jobs run-now` from CLI without overriding the `user_email` param; the GWB app UI injects the logged-in user's email so it sidesteps. Deferred to a sweep PR.
- **Warehouse `SECRET_CREATION_FAILURE` / `SERVICE_FAULT`**: Hit on BOTH e2-demo-field-eng AND fe-vm-hls-amer (the backup-app workspace) **during the demo** — not workspace-specific, was a Databricks-wide serverless platform issue. fe-vm-hls-amer recovered faster than e2fe. Bundle-managing the warehouse spec (Pro Classic + `auto_stop_mins=60`) across both workspaces would prevent drift back to the vulnerable serverless 10-min-auto-stop default. Deferred.

### Workspace differential to investigate

- **ESMFold v5 deployment behavior**: `transformers==4.57.3 + torch>2.0` (the 3962ee5 version) deploys cleanly on fe-vm-hls-amer Model Serving and serves there. Same dep set on e2-demo-field-eng causes the build container to install old versions despite the spec, model fails to load. Workspace-level package-repo or build-container differential — root cause not investigated. e2fe currently serves with the older `transformers==4.41.2` deps which match its package-repo.

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
