# Genesis Workbench — Changelog

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
