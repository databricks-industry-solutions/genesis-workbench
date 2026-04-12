---
name: genesis-workbench-installation
description: Step-by-step installation, deployment, configuration, and troubleshooting of Genesis Workbench on Databricks (AWS, Azure, GCP) including module deployment, environment setup, and known issues.
---

# Genesis Workbench Installation Skill

Install, deploy, configure, and troubleshoot [Genesis Workbench](https://github.com/databricks-industry-solutions/genesis-workbench) on Databricks workspaces across AWS, Azure, and GCP.

## Architecture

Genesis Workbench deploys through **modules**, each containing sub-modules with independent `deploy.sh` and `destroy.sh` scripts. Modules use [Databricks Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/) to provision resources.

**Deployment pattern per module:**
1. Asset Bundles create a Unity Catalog Volume and a Job
2. The Job runs notebooks that register models as PyFunc in Unity Catalog via MLflow
3. `deploy.sh` triggers the workflow, deploys Model Serving endpoints, and updates the `settings` table
4. A `.deployed` lock file is created to prevent accidental removal

**Available modules:**
- `core` — UI app (Streamlit-based Databricks App), shared library, initialization workflows (MUST be deployed first)
- `protein_studies` — ESMFold, AlphaFold2, ProteinMPNN, RFDiffusion, Boltz-1
- `single_cell` — scGPT, SCimilarity, scanpy, rapids-singlecell
- `bionemo` — NVIDIA BioNeMo container-based workflows
- `parabricks` — NVIDIA Parabricks genomics pipelines
- `small_molecule` — Small molecule drug discovery (QSAR)

## Prerequisites

1. **Workspace Admin** privileges on the target Databricks workspace
2. **Python 3.11** installed (conda or venv recommended)
3. **Databricks CLI** installed and authenticated as the **DEFAULT** profile
   - Install: https://docs.databricks.com/aws/en/dev-tools/cli/install
   - Auth: https://docs.databricks.com/aws/en/dev-tools/cli/authentication
4. **Target workspace** identified (AWS, Azure, or GCP)
5. **Unity Catalog** — identify or create a catalog for the application
6. **Exclusive schema** — a schema name used only by Genesis Workbench (created automatically if it doesn't exist)
7. **SQL Warehouse** — provision a `2X-Small` warehouse
8. **BioNeMo only**: Build Docker container from `/modules/bionemo/docker/` and push to a container repo. See `build_docker.sh` in that directory.

## Environment Configuration

### application.env (root directory)

```
workspace_url=<Workspace URL>
core_catalog_name=<UC Catalog name>
core_schema_name=<Exclusive schema name>
sql_warehouse_id=<SQL Warehouse ID>
```

**IMPORTANT:** Keep env files free of comments and blank lines — `deploy.sh` uses `paste -sd,` which breaks on them.

### module.env — Core Module (`modules/core/module.env`)

```
dev_user_prefix=<Prefix for dev resources>
app_name=<Databricks App name>
secret_scope_name=<Unique secret scope name — app will create it>
```

### module.env — BioNeMo Module (`modules/bionemo/module.env`)

```
bionemo_docker_userid=<Docker repo user ID>
bionemo_docker_token=<Docker repo token>
bionemo_docker_image=<Docker image tag>
```

**Security note:** `bionemo_docker_token` (and `parabricks_docker_token`) are passed as plaintext DAB variables — visible in job definitions via API. For hardening, store in a secret scope and reference via `{{secrets/scope/key}}`.

### Cloud-Specific Configuration

**aws.env:**
```
cpu_node_type=i3.4xlarge
t4_node_type=g4dn.4xlarge
a10_node_type=g5.16xlarge
gpu_small_setting=GPU_SMALL
gpu_medium_setting=GPU_MEDIUM
gpu_large_setting=MULTIGPU_MEDIUM
```

**azure.env:**
```
cpu_node_type=Standard_F8
t4_node_type=Standard_NC4as_T4_v3
a10_node_type=Standard_NV36ads_A10_v5
gpu_small_setting=GPU_SMALL
gpu_medium_setting=GPU_LARGE
gpu_large_setting=GPU_LARGE
```

**gcp.env:**
```
cpu_node_type=c3d-highmem-8-lssd
t4_node_type=g2-standard-32
a10_node_type=g2-standard-32
gpu_small_setting=GPU_MEDIUM
gpu_medium_setting=GPU_MEDIUM
gpu_large_setting=GPU_MEDIUM
```

**Note (single_cell):** The single_cell module has `module_aws.env.tmp` and `module_azure.env.tmp` — remove the `.tmp` suffix to override default compute settings for those modules.

## Installation Steps

### Step 1: Clone and Configure

```bash
git clone https://github.com/databricks-industry-solutions/genesis-workbench.git
cd genesis-workbench
```

Create/edit `application.env`, cloud-specific env file, and module env files as described above.

### Step 2: Deploy Core Module (MUST be first)

```bash
./deploy.sh core <cloud>
```

Where `<cloud>` is `aws`, `azure`, or `gcp`.

### Step 3: Deploy Additional Modules (one at a time)

Deploy modules sequentially. Wait for all background jobs to complete before deploying the next module. Many jobs download, register, and deploy models — **this can take many hours**.

```bash
./deploy.sh protein_studies <cloud>
./deploy.sh single_cell <cloud>
./deploy.sh bionemo <cloud>
./deploy.sh parabricks <cloud>
./deploy.sh small_molecule <cloud>
```

### Step 4: Post-Deploy Verification

After deployment, verify:
- All jobs have correct availability (ON_DEMAND, not SPOT) — see Known Issues below
- All Model Serving endpoints reach READY state
- UC Volumes are created in the designated schema
- The Databricks App is accessible
- The `settings` table is populated with module entries

## Destroy / Teardown

```bash
# Destroy individual modules first
./destroy.sh <module> <cloud>

# Destroy core LAST (after all other modules are destroyed)
./destroy.sh core <cloud>
```

The destroy script:
- Blocks `core` destruction if any modules are still deployed
- Removes asset bundles, endpoints, and artifacts
- Archives inference tables
- Cleans up the `settings` table
- Deletes `.deployed` lock files

For residual cleanup: `./cleanup.sh`

**WARNING:** Do NOT manually delete resources via the workspace UI. This will cause build/destroy failures. Always use the provided destroy script.

## Known Issues & Fixes (from CHANGELOG)

### ON_DEMAND Enforcement (AWS)
DAB bug: On initial deploy, cluster-based jobs may be created with `SPOT_WITH_FALLBACK` despite YAML specifying `ON_DEMAND`. Fix with:
```bash
databricks jobs get <job_id>  # Check availability
databricks jobs reset --json '<updated_job_spec>'  # Fix if needed
```
Affected jobs: `run_scanpy_gwb`, `run_rapidssinglecell_gwb`, `register_scgpt`, `register_scimilarity`, `register_proteinmpnn`, `register_rfdiffusion`

### AlphaFold Download Failures
Five stacked issues on AWS: spot preemption, FTP/rsync blocked on VPC, heredoc quoting in `%sh` cells, path parsing, and HTML href regex. Fixed in code — `download_setup.py` and `download_pdb_mmcif.py` now use `aria2c` with explicit URL parsing.

### Boltz PyTorch Version Mismatch
`conda_env.yml` had `torch==2.3.1+cu121` but `transformers>=4.41` required torch>=2.4. Fixed: updated to `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`. Pins: `transformers>=4.41,<4.50`, `sentence-transformers>=2.7,<3`.

### Single Cell Dependency Pins
- `scikit-learn==1.5.*` — cuml 25.10 wraps a method removed in newer scikit-learn
- `numpy<2` — prevents TensorFlow import failure on DBR 16.4 GPU ML runtime

### SCimilarity Endpoint Sizing
| Endpoint | Workload Type | Workload Size | Rationale |
|---|---|---|---|
| gene_order | CPU | Small | No torch needed |
| get_embedding | MULTIGPU_MEDIUM | Small | NN inference needs GPU |
| search_nearest | MULTIGPU_MEDIUM | Small | ~23M cell ref in RAM (~12GB/worker); Medium OOMs |

### Shared Catalog GRANT
`initialize_core.py` wraps `GRANT USE CATALOG` in try/except — user may not own the catalog but `account users` may already have `ALL_PRIVILEGES`.

### MLflow Experiment Paths
Two-path design:
- **System-level**: `/Shared/dbx_genesis_workbench_models/` — model registration (deploy-time, admin)
- **User-level**: `/Users/<email>/<mlflow_experiment_folder>/` — user workflow results (configurable in App Profile page)

### AI Gateway / Inference Tables
Endpoints deploy without inference tables by default. To enable, use `AiGatewayConfig` with `AiGatewayInferenceTableConfig` — see `models.py` for commented-out config blocks ready to enable.

## Instructions

1. Before any installation, authenticate using the `databricks-authentication` skill if using vibe.
2. Verify all prerequisites are met — especially Workspace Admin, Python 3.11, Databricks CLI DEFAULT profile.
3. Always deploy `core` first. The script enforces this (`modules/core/.deployed` must exist).
4. Deploy one module at a time. Background jobs can take hours — check job status before deploying the next.
5. After deployment, verify ON_DEMAND settings on cluster-based jobs (especially on AWS).
6. Never manually delete deployed resources through the UI — always use `./destroy.sh`.
7. When destroying, remove all non-core modules before destroying `core`.
8. For GCP deployments, use `gcp` as the cloud parameter and ensure `gcp.env` has correct instance types.
9. Keep all `.env` files free of comments and blank lines.
10. For BioNeMo, build and push the Docker container before deploying the module.
