# Installation Instructions

### Claude Code Skills Available

This project includes [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills in the `claude_skills/` directory that can assist with deployment, troubleshooting, and development. If you use Claude Code, these skills are automatically available:

| Skill | Description |
|-------|-------------|
| `genesis-workbench` | Overview of all modules, models, and workflows |
| `genesis-workbench-installation` | Step-by-step deployment, environment configuration, and known issues |
| `genesis-workbench-deploy-wizard` | Guided interactive deployment: asks for cloud/catalog/schema/warehouse, writes env files, runs deploy.sh, auto-fixes common failures |
| `genesis-workbench-troubleshooting` | Common deployment failures, endpoint errors, and fixes |
| `genesis-workbench-workflows` | End-to-end user guide for every UI workflow tab |
| `genesis-workbench-development` | How to add new models, endpoints, and UI workflows |

To use these skills, copy the skill files to your Claude Code skills directory or reference them directly. They cover everything from initial setup to adding new biological AI models.

---

### Module Deploy and Destroy process

#### Anatomy
A module is a deployable unit in Genesis Workbench. A module consist of sub-modules or models, each sub-modules having its own deployment process controlled by a **`deploy.sh`** and **`destroy.sh`**. This gives flexibility and autonomy to design sub-modules in a way it can be deployed from the module. Every sub-module can utilize [Databricks Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/) for configuring Databricks resources. 

The primary pattern followed in Genesis Workbench is given below
- Use Databricks Asset Bundles to:
 - Create a [Unity Catalog Volume](https://docs.databricks.com/aws/en/volumes/) in the Genesis Workbench schema.
 - Create a [Job](https://docs.databricks.com/aws/en/jobs/) that runs the notebook containing the logic for [registering model(s) in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) as [PyFunc](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/creating-custom-pyfunc/part2-pyfunc-components/)
- Use `deploy.sh` script to:
 - Run the above workflow.
 - Deploy the models to [Model Serving](https://www.databricks.com/product/model-serving) using Genesis Workbench library
 - Update module specific settings in `settings` table

##### Deploy

Deploy gets initiated by running the `deploy.sh` script in the root folder using the syntax `./deploy.sh <module> <cloud>` . This script should be called after the Prerequisites given below are completed

**Architecture view** — how the deploy scripts and Asset Bundles are organized:

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deploy_process.png" alt="Deploy Architecture" width="700"/>

**Sequence view** — what happens when you run `./deploy.sh`:

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deploy_process_flow.png" alt="Deploy Sequence" width="900"/>

The deploy script does the following:
- Checks if `core` is deployed before modules
- Initiate deploy of `module` by executing the `deploy.sh` of module
  - Reads `application.env`, `aws/azure.env` and `module.env` if present
  - Module deploys asset bundle that created the workflow that loads and registers the model and all related artifacts. This is a background process that might take many hours to complete.
  - Registers all workflows to the `core` module and grant Databricks App access to workflows and endpoints, if necessary
  - Deploys model endpoints if ncessary using common `core` module workflow  
  - `core` module deploys the UI application as well
- Add module specific values to `settings` table
- Creates a `.deployed` file in the module indicating deployment is complete. This file acts as a lock for accidental destroys 

##### Destroy

Destroy gets initiated by running the `destroy.sh` script in the root folder using the syntax `./destroy.sh <module> <cloud>` . This script should be called after the Prerequisites given below are completed

**Architecture view** — how the destroy scripts cascade through modules and sub-modules:

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/destroy_process.png" alt="Destroy Architecture" width="700"/>

**Sequence view** — what happens when you run `./destroy.sh`:

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/destroy_process_flow.png" alt="Destroy Sequence" width="900"/>

The destroy script does the following:
- Before `core` is destroyed, checks if all modules are destroyed 
  - If not, it alerts the user to which module are still deployed and why the `core` cannot be destroyed: 
    - `🚫 Deployment exist in modules/<name_of_module> Cannot remove core module`
  - Destroy `module` by executing the `destroy.sh` of module e.g. 
    - `./destroy.sh <module_name> <cloud_name>`
- When `core` is being destroyed, `destroy.sh` triggers the following processes: 
  - Reads `application.env`, `aws/azure.env` and `module.env` if present
  - Module destroys asset bundle that destroys the model and all related artifacts
  - Uses the job from core module to delete all endpoints, archive inference tables
  - Remove module specific values from `settings` table
  - Deletes the `.deployed` file in the module

In order to install Genesis Workbench you'll clone the repo locally and then use the provided scripts to install the Workbench to a Databricks Workspace.

### What each module deploys

The diagrams below show the sub-modules, Databricks workflows, and model-serving endpoints that each module creates when deployed.

#### Core

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment_core.png" alt="Core module deployment" width="900"/>

Deploying `core` stands up **two** Databricks Apps: the main UI (`genesis-workbench`) and the **MCP server** (`mcp-genesis-workbench`). Both run as their own service principals; the deploy grants both access to the catalog, endpoints, jobs, volumes, and models. See [MCP Server](#mcp-server) below.

#### Single Cell

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment_single_cell.png" alt="Single Cell module deployment" width="900"/>

**Sub-modules:** `scanpy`, `rapidssinglecell`, `scgpt`, `scimilarity`, `teddy`

#### Large Molecule

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment_large_molecule.png" alt="Large Molecule module deployment" width="900"/>

**Sub-modules:** `alphafold`, `esmfold`, `boltz`, `esm2_embeddings`, `protein_mpnn`, `rfdiffusion`, `sequence_search`, `enzyme_optimization`

#### Small Molecule

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment_small_molecule.png" alt="Small Molecule module deployment" width="900"/>

**Sub-modules:** `genmol`, `kermt`, `diffdock`, `chemprop`, `proteina_complexa`, `netsolp`, `pltnum`, `deepstabp`, `mhcflurry`

#### Genomics

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment_genomics.png" alt="Genomics module deployment" width="900"/>

**Sub-modules:** `parabricks`, `vcf_ingestion`, `variant_annotation`, `gwas`

#### BioNeMo

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment_bionemo.png" alt="BioNeMo module deployment" width="900"/>

**Sub-modules:** ESM-2 fine-tuning + inference (container-based)

A single sub-module can be (re)deployed on its own with `--only-submodule`, e.g. `./deploy.sh small_molecule <cloud> --only-submodule genmol/genmol_v1`.

### **IMPORTANT NOTE:**
** ⚠️ Do not manually delete the resources that are created by the script, using the workspace UI.**
<br>
**Doing so, might cause build/destroy failures.Always use the provided destroy script to remove packages**

## Prerequisites

 - Installation should be done by a **Workspace Admin**
 - Python 3.11 installed. Its recommended to use a conda or venv specific for this application. 
 - Identify the workspace where you want to install the application.
 - Identify the cloud where the databricks workspace is deployed. Current deployment process supports `aws` and `azure`

 - You'll need to have the databricks CLI installed ([docs here](https://docs.databricks.com/aws/en/dev-tools/cli/install)) and authenticate to a workspace. 
   - You should have the workspace you want to install to as the **"DEFAULT"** profile. Further details on authentication with the databricks CLI is [here](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
 - The deploy script reads the configuration values from a set of `.env` files which you will need to manually create under each module. Details of the content is given below.
 - You need to identify a UC Catalog that the application will use. You can use an existing catalog or create a new one.
 - You need to identify a unique schema name that the application will use. If the schema does not exist it will be created by the deploy script<br> **NOTE:** The schema **must be exclusively for the Genesis Workbench application**.
 - You will need a [SQL Warehouse](https://docs.databricks.com/aws/en/compute/sql-warehouse/) to be used by the application. Follow the instructions given [here](https://docs.databricks.com/aws/en/compute/sql-warehouse/create) to create a `2X-Small` warehouse.
 - **NVIDIA BioNeMo Container Build**: If you are planning to use NVIDIA BioNeMo modules, you need to build a docker container and push it to a container repository. The `dockerfile` is provided at `/modules/bionemo/docker/` folder. The `build_docker.sh` file in the same directory has the commands that need to be executed to build the docker image. Please read the disclaimer in the README about usage of NVIDIA and BioNeMo. 

### Env files

Configuration for the application is done using `.env` files. There are four types of env files being used
- `application.env` : Application level settings
- `module.env` : Module specific settings
- `aws.env` : AWS Cloud specific settings
- `azure.env` : Azure Cloud specific settings

##### Application configurations
Create `application.env` file with the following fields in the root folder

```
workspace_url=Workspace URL to which the application need to be deployed
core_catalog_name=Catalog to be used by Genesis Workbench
core_schema_name=Schema to be used by Genesis Workbench
sql_warehouse_id=ID of the SQL Warehouse that was created
```

##### Module configurations
**For the `core` module, create `module.env` file with the following fields in the `module/core` folder**
```
dev_user_prefix=Prefix to be applied for resources during development
app_name=Name for the Databricks App
secret_scope_name=A unique secret scope name that will be used by application. Application will create the scope
```

**For the `bionemo` module, create `module.env` file with the following fields in the `module/bionemo` folder**
```
bionemo_docker_userid=User ID for the bionemo image repo
bionemo_docker_token=Token for the bionemo image repo
bionemo_docker_image=Image tag for the bionemo image
```
For the other modules currently no additional configurations required

##### Cloud configurations
Cloud configuration is used to specify what instance type to use and endpoint configurations. **A default configuration is supplied, you can change it if you want to change the default behavior**

`aws.env`
```
cpu_node_type=i3.4xlarge
t4_node_type=g4dn.4xlarge
a10_node_type=g5.16xlarge
gpu_small_setting=GPU_SMALL
gpu_medium_setting=GPU_MEDIUM
gpu_large_setting=MULTIGPU_MEDIUM <<-Since there is no GPU_LARGE in AWS
```

`azure.env`
```
cpu_node_type=Standard_F8
t4_node_type=Standard_NC4as_T4_v3
a10_node_type=Standard_NV36ads_A10_v5
gpu_small_setting=GPU_SMALL
gpu_medium_setting=GPU_LARGE <<-Since there is no GPU_MEDIUM in Azure
gpu_large_setting=GPU_LARGE
```

Additionally the single_cell module contains two files:
 - module_aws.env.tmp
 - module_azure.env.tmp

You can remove the .tmp from these two files to use the standard compute settings for these modules. Note these two env files in single_cell contain only default compute settings and no secrets. 

## Running the installation
**Step - 1:**
Setup the prerequisites mentioned above.
Clone the [Genesis Workbench repo](https://github.com/databricks-industry-solutions/genesis-workbench) to a local folder. The deploy script requires bash.

**Step - 2:**
Once you have installed and set up the Prerequisites, `core` module need to be installed first. 
Any module can be installed by using the provided `deploy` script. The syntax to use the deploy script is `./deploy.sh <module> <cloud>`
An example deploy command for `core` module in a workspace in Azure is:
 - `./deploy.sh core azure`

**Step - 3:** Now we can start installing rest of the modules. It is recommended to install one module at a time and wait for all deployment jobs to complete before deploying the next module.
Example:
- `./deploy.sh large_molecule azure`
- `./deploy.sh single_cell azure`
- `./deploy.sh small_molecule azure`
- `./deploy.sh genomics azure`
- `./deploy.sh bionemo azure` (optional — requires BioNeMo container build)


**IMPORTANT NOTE:**
Many jobs run in the background to download, register and deploy the models. This process can take many hours to complete.

### Redeploying the UI after the initial install

For UI-only changes (frontend, FastAPI backend, app config), use `update.sh` from inside `modules/core/`. It rebuilds the React frontend, wheel, and bundle, then redeploys the app without touching the `settings`, `model_deployments`, `models`, or `batch_models` Delta tables — so existing model registrations and configuration are preserved.

```
cd modules/core
./update.sh <cloud>              # full redeploy (wheel rebuild + bundle deploy + grants + UC volume copy)
./update.sh <cloud> --ui-only    # fastest path: skips secret refresh / grants / UC volume copy
```

**Never run `./deploy.sh core <cloud>` on a populated install** — its `initialize_core_job` drops and recreates the settings/models tables and you will lose all configuration. 

### MCP Server

Alongside the UI, the `core` deploy stands up a **Model Context Protocol (MCP) server** as a second Databricks App, `mcp-genesis-workbench`. It exposes every deployed model endpoint and prebuilt workflow as MCP tools (FastMCP, streamable HTTP) so MCP clients — the Databricks AI Playground, Claude, Cursor, or your own agents — can discover and call them.

**Deployment** — the MCP app is deployed automatically by both `./deploy.sh core <cloud>` and a full `./update.sh <cloud>` (it is **skipped** by `./update.sh <cloud> --ui-only`). The deploy stages the `genesis_workbench` wheel + app code into `modules/core/mcp_app/`, deploys the `mcp_genesis_workbench_app` bundle resource, and runs `grant_app_permissions_job` for **both** app service principals (UI + MCP) — granting CAN_QUERY on serving endpoints, CAN_MANAGE_RUN on the orchestrator/workflow jobs, and the necessary volume/model/catalog access. No extra commands are required; to (re)deploy just the MCP app changes, run a full `./update.sh <cloud>`.

**Connecting** — the server is reachable at `https://<mcp-genesis-workbench-app-url>/mcp` (find the app URL with `databricks apps get mcp-genesis-workbench`). In the Databricks AI Playground the `mcp-` server is auto-discovered; external clients register the `/mcp` URL over OAuth.

**Tools** — call `list_capabilities` to enumerate what's available, then `endpoint_<name>` (synchronous, returns predictions) or `workflow_<name>` (dispatches a Job, returns a run id — poll `get_workflow_run_status`). All calls run as the MCP app's service principal. See the [MCP Server documentation](modules/core/app/backend/documentation/mcp_server.md) for details.

**⚠️ Access control** — the MCP server runs every capability under the app service principal with **no per-user authorization**, so anyone who can open the app can invoke any capability the app SP is entitled to. The app's accessor list is the control: it's pinned in `resources/mcp_app.yml` to the deployer (`CAN_MANAGE`), admins (always), and the `mcp_app_access_group` group (`CAN_USE`, default `admins`). **Scope it to the group entitled to run GWB workflows/endpoints** — deploy with `--var mcp_app_access_group=<your-group>` — and do not share the app with "all users".


