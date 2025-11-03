# Installation Instructions

### Module Deploy and Destroy process

#### Anatomy
A module is a deployable unit in Genesis Workbench. A module consist of sub-modules or models, each sub-modules having its own deployment process controlled by a `deploy.sh` and `destroy.sh`. This gives flexibility and autonomy to design sub-modules in a way it can be deployed from the module. Every sub-module can utilize [Databricks Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/) for configuring Databricks resources. 

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

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deploy_process.png" alt="Deploy Process" width="700"/>

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

Destroy gets initiated by running the `destroy.sh` script in the root folder using the syntax `./destroy.sh <module> <cloud>` . This script should be called aftaer the Prerequisites given below are completed

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/destroy_process.png" alt="Deploy Process" width="700"/>

The destroy script does the following:
- Before `core` is destroyed, checks if all modules are destroyed
- Initiate destroy of `module` by executing the `destroy.sh` of module
  - Reads `application.env`, `aws/azure.env` and `module.env` if present
  - Module destroys asset bundle that destroys the model and all related artifacts
  - Uses the job from core module to delete all endpoints, archive inference tables
- Remove module specific values from `settings` table
- Deletes the `.deployed` file in the module

In order to install Genesis Workbench you'll clone the repo locally and then use the provided scripts to install the Workbench to a Databricks Workspace. The below diagram shows the resources being deployed into the workspace.
<br>

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment.png" alt="Generative AI in Life Sciences" width="700"/>

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
a10_node_type=g5.8xlarge
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

**Step - 3:** Now we can start installing rest of the modules. It is recommended to install one module at a time and wait for all deployment jobs to coplete before deploying the next module
Example:
- To deploy module `protein_studies` module in the above workspace `./deploy.sh protein_studies azure`
- To deploy module `single_cell` module in the above workspace `./deploy.sh single_cell azure`
- To deploy module `bionemo` module in the above workspace `./deploy.sh bionemo azure`


**IMPORTANT NOTE:**
Many jobs run in the background to download, register and deploy the models. This process can take many hours to complete. 


