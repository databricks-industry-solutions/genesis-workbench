# Installation Instructions

In order to install Genesis Workbench you'll clone the repo locally and then use Databricks Asset Bundles to install the Workbench to a Databricks Workspace. 

## Prerequisites

 - You'll need to have the databricks CLI installed ([docs here](https://docs.databricks.com/aws/en/dev-tools/cli/install)) and authenticate to a workspace. 
   - You should have the workspace you want to install to as the "DEFAULT" profile. Further details on authentication with the databricks CLI is [here](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
 - You need poetry installed locally (or in the environment you're using for install)
   - For instance on macOS you can use "brew install poetry"
 - You will need to manually include "env.env" files in each module in modules - details of what should be in those is below.
 - You must create the catalog (the one you'll specify in the env files) before running the deploy script

### Env files

For the core module create env.env file with the follwowing fields

```
dev_user_prefix=
core_catalog_name=
core_schema_name=
sql_warehouse_id=
app_name=
secret_scope_name=
bionemo_docker_userid=
bionemo_docker_token=
bionemo_docker_image=
```

For the other modules place env.env in the module with fields:

```
core_catalog_name=
core_schema_name=
sql_warehouse_id=
```

## Running the installation

Once you have installed and set up the Prerequisites you should run in terminal from the root directory of the genesis workbench project:

 - ./deploy.sh core dev           

then

 - ./deploy.sh core protein_studies

and repeat for all modules you wish to install.           


