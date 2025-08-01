# Installation Instructions

In order to install Genesis Workbench you'll clone the repo locally and then use the provided scripts to install the Workbench to a Databricks Workspace. 

The scripts will use Databricks Asset Bundles and other CLI commands to install the application. The below diagram shows the installation process
<br>

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deployment.png" alt="Generative AI in Life Sciences" width="700"/>

### **IMPORTANT NOTE:**
** $${\color{orange}Do not manually delete the resources that are created by the script, using the workspace UI.}$$ **
<br>
** $${\color{orange}Doing so, might cause build/destroy failures.Always use the provided destroy script to remove packages}$$ **

## Prerequisites

 - Python 3.11 installed

 - You'll need to have the databricks CLI installed ([docs here](https://docs.databricks.com/aws/en/dev-tools/cli/install)) and authenticate to a workspace. 
   - You should have the workspace you want to install to as the "DEFAULT" profile. Further details on authentication with the databricks CLI is [here](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
 - The deploy script reads the configuration values from an `env.env` file which you will need to manually create under each module. Details of the content is given below.
 - You need to identify a UC Catalog that the application will use. You can use an existing catalog or create a new one.
 - You need to identify a unique schema name that the application will use. If the schema does not exist it will be created by the deploy script<br> **NOTE:** The schema must be exclusively for the Genesis Workbench application.
 - You will need a SQL Warehouse to be used by the application. Follow the instructions given [here](https://docs.databricks.com/aws/en/compute/sql-warehouse/create) to create a `2X-Small` warehouse.
 - If you are planning to use NVIDIA BioNeMo modules, you need to build a docker container and push it to a container repository. The `dockerfile` is provided at `/modules/core/resources/` folder. The `build_docker.sh` file in the same directory has the commands that need to be executed to build the docker image. Please read the disclaimer in the README about usage of NVIDIA and BioNeMo.

### Env files

For the core module create env.env file with the follwowing fields in the `/module/core` folder

```
dev_user_prefix=A prefix to be used to avoid conflict while development
core_catalog_name=Catalog to be used by Genesis Workbench
core_schema_name=Schema to be used by Genesis Workbench
sql_warehouse_id=ID of the SQL Warehouse that was created
app_name=A unique name for the UI application. 
secret_scope_name=A unique name for the secret scope that will be created by the application
bionemo_docker_userid=If using BioNeMo packages, provide the userid for the container repository
bionemo_docker_token=If using BioNeMo packages, provide the token for the container repository
bionemo_docker_image=If using BioNeMo packages, provide the image uri
```

For the other modules place env.env in the module with below fields:

For example, for `protein_studies`you will create an `env.env` file in `/module/protein_studies` folder

```
core_catalog_name=Catalog to be used by Genesis Workbench
core_schema_name=Schema to be used by Genesis Workbench
sql_warehouse_id=ID of the SQL Warehouse that was created
```

## Running the installation

Once you have installed and set up the Prerequisites you should run in terminal from the root directory of the genesis workbench project:

 - ./deploy.sh core dev           

then

 - ./deploy.sh core protein_studies

and repeat for all modules you wish to install.           


