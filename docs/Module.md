# Module Deploy and Destroy process

#### Anatomy
A module is a deployable unit in Genesis Workbench. A module consist of sub-modules or models, each sub-modules having its own deployment process controlled by a `deploy.sh` and `destroy.sh`. This gives flexibility and autonomy to design sub-modules in a way it can be deployed from the module. The deployment process is shown below:

##### Deploy

<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/deploy_process.png" alt="Deploy Process" width="700"/>

- Deploy starts from the root `deploy.sh` script
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
<img src="https://github.com/databricks-industry-solutions/genesis-workbench/blob/main/docs/images/destroy_process.png" alt="Deploy Process" width="700"/>
- Destroy starts from the root `destroy.sh` script
- Before `core` is destroyed, checks if all modules are destroyed
- Initiate destroy of `module` by executing the `destroy.sh` of module
  - Reads `application.env`, `aws/azure.env` and `module.env` if present
  - Module destroys asset bundle that destroys the model and all related artifacts
  - Uses the job from core module to delete all endpoints, archive inference tables
- Remove module specific values from `settings` table
- Deletes the `.deployed` file in the module

