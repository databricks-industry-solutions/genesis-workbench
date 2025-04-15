# Genesis Workbench

## Starting Architecture Diagram

```pending
```


## Benefits of Partnering with Nvidia
1. Potential willingness to share information from their end
	1. No secret sauce, but maybe things like dependency management for some of the models.
2. They offer some convenient and potentially must more performant open source options and wrappers around some of these models.
	1. build.nvidia.com/deepmind


## Approaches and Workflows

### Depends on the model.

1. Model Serving
	1. Containers - long-term, not now. Engineering and Nvidia would have to collab. Not preferred until we get to that.
	2. Init scripts - I believe these are not an option for model serving.
	3. Load context - would work for certain use cases.
	4. Load binaries to the models as artifacts
2. Batch inference
	1. Containers - not preferred unless we get Databricks engineering and Nvidia team to collab.
	2. Init scripts
	3. Load binaries to the models as artifacts
3. Fine Tuning API
	1. ???
4. Other?


## Model Types

### Inference-only

#### ESM2
- They've wrapped a version of this in Megatron.
- A couple lines of mlflow.

### Need to be fine-tuned

#### Geneformer
- High demand
- Medium difficulty

### Unclassified

#### Alphafold
- Requires a certain amount of CPU stuff to be done at the same time
- Model takes >2 min
#### RF Diffusion
- Can fiddle things a bit
- Can use model serving and make some workflows that are reasonable

## Instructions for developers

### Configuration
Make sure you update databricks.yml, resources for any jobs you need, and if you're creating a new job create a pyproject.yml following the existing structure.

The intent of the complexity of the setup of this project is to allow multiple independent projects with different dependencies and that can effectively run independently, but with a core set of libraries and a unitied UI/API for serving and calling served models as well as for running batch jobs as appropriate.

Deployment structure is intended to be as simple as possible for a variety of users.

### Deployment
#####Simplest deployment (requires web terminal)
1. Clone the repo or pull the newest changes.
1. Run the notebook 'deploy.py' in the root folder. 
1. Follow the simple instructions in the notebook.
    1. Copy the command that prints out in the final cell.
    1. Open the web terminal, paste in the command, and run it.

Troubleshooting:
1. Use the standard terminal, not tmux
1. Web terminal is not currently available in privatelink and is disabled by default in workspaces. If you cannot use these, then deploying via command-line would be the recommended approach.

##### Standard deployment 
1. Clone the repo locally, make any modifications to databricks.yml that are appropriate (most notably which projects to deploy and run), and then validate and deploy via asset bundle.

