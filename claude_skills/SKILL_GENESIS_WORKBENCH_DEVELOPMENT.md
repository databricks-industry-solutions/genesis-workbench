---
name: genesis-workbench-development
description: How to add new models, workflows, and UI tabs to Genesis Workbench — registration patterns, deployment pipelines, job YAML structure, endpoint wiring, and UI integration.
---

# Genesis Workbench Development Skill

Add new biological AI models, serving endpoints, and UI workflows to Genesis Workbench following established patterns.

## Dependency hygiene (hard rule)

Every pip dependency introduced anywhere in this repo — registration notebook
`%pip install` lines, DAB `environments.spec.dependencies`, PyFunc
`pip_requirements` / `conda_env`, orchestrator job notebooks — **must use
exact version pins** (`pkg==X.Y.Z`). No `latest`, no `>=`, no unpinned bare
package names. This includes transitively-important deps the upstream
might leave loose (torch, transformers, numpy, pandas, scikit-learn, mlflow,
cloudpickle, biopython).

**Reason:** unpinned installs broke prior deploys silently — the same
upstream package would resolve to different versions across deploys depending
on PyPI release timing, and PyFunc artifacts logged with one resolved version
would fail to load on a serving endpoint that resolved a different one. The
repo lives on the `version_pinning` branch precisely as a response to that
incident.

**Pattern to mirror:** `modules/single_cell/scimilarity/scimilarity_v0.4.0_weights_v1.1/notebooks/utils.py:11-15`
— every dep pinned including `numpy==1.26.4`, `pandas==1.5.3`, `mlflow==2.22.0`,
`numcodecs[crc32c]==0.13.1`, `cloudpickle==2.0.0`. The four new predictor
submodules under `modules/small_molecule/` (`netsolp_v1`, `pltnum_v1`,
`deepstabp_v1`, `mhcflurry_v2`) and the orchestrator (`enzyme_optimization_v1`)
follow the same exact-pin convention; copy from any of them when adding the
next predictor.

When the README's per-module dependency table is updated, every new pip dep
gets its own row with the exact pin and license verified at the upstream
source (not from package summary). License-disqualified deps (academic-only,
CC-BY-NC, "research-only", `git+...` non-pinnable installs) are blockers,
not workaround-able.

---

## On-demand compute (hard rule)

Every workflow job in this repo runs on **on-demand** compute on every cloud,
not spot. Spot instances get reclaimed mid-run when the cloud provider takes
the capacity back, and GWB workflow jobs are typically minutes-to-hours long
(predictor registration, optimization loops, batch scoring) — the spot
reclamation rate exceeds the run completion rate often enough that spot is
not viable for production workflows.

**Pattern:** the cluster spec in `resources/<job>.yml` declares
`node_type_id`, `spark_version`, etc. but does NOT declare `availability`.
The per-cloud `targets:` block in `databricks.yml` overlays the right
`<cloud>_attributes.availability` per environment:

```yaml
# databricks.yml
targets:
  prod_aws:
    resources:
      jobs:
        my_workflow:
          job_clusters:
            - job_cluster_key: my_cluster
              new_cluster:
                aws_attributes:
                  availability: ON_DEMAND

  prod_azure:
    resources:
      jobs:
        my_workflow:
          job_clusters:
            - job_cluster_key: my_cluster
              new_cluster:
                azure_attributes:
                  availability: ON_DEMAND_AZURE

  prod_gcp:
    resources:
      jobs:
        my_workflow:
          job_clusters:
            - job_cluster_key: my_cluster
              new_cluster:
                gcp_attributes:
                  availability: ON_DEMAND_GCP
```

**Reason:** A10 spot instances were reclaimed mid-run twice consecutively
during the Phase 1.5 enzyme-optimization Accurate-path verification (~13 min
and ~35 min into the runs). Each Accurate run is ~1-6 hours wall-clock,
much longer than typical spot reclamation intervals. The ~30% cost premium
of on-demand is paid only for the duration of the run and is far cheaper
than re-running an interrupted multi-hour job.

**Pattern to mirror:** `modules/protein_studies/boltz/boltz_1/databricks.yml`
— it overlays on-demand per cloud for `register_boltz`. Every new workflow
must do the same for every job it adds. When a submodule defines multiple
jobs (e.g. `enzyme_optimization_v1` has both Fast and Accurate jobs), each
job needs its own block under each cloud target — see
`modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/databricks.yml`.

---

## Architecture Overview

```
Model Registration (Databricks notebook)
  → MLflow PyFunc log_model() → Unity Catalog
    → import_model_from_uc() → GWB models table
      → deploy_model() → Model Serving Endpoint
        → UI workflow calls endpoint via SDK
```

Every model in Genesis Workbench follows this pipeline. The patterns are consistent across all modules.

## Adding a New Model

### Step 1: Create the sub-module directory

Follow existing structure. Example for a new protein model called `mymodel`:

```
modules/protein_studies/mymodel/mymodel_v1/
├── databricks.yml          # DAB bundle config
├── variables.yml           # Cloud-specific variables
├── deploy.sh               # Deployment script
├── destroy.sh              # Teardown script
├── resources/
│   ├── volumes.yml         # UC Volume for model artifacts
│   └── register_mymodel.yml  # Job definition with tasks
└── notebooks/
    ├── 01_register_mymodel.py      # Download weights + define PyFunc + register in UC
    └── 02_import_model_gwb.py      # Import to GWB + deploy endpoint
```

### Step 2: Write the registration notebook (01_register_mymodel.py)

**Pattern** (from all existing models):

```python
# 1. Install dependencies
# %pip install ...
# %restart_python

# 2. Widgets for parameters
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "genesis_schema", "Schema")
dbutils.widgets.text("model_name", "mymodel", "Model Name")
# ... etc

# 3. Download model weights (with skip-if-exists)
model_dir = f"/Volumes/{catalog}/{schema}/{cache_dir}/models/"
model_file = os.path.join(model_dir, "model.pt")
if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
    print(f"Model already exists, skipping download")
else:
    # Download from HuggingFace / Google Drive / NGC / Zenodo
    ...

# 4. Define MLflow PyFunc wrapper
class MyModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load model weights ONCE here (not in predict)
        self.model = load_model(context.artifacts["model_file"])
        self.model.float()  # Force float32 to avoid dtype issues
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input, params=None):
        # Use self.model (pre-loaded)
        # Return dict or DataFrame
        ...

# 5. Dry-load test
context = PythonModelContext(artifacts={...})
model = MyModelWrapper()
model.load_context(context)

# 6. Generate input_example (keep it small!)
# Use 10 cells x 1500 genes, or a single short sequence
# Large examples slow down model logging

# 7. Register in Unity Catalog
mlflow.pyfunc.log_model(
    "mymodel",
    python_model=MyModelWrapper(),
    artifacts={"model_file": str(model_file)},
    pip_requirements="../requirements.txt",
    signature=signature,
    input_example=(input_data, params),
    registered_model_name=f"{catalog}.{schema}.{model_name}",
)
```

**Key gotchas:**
- Always pre-load model weights in `load_context()`, NOT in `predict()` — avoids re-reading GB from disk per request
- Call `self.model.float()` after loading to avoid float16/float32 dtype mismatches
- Keep `input_example` small — 10 cells, not 1000. Must survive HVG/preprocessing filters.
- Use `os.path.exists()` + `os.path.getsize() > 0` to skip downloads on re-deploy
- If the model returns a dict, handle it in the caller (don't assume tensor output)

### Step 3: Write the import/deploy notebook (02_import_model_gwb.py)

**Standard pattern** (identical across all models):

```python
from genesis_workbench.models import (ModelCategory, import_model_from_uc,
                                      deploy_model, get_latest_model_version)
from genesis_workbench.workbench import wait_for_job_run_completion

model_uc_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(model_uc_name)

gwb_model_id = import_model_from_uc(
    user_email=user_email,
    model_category=ModelCategory.PROTEIN_STUDIES,  # or SINGLE_CELL, SMALL_MOLECULES
    model_uc_name=model_uc_name,
    model_uc_version=model_version,
    model_name="MyModel Display Name",
    model_display_name="MyModel Display Name",
    model_source_version="v1.0",
    model_description_url="https://github.com/...",
)

run_id = deploy_model(
    user_email=user_email,
    gwb_model_id=gwb_model_id,
    deployment_name="MyModel",
    deployment_description="Description",
    input_adapter_str="none",
    output_adapter_str="none",
    sample_input_data_dict_as_json="none",
    sample_params_as_json="none",
    workload_type=workload_type,  # GPU_SMALL, GPU_MEDIUM, MULTIGPU_MEDIUM, CPU
    workload_size="Small",
)

result = wait_for_job_run_completion(run_id, timeout=3600)
```

### Step 4: Write the job YAML

```yaml
resources:
  jobs:
    register_mymodel:
      name: register_mymodel
      email_notifications:
        on_failure:
          - ${var.current_user}
      tasks:
        - task_key: register_mymodel_task
          job_cluster_key: job_cluster_gpu
          notebook_task:
            notebook_path: ../notebooks/01_register_mymodel.py
        - task_key: import_mymodel_task
          depends_on:
            - task_key: "register_mymodel_task"
          notebook_task:
            notebook_path: ../notebooks/02_import_model_gwb.py
      environments:
        - environment_key: default
          spec:
            client: '2'
      parameters:
        - name: catalog
          default: ${var.core_catalog_name}
        # ... standard parameters
      job_clusters:
        - job_cluster_key: job_cluster_gpu
          new_cluster:
            num_workers: 0
            node_type_id: ${var.t4_node_type}
            spark_version: '15.4.x-gpu-ml-scala2.12'
            # ... standard config
```

**Cluster selection:**
- Download-only tasks: Use `cpu_node_type` + `cpu-ml` runtime
- Model registration (needs GPU for test prediction): Use `t4_node_type` + `gpu-ml` runtime
- Import/deploy (serverless): No cluster needed, use `environments` with `client: '2'`

### Step 5: Wire into the parent module deploy/destroy scripts

Edit `modules/<module>/deploy.sh`:
```bash
for module in existing_model/existing_v1 mymodel/mymodel_v1
```

Edit `modules/<module>/destroy.sh`:
```bash
for module in existing_model/existing_v1 mymodel/mymodel_v1
```

## Adding a New UI Workflow

### Step 1: Add endpoint mapping

In `modules/core/app/utils/streamlit_helper.py`, add to `_MODEL_ENDPOINT_MAP`:
```python
"MyModel Display Name": "mymodel",
```

The key must match what you pass to `hit_model_endpoint()` or `get_endpoint_name()`. Case-sensitive.

### Step 2: Add endpoint wrapper function

For protein studies, add to `modules/core/app/utils/protein_design.py`:
```python
@mlflow.trace(span_type="LLM")
def hit_mymodel(input_data):
    return hit_model_endpoint('MyModel Display Name', [input_data])[0]
```

For single cell, create utility functions in a dedicated file (e.g., `utils/scimilarity_tools.py`).

For endpoints with complex input schemas (not just a list), call the SDK directly:
```python
response = workspace_client.serving_endpoints.query(
    name=endpoint_name,
    inputs=[{"field1": value1, "field2": value2}],  # Always use inputs=
)
```

### Step 3: Create the workflow UI file

Follow the module's pattern:
- **Protein Studies**: `views/protein_studies_workflows/myworkflow.py` with a `render()` function
- **Single Cell**: `views/single_cell_workflows/myworkflow.py` with a `render()` function

Standard structure:
```python
"""Module — My Workflow tab."""
import streamlit as st
import streamlit.components.v1 as components

def render():
    st.markdown("###### My Workflow")

    # Input section
    c1, c2 = st.columns([3, 1])
    with c1:
        user_input = st.text_area("Input:", ...)
    with c2:
        run_btn = st.button("Run", type="primary")

    # Execution with spinner
    if run_btn:
        status_container = st.container()
        with status_container:
            progress = st.progress(0, text="Starting...")
            spinner = st.empty()
        with spinner, st.spinner("Running..."):
            try:
                result = call_endpoint(user_input)
                st.session_state["my_result"] = result
            except Exception as e:
                st.error(f"Failed: {e}")

    # Results display
    if "my_result" in st.session_state:
        # Show results: tables, charts, Mol* viewer, etc.
        ...
```

### Step 4: Wire into the main page

Edit the module's main view file (e.g., `views/protein_studies.py`):

```python
# Add import
from views.protein_studies_workflows import myworkflow

# Add to tabs
settings_tab, ..., myworkflow_tab = st.tabs(["Settings", ..., "My Workflow"])

# Add tab content
with myworkflow_tab:
    myworkflow.render()
```

## UI Patterns to Follow

### Mol* 3D Viewer
```python
from utils.molstar_tools import molstar_html_multibody
html = molstar_html_multibody(pdb_string)  # or list of PDB strings for overlay
components.html(html, height=540)
```
Always use height=540 for consistency. The viewer includes dark theme CSS automatically.

### Progress bar + spinner (for long operations)
```python
status_container = st.container()
with status_container:
    progress = st.progress(0, text="Starting...")
    spinner = st.empty()
with spinner, st.spinner("Running..."):
    progress.progress(50, text="Halfway...")
    # ... work ...
    progress.progress(100, text="Done!")
```

### Design selector (for multiple results)
```python
design_options = [f"Design {i + 1}" for i in range(len(results))]
selected = st.selectbox("Select design:", design_options)
idx = design_options.index(selected)
st.code(results[idx])
```

### Run selector (for MLflow-based workflows)
```python
from utils.single_cell_analysis import search_singlecell_runs, download_singlecell_markers_df
runs_df = search_singlecell_runs(user_email=user_info.user_email, processing_mode="scanpy")
# Build display names, selectbox, load button...
```

### Session state caching
```python
cache_key = f"result_{run_id}_{params}"
if compute_btn:
    result = expensive_computation()
    st.session_state[cache_key] = result
if cache_key in st.session_state:
    display_results(st.session_state[cache_key])
```

## Adding to Results Viewer (Single Cell)

To add a new analysis section to the existing results viewer:

Edit `views/single_cell_workflows/processing.py`, add a new `st.expander` before the QC section:

```python
st.markdown("---")
with st.expander("My New Analysis", expanded=False):
    cluster_col = _get_cluster_column(df)
    if cluster_col and expr_cols:
        # UI controls
        # Computation
        # Visualization (Plotly)
    else:
        st.warning("Data not available")
```

## Model Categories

When registering models, use the correct `ModelCategory`:
- `ModelCategory.PROTEIN_STUDIES` — protein structure/design models
- `ModelCategory.SINGLE_CELL` — single cell genomics models
- `ModelCategory.SMALL_MOLECULES` — drug discovery models

## Instructions

1. When a user wants to add a new model, walk them through the 5-step process above.
2. Emphasize: always pre-load in `load_context()`, always `model.float()`, always small input_example.
3. For endpoint input format issues, check `_MODEL_ENDPOINT_MAP` case sensitivity and use `inputs=` (never `dataframe_split=`).
4. For UI, follow the module's existing pattern — separate workflow files with `render()` functions.
5. Always add the endpoint mapping in `streamlit_helper.py` before the UI can call it.
6. Test the model locally (dry-load + predict) before registering in UC.

## When to Use This Skill

- User wants to add a new biological AI model to Genesis Workbench
- User wants to create a new UI workflow or tab
- User asks about the model registration pipeline or deployment pattern
- User needs to understand how endpoints are wired to the UI
- User is extending an existing module with a new sub-module
