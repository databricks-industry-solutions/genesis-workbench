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

**Pattern to mirror:** `modules/large_molecule/boltz/boltz_1/databricks.yml`
— it overlays on-demand per cloud for `register_boltz`. Every new workflow
must do the same for every job it adds. When a submodule defines multiple
jobs (e.g. `enzyme_optimization_v1` has both Fast and Accurate jobs), each
job needs its own block under each cloud target — see
`modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/databricks.yml`.

---

## Documentation (hard rule)

Every new feature (UI workflow, model, batch pipeline) ships **with three docs
artifacts in the same PR as the code**. A feature that lands without these is
considered incomplete and should not be merged.

### 1. Per-feature doc page in `modules/core/app/docs/`

One markdown file named `<module>_<feature>.md` (snake_case). Required
sections — see [`modules/core/app/docs/README.md`](../modules/core/app/docs/README.md)
for the canonical template:

- **What it does** (one paragraph: input, output, problem solved)
- **How to use** (UI walkthrough — tab, form fields, expected wait, where
  results appear)
- **Inputs** (schema: file formats, column names, parameter ranges)
- **Outputs** (MLflow run name + artifacts + tags, Delta tables, result dialog
  contents)
- **Underlying models / endpoints** (which serving endpoints, UC models, VS
  indexes the feature depends on; link to the submodule README)
- **Limitations and known issues**

### 2. Bullet under the matching module in the root `README.md`

The "Inside Genesis Workbench" section
(`README.md#inside-genesis-workbench`) has one bullet per module listing the
features under it. Add a bullet for the new feature linking to the doc page
created in step 1.

### 3. Dated entry in the root `CHANGELOG.md`

Follow the existing pattern: a feature-name + date header
(`## <feature-or-module> (YYYY-MM-DD) — <one-line tagline>`) followed by `###`
subsections that explain (a) what changed and why, (b) anti-patterns avoided
and the bug they were caused by, (c) reference implementations or files to
mirror. Don't write a generic "added X" line — the CHANGELOG is meant to be
read by future contributors as a record of *decisions*, not a release log.

**Reference implementations** (mirror these — most recent entries are most
representative): `guided_enzyme_creation (2026-05-07)`, `version_pinning
(2026-04-…)` in [`CHANGELOG.md`](../CHANGELOG.md).

**Order of operations:** write the docs **before** declaring the feature
shipped. Specifically: the user-facing doc in `/docs` is what proves the
feature is actually usable end-to-end — if the doc can't be written without
hand-waving, the UI / inputs / outputs / errors aren't done yet.

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
modules/large_molecule/mymodel/mymodel_v1/
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

In `modules/core/app/backend/app/services/endpoints.py`, add to `_MODEL_ENDPOINT_MAP`:
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

### Step 3: Add a FastAPI route

Pick the right router under `modules/core/app/backend/app/routers/` (one of
`large_molecule.py`, `small_molecule.py`, `single_cell.py`, `genomics.py`).
Define your request/response models with Pydantic and add a `@router.post`
or `@router.get` handler:

```python
class MyWorkflowRequest(BaseModel):
    sequence: str = Field(..., min_length=1)

class MyWorkflowResponse(BaseModel):
    result: str
    viewer_html: str | None = None

@router.post("/my_workflow", response_model=MyWorkflowResponse)
def my_workflow(payload: MyWorkflowRequest, _: CurrentUserDep) -> MyWorkflowResponse:
    w = WorkspaceClient()  # app SP for endpoint calls; OBO tokens lack model-serving scope
    try:
        result = hit_mymodel(w, payload.sequence)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"MyModel call failed: {e}")
    return MyWorkflowResponse(result=result)
```

For long-running operations that should stream stage progress to the UI,
return a `StreamingResponse(stream_with_progress(work), media_type="text/event-stream", headers=_SSE_HEADERS)`
instead — the existing `protein_design_stream`, `binder_design_stream`, and
`diffdock_stream` handlers are good copy-paste references.

### Step 4: Add a React tab component

Create `modules/core/app/frontend/src/components/MyWorkflowTab.tsx`:

```tsx
import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'

export function MyWorkflowTab() {
  const [sequence, setSequence] = useState('')
  const run = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/large_molecule/my_workflow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      })
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },
  })

  return (
    <div className="space-y-4">
      <textarea value={sequence} onChange={(e) => setSequence(e.target.value)} />
      <button onClick={() => run.mutate()} disabled={run.isPending}>Run</button>
      {run.data && <pre>{run.data.result}</pre>}
    </div>
  )
}
```

### Step 5: Add the tab to the page

Open the relevant page under `modules/core/app/frontend/src/pages/` (e.g.
`LargeMolecule.tsx`). Add an `import` and a new entry to the `tabs={[…]}`
prop on `<Tabs>`:

```tsx
import { MyWorkflowTab } from '@/components/MyWorkflowTab'
…
<Tabs
  tabs={[
    …existing tabs…,
    { id: 'my_workflow', label: 'My Workflow', content: <MyWorkflowTab /> },
  ]}
/>
```

## UI Patterns to Follow

### Mol* 3D viewer
The backend builds the viewer HTML and the frontend mounts it via
`<iframe srcDoc={viewerHtml} />` or `dangerouslySetInnerHTML`. Use the
existing helper:

```python
from app.services.molstar import molstar_html_multibody, molstar_html_singlebody

viewer_html = molstar_html_singlebody(pdb, name="My prediction", with_iframe=False)
# return viewer_html in the response model
```

### Realtime progress (SSE)
For workflows that take more than ~5 seconds, expose a `/stream` endpoint
and consume it from the React tab via `useSSE` (see `useSSE.ts`). The
backend uses `stream_with_progress` which adapts a regular work function
into stage events. The frontend's `<RealtimeProgress>` component renders
the progress bar + per-stage messages.

### Search past runs
Mirror the existing patterns in `RunSearchSection.tsx`. The backend offers
`/<workflow>/search?by=run_name|experiment_name&text=...` and the React
component handles paging + the dialog that renders the View result.

### React Query keys
Use `[<module>, '<feature>', <subkey>]` as the queryKey, e.g.
`['large_molecule', 'sequence_search', 'organism']`. Keep them stable so
cache invalidation works across navigation.

### Popovers, dropdowns, and modals — reuse, don't hand-roll (hard rule)
Never write per-component "close on outside click / Esc" logic. There are
shared primitives — use them so the behavior is correct and fixed in one place:

- **Modals / full-screen dialogs** → `Dialog` (`Dialog.tsx`) or `Drawer`
  (`Drawer.tsx`). They own the backdrop + Esc.
- **Lightweight popovers / dropdowns** (e.g. `ClipboardPaste`,
  `GeneResolveInput`) → the **`useOutsideDismiss(ref, onClose, enabled)`** hook
  (`hooks/useOutsideDismiss.ts`):

  ```tsx
  const ref = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  useOutsideDismiss(ref, () => setOpen(false), open)
  return <div ref={ref}>{/* trigger + (open && <panel/>) */}</div>
  ```

**Why it's a hook, not copied per component:** the dismiss listener MUST use a
**capture-phase `pointerdown`** (not a bubble-phase `mousedown`). A bubble-phase
listener can be silently blocked by an ancestor that calls `stopPropagation()`,
leaving the popover stuck open on outside clicks while only Esc closes it — a
real bug we hit in two separately-coded popovers. And **never** dismiss with a
`fixed inset-0` backdrop on a non-modal popover: if it gets stuck it blocks all
page interaction. The hook avoids both traps; copying the logic per page
re-introduces them.

## Model Categories

When registering models, use the correct `ModelCategory`:
- `ModelCategory.LARGE_MOLECULE` — protein structure/design models
- `ModelCategory.SINGLE_CELL` — single cell genomics models
- `ModelCategory.SMALL_MOLECULE` — drug discovery models
- `ModelCategory.GENOMICS` — variant calling / GWAS / VCF ingestion / annotation models

## Instructions

1. When a user wants to add a new model, walk them through the 5-step process above.
2. Emphasize: always pre-load in `load_context()`, always `model.float()`, always small input_example.
3. For endpoint input format issues, check `_MODEL_ENDPOINT_MAP` case sensitivity and use `inputs=` (never `dataframe_split=`).
4. For UI, follow the module's existing pattern — separate workflow files with `render()` functions.
5. Always add the endpoint mapping in the endpoint map in `services/endpoints.py` before the UI can call it.
6. Test the model locally (dry-load + predict) before registering in UC.

## When to Use This Skill

- User wants to add a new biological AI model to Genesis Workbench
- User wants to create a new UI workflow or tab
- User asks about the model registration pipeline or deployment pattern
- User needs to understand how endpoints are wired to the UI
- User is extending an existing module with a new sub-module
