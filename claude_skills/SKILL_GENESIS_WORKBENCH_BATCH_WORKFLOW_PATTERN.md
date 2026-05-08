---
name: genesis-workbench-batch-workflow-pattern
description: Canonical pattern for adding a long-running batch workflow to Genesis Workbench — orchestrator job, registration, app-side dispatcher, MLflow run pre-creation, search past runs, and result dialog. Use this for any new feature where the user clicks Launch in Streamlit, a Databricks job runs for minutes-to-hours, and results are reviewed later.
---

# Batch-Workflow Pattern

When a new feature looks like:

> "User fills a form → app dispatches a Databricks job → job runs for several minutes / hours → user comes back later to review results."

Follow this pattern end-to-end. Don't invent a parallel approach — every batch workflow in this repo (AlphaFold, Disease Biology GWAS / Variant Annotation, Guided Enzyme Optimization, Scanpy) has converged on it. Newer implementations refine it slightly; the most complete reference is the **Guided Enzyme Optimization** stack landed on the `guided_enzyme_creation` branch.

## When to apply

- Any workflow whose runtime exceeds a few seconds. (Sub-second flows that hit a serving endpoint live can use the simpler in-app pattern; this skill is for *job-dispatched* work.)
- Anything the user wants to launch and walk away from.
- Anything that needs to surface result history *after* the user closes the tab.

## What NOT to do (anti-patterns from real bugs in this repo)

These each cost a deploy cycle when they slipped in. Read them before writing code.

1. **Don't render an inline auto-polling result pane on the launch page.** It conflicts with the *Search Past Runs* dialog flow, can't survive a tab refresh, and forces awkward `st.session_state["active_run"]` plumbing. Use the AlphaFold-style success-message + Search-Past-Runs dialog instead. Reference: `modules/core/app/views/protein_studies_workflows/structure_prediction.py:104-186`.

2. **Don't return `job_run.run_id` from the dispatcher and treat it as the MLflow run_id.** They are different things. `job_run.run_id` (int) is the Databricks job-run id; the MLflow run_id (str hex) is what `MlflowClient.get_run()`, `download_artifacts()`, etc. expect. The dispatcher's success message displays `job_run_id`; the *search* discovers the MLflow run via tags.

3. **Don't pass `experiment.name` (the full path `/Users/<email>/mlflow_experiments/<tag>`) to the orchestrator's `mlflow_experiment` job parameter.** The orchestrator's `set_mlflow_experiment` will prepend the user-folder again, producing `/Users/.../mlflow_experiments//Users/.../mlflow_experiments/<tag>` — MLflow's REST endpoint surfaces this as `BAD_REQUEST: For input string: "None"`. Pass the *short tag* (e.g., `"gwb_enzyme_optimization"`) and let the orchestrator resolve.

4. **Don't create the MLflow run only inside the orchestrator notebook.** Cluster cold-start + `pip install` takes 3-5 min; for that whole window the *Search Past Runs* table is empty even though the job is in flight. Pre-create the run from the dispatcher (Disease Biology pattern) and pass `mlflow_run_id` so the orchestrator attaches.

5. **Don't surface raw MLflow run lifecycle status (`RUNNING` / `FINISHED`) in the search-results status column.** Use a progressive `job_status` tag (`submitted` → `started` → stage-specific → `complete` / `failed`) so the user sees the *pipeline stage*, not the lifecycle state.

6. **Don't skip the `register_<feature>_job` notebook + DAB resource.** Without it, the app SP doesn't have `CAN_MANAGE_RUN` on the orchestrator job, and `WorkspaceClient().jobs.list(name=...)` returns empty from the app context — the dispatcher errors with "Orchestrator job '...' not found."

7. **Don't `./deploy.sh core` to refresh app code.** Use `cd modules/core && ./update.sh <cloud>`. `deploy.sh core` re-runs `initialize_module_job` and **wipes settings + user-profile tables**.

8. **Always exact-pin every pip dependency.** No ranges, no `latest`, no unpinned bares. See `SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md` for the rationale (a real outage caused this rule).

## The five layers

Every batch workflow has these five pieces. Each links to the freshest reference (Guided Enzyme Optimization) — copy from there when adding a new feature.

### Layer 1 — Orchestrator Job (DAB)

`modules/<module>/<feature>/<feature_v1>/resources/run_<feature>.yml`

- Defines `run_<feature>` job whose only task runs `notebooks/01_run_<feature>.py`.
- Per-cloud overlay in the submodule's `databricks.yml`:
  - AWS: `aws_attributes.availability: ON_DEMAND`
  - Azure: `azure_attributes.availability: ON_DEMAND_AZURE`
  - GCP: `gcp_attributes.availability: ON_DEMAND_GCP`
  - **All Genesis Workbench batch jobs use on-demand compute on every cloud.** No spot/preemptible. (Hard rule from `SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md`.)
- Job parameters block declares every input the dispatcher will pass. **Always include** `mlflow_run_id` (default `""`):
  ```yaml
  - name: mlflow_experiment
    default: ""
  - name: mlflow_run_name
    default: ""
  - name: mlflow_run_id
    default: ""
  ```
- Reference: `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/resources/run_optimization.yml`.

### Layer 2 — Registration job (DAB + notebook)

`modules/<module>/<feature>/<feature_v1>/resources/register_<feature>_job.yml`
`modules/<module>/<feature>/<feature_v1>/notebooks/register_<feature>_job.py`

The registration job runs **once per deploy** and does two jobs:

a) **Persist the orchestrator job ID(s) to the `settings` table** so the next run of `core/notebooks/grant_app_permissions.py` (which queries `key LIKE '%_job_id'`) keeps the app SP's `CAN_MANAGE_RUN` grant in sync after subsequent redeploys.

b) **Grant the app SP `CAN_MANAGE_RUN` directly** via `set_app_permissions_for_job(job_id, user_email)` so the app can launch the job *immediately* after this deploy (without waiting for the next `update.sh`).

Notebook skeleton (copy from `modules/small_molecule/enzyme_optimization/enzyme_optimization_v1/notebooks/register_enzyme_optimization_job.py`):

```python
dbutils.widgets.text("catalog", "...")
dbutils.widgets.text("schema", "...")
dbutils.widgets.text("run_<feature>_job_id", "", "Orchestrator Job ID")
dbutils.widgets.text("user_email", "...")
dbutils.widgets.text("sql_warehouse_id", "...")
dbutils.widgets.text("databricks_app_name", "genesis-workbench", "Databricks App Name")

# %pip install databricks-sdk==0.50.0 databricks-sql-connector==4.0.3 mlflow==2.22.0
# install gwb library wheel from /Volumes/.../libraries
# %pip install {gwb_library_path} --force-reinstall
# dbutils.library.restartPython()

import os
os.environ["DATABRICKS_APP_NAME"] = databricks_app_name      # set BEFORE importing helpers

from genesis_workbench.workbench import initialize, set_app_permissions_for_job
initialize(core_catalog_name=catalog, core_schema_name=schema,
           sql_warehouse_id=sql_warehouse_id, token=databricks_token)

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

spark.sql(f"""
MERGE INTO settings AS target
USING (SELECT 'run_<feature>_job_id' AS key, '{run_<feature>_job_id}' AS value, '<module>' AS module) AS source
ON target.key = source.key AND target.module = source.module
WHEN MATCHED THEN UPDATE SET target.value = source.value
WHEN NOT MATCHED THEN INSERT (key, value, module) VALUES (source.key, source.value, source.module)
""")

set_app_permissions_for_job(job_id=run_<feature>_job_id, user_email=user_email)
```

Resource YAML wires the job ID via `${resources.jobs.run_<feature>.id}`. Submodule `deploy.sh` runs the registration job after `bundle deploy`:

```bash
databricks bundle run --target $TARGET register_<feature>_job $EXTRA_PARAMS
```

(Foreground, NOT `--no-wait` — registration must finish before the user can launch.)

### Layer 3 — App-side dispatcher (`modules/core/app/utils/<feature>_tools.py`)

The dispatcher's job: pre-create the MLflow run (so the search lights up immediately), launch the orchestrator, return `(job_id, job_run_id)`.

```python
def start_<feature>_job(...) -> Tuple[int, int]:
    catalog = os.environ["CORE_CATALOG_NAME"]
    schema = os.environ["CORE_SCHEMA_NAME"]

    # Any UC volume writes from the app must use Files API — POSIX writes to
    # /Volumes/ raise PermissionError in the Databricks Apps sandbox.
    # Pattern: w.files.upload(file_path, contents=BytesIO(...), overwrite=True)

    experiment = set_mlflow_experiment(
        experiment_tag=mlflow_experiment,            # SHORT TAG, not full path
        user_email=user_info.user_email,
    )

    w = WorkspaceClient()
    job_id = _resolve_orchestrator_job_id(workspace_client=w)

    with mlflow.start_run(run_name=mlflow_run_name,
                          experiment_id=experiment.experiment_id) as pre_run:
        mlflow_run_id = pre_run.info.run_id

        # Discovery tags — drives the in-app Search Past Runs filter.
        # Match the four tags exactly: origin, feature, created_by, job_status.
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "<feature>")
        mlflow.set_tag("created_by", user_info.user_email)
        mlflow.set_tag("job_status", "submitted")

        # Log a few high-level params so the search row has columns *now*
        # (before the orchestrator gets to its own log_params call).
        mlflow.log_param("<key1>", val1)
        ...

        job_run = w.jobs.run_now(
            job_id=job_id,
            job_parameters={
                ...
                "mlflow_experiment": mlflow_experiment,    # short tag
                "mlflow_run_name": mlflow_run_name,
                "mlflow_run_id": mlflow_run_id,            # crucial
                ...
            },
        )
        mlflow.set_tag("job_run_id", str(job_run.run_id))

    return job_id, job_run.run_id
```

`_resolve_orchestrator_job_id`: cache job ID by name, fall back to `RUN_<FEATURE>_JOB_ID` env var if present, otherwise `WorkspaceClient().jobs.list(name=...)` and pick the first match. Raise `RuntimeError` with the exact deploy command if not found.

Search helpers — direct port of `protein_structure.search_alphafold_runs_*`:

```python
def search_<feature>_runs_by_run_name(user_email: str, run_name: str) -> pd.DataFrame:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    experiments = mlflow.search_experiments(
        filter_string="tags.used_by_genesis_workbench='yes'"
    )
    if not experiments:
        return pd.DataFrame()
    exp_map = {e.experiment_id: e.name.split("/")[-1] for e in experiments}
    runs = mlflow.search_runs(
        filter_string=(
            f"tags.feature='<feature>' AND "
            f"tags.created_by='{user_email}' AND "
            f"tags.origin='genesis_workbench'"
        ),
        experiment_ids=list(exp_map.keys()),
    )
    if runs.empty:
        return pd.DataFrame()
    runs = runs[runs["tags.mlflow.runName"].str.contains(run_name, case=False, na=False)]
    return _format_search_runs(runs, exp_map)
```

Plus a sibling `search_<feature>_runs_by_experiment_name(user_email, experiment_name)` that pre-filters experiments by name substring before the run search.

`_format_search_runs(runs, exp_map)` — projects a fixed column list, collapses `tags.` / `params.` / `metrics.` prefixes, returns the formatted DataFrame. Surface `run_id` (hidden in UI), `run_name`, `experiment_name`, key params/metrics, `job_status`. Don't surface internal-only tags like `stop_reason` — they live in MLflow for users who care.

### Layer 4 — Orchestrator notebook (`notebooks/01_run_<feature>.py`)

Three required pieces:

a) **Read `mlflow_run_id` widget** alongside the others:

```python
dbutils.widgets.text("mlflow_run_id", "", "Pre-created MLflow run id (set by dispatcher; empty = create new)")
mlflow_run_id = dbutils.widgets.get("mlflow_run_id") or None
```

b) **Branch `start_run` based on whether dispatcher pre-created**, then attach inside a try/except so failures flip `job_status="failed"`:

```python
from mlflow.tracking import MlflowClient as _MlflowClient
_active_run_id = mlflow_run_id

if mlflow_run_id:
    _run_ctx = mlflow.start_run(run_id=mlflow_run_id)
else:
    _run_ctx = mlflow.start_run(
        run_name=mlflow_run_name or "<feature>",
        experiment_id=experiment.experiment_id,
    )

try:
    with _run_ctx as run:
        _active_run_id = run.info.run_id

        mlflow.log_params({...})

        # Tags re-set defensively (idempotent) — supports running this
        # notebook directly from the Databricks Jobs UI without a dispatcher.
        mlflow.set_tag("origin", "genesis_workbench")
        mlflow.set_tag("feature", "<feature>")
        mlflow.set_tag("created_by", user_email)
        mlflow.set_tag("job_status", "started")        # overwrites "submitted"

        # ... existing loop / processing logic ...
        # Update the job_status tag at every meaningful checkpoint so the
        # Search-Past-Runs row reflects the current pipeline stage:
        mlflow.set_tag("job_status", f"<stage_name>")

        mlflow.set_tag("job_status", "complete")        # final happy path
except Exception as _exc:
    if _active_run_id:
        try:
            _MlflowClient().set_tag(_active_run_id, "job_status", "failed")
            _MlflowClient().set_tag(_active_run_id, "failure_reason", str(_exc)[:500])
        except Exception:
            pass
    raise
```

The else-branch keeps the notebook runnable directly from the Databricks Jobs UI for debugging without a dispatcher.

c) **Set the experiment tag** so it's discoverable by `mlflow.search_experiments(filter_string="tags.used_by_genesis_workbench='yes'")`. This is automatic when you call `set_mlflow_experiment(experiment_tag, user_email)` from `genesis_workbench.models` (it sets the tag internally). Just make sure the orchestrator passes the **short tag** (received via the `mlflow_experiment` widget), NOT the full path.

### Layer 5 — Streamlit view (`modules/core/app/views/<module>_workflows/<feature>.py`)

Three sections, in order:

a) **Form** — inputs for the run.

b) **Dispatch** — AlphaFold-style success message + View Run button:

```python
if run_btn:
    user_info = get_user_info()
    with st.spinner("Dispatching <feature> job..."):
        try:
            job_id, job_run_id = start_<feature>_job(...)
        except Exception as e:
            st.error(f"Failed to dispatch job: {e}")
            return
    st.success(f"Job started with run id: {job_run_id}.")
    st.button(
        "View Run",
        on_click=lambda: open_run_window(job_id, job_run_id),
        key="<feature>_view_run_btn",
    )
```

NO inline polling, NO `st.session_state["active_run"]`. The user reviews results via Search Past Runs.

c) **Search Past Runs + dialog** — direct port of `structure_prediction.py:132-186`:

```python
st.divider()
st.markdown("###### Search Past Runs:")
c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
with c1:
    search_mode = st.pills("Search By:", ["Experiment Name", "Run Name"],
                           selection_mode="single", default="Run Name",
                           key="<feature>_search_mode")
with c2:
    search_text = st.text_input(f"{search_mode} contains:",
                                value="<sensible_default_prefix>",
                                key="<feature>_search_text")
with c3:
    search_btn = st.button("Search", key="<feature>_search_btn")

if search_btn:
    # populate st.session_state["<feature>_search_result_df"]

if "<feature>_search_result_df" in st.session_state:
    view_enabled = _is_viewable_status(
        st.session_state.get("selected_<feature>_run_status", "")
    )
    view_btn = st.button("View", disabled=not view_enabled, ...)

    selected_event = st.dataframe(
        st.session_state["<feature>_search_result_df"],
        column_config={
            "run_id": None,                                         # hidden
            "run_name":         st.column_config.TextColumn("Run Name"),
            "experiment_name":  st.column_config.TextColumn("Experiment"),
            ...
            "job_status":       st.column_config.TextColumn("Stage"),
            "progress":         st.column_config.TextColumn("Progress"),
        },
        use_container_width=True, hide_index=True,
        on_select=_set_selected_<feature>_run_status,
        selection_mode="single-row",
        key="<feature>_search_display_df",
    )

    selected_rows = selected_event.selection.rows
    if len(selected_rows) > 0 and view_btn:
        _display_<feature>_result_dialog(selected_rows)
```

Module-level helpers:

- `_set_selected_<feature>_run_status()` — `on_select` callback; reads `selection.rows[0]` from session state, writes the row's `job_status` to `selected_<feature>_run_status`. Cleans up on deselect.
- `_is_viewable_status(status: str) -> bool` — returns True for `complete`, `failed`, and any `<stage>_complete`. Drives the **View** button's `disabled` flag.
- `@st.dialog("...", width="large") def _display_<feature>_result_dialog(selected_rows):` — the result viewer. Inside:
  - Status banner + `st.caption` for any extra metadata
  - "View in MLflow" link (`open_mlflow_experiment_window(experiment_id)`)
  - Live progress chart (`st.line_chart` of `iter_max_reward_history` etc.)
  - **Top-K table** wrapped in `with st.expander("Top candidates...", expanded=False):` (collapsed by default — keeps the dialog short)
  - Candidate selectbox
  - **Metric row** above the viewer: `st.columns(N)` + `col.metric(label, value)` for each developability axis of the selected candidate (so the user sees the numbers at a glance)
  - Mol\* / 3D viewer
  - Download button

### Optional but recommended — progress dots in the search table

Mirror `disease_biology._PROGRESS_MAP` / `add_progress_column` (`modules/core/app/utils/disease_biology.py:23-54`). For the enzyme stack, the implementation lives at `modules/core/app/utils/enzyme_optimization_tools.py:_ENZYME_PROGRESS_MAP / _enzyme_progress / _add_progress_column`. Define a fixed-width emoji scheme (e.g., `🟩🟩⬜⬜`), call from `_format_search_runs` so both search functions get it for free.

For variable-stage workflows (e.g., enzyme optimization has N iterations × 4 sub-stages each), collapse intermediate stages onto a single bucket so the dot count stays fixed-width. Don't try to render `iter_3_scoring_substep_2` — encode it as one of the four buckets.

## Reference implementations (in order of recency)

| Feature | Module | Notes |
|---|---|---|
| **Guided Enzyme Optimization** | `modules/small_molecule/enzyme_optimization/` + `modules/core/app/utils/enzyme_optimization_tools.py` + `modules/core/app/views/small_molecule_workflows/enzyme_optimization.py` | **Most complete reference.** Has all five layers, progress dots, dialog with metrics + collapsed expander, Fast/Accurate dual-job dispatch toggle, in-process AME failure-handling. Copy from here when adding the next batch workflow. |
| **AlphaFold** | `modules/protein_studies/alphafold/` + `modules/core/app/utils/protein_structure.py` + `modules/core/app/views/protein_studies_workflows/structure_prediction.py:104-186` | The original Search Past Runs UX template. Simpler — single status (`fold_complete`), no progress dots. |
| **Disease Biology — GWAS / Variant Annotation** | `modules/disease_biology/*` + `modules/core/app/utils/disease_biology.py` | Source of the **MLflow run pre-creation pattern** (`disease_biology.py:122-166`) and progress-dots helper. |
| **Scanpy** | `modules/single_cell/scanpy/scanpy_v0.0.1/` | Earlier batch-flow without pre-creation — useful to see what NOT to do for the run-visibility piece. |

When uncertain how to wire something, start by reading the equivalent block of the enzyme_optimization stack — it's been hardened across several deploy cycles in this branch.

## Deploy commands

Two-step (matches the GWB always-rules):

```bash
# 1. Submodule — orchestrator + registration
./deploy.sh <module> <cloud> --only-submodule <feature>/<feature_v1>

# 2. Core app — dispatcher + view
cd modules/core && ./update.sh <cloud>
```

**Never** `./deploy.sh core` — wipes settings + user-profile tables.

## Verification (smoke test for any new batch workflow)

After both deploys land:

1. **Immediate visibility** — click the new feature's launch button with a uniquely-named run. Within seconds, the *Search Past Runs* table should show that row with `job_status=submitted` (or whatever your initial stage value is) and the right discovery tags. If empty, the dispatcher's `mlflow.start_run(...)` block didn't fire — check the *4 anti-patterns* above.
2. **Stage progression** — wait for cluster startup (~3-5 min for cold A10, ~2 min for CPU). Re-search. Status should advance through your stages.
3. **Viewer dialog** — click a row + **View**. Status banner, MLflow link, progress chart, results table all render.
4. **Failure surfacing** — force a failure (malformed input, etc.). Row should land on `job_status=failed`; the dialog should still open and show partial state.
5. **Direct Jobs-UI fallback** — trigger the orchestrator from the Databricks Jobs UI directly with `mlflow_run_id=""`. Should work — orchestrator falls back to creating its own run (Layer 4 else-branch).

If any of these don't work, the issue is almost always one of:

- Missing `register_<feature>_job` deploy step (anti-pattern 6) — app SP doesn't see the job.
- Returning the wrong run_id type (anti-pattern 2) — TypeError in the view.
- Dispatcher passing full path instead of short tag (anti-pattern 3) — orchestrator crashes with `BAD_REQUEST: For input string: "None"`.
- MLflow run created only inside the orchestrator (anti-pattern 4) — search empty for ~5 min.
