# Streamlit → React+FastAPI Conversion Patterns

Captured during the Single Cell + Large Molecule migration. Apply these
when porting any remaining Streamlit workflow to this React+FastAPI app.

---

## 1. Routing & concurrency

- **Every FastAPI route is `def`, not `async def`.** Handler bodies call
  blocking SDK methods (Databricks SDK, MLflow, SQL connector, UC Volumes
  download). With `def`, Starlette runs the handler in its threadpool, so
  concurrent requests don't serialize behind one another. `async def` +
  blocking calls **silently destroys concurrency** — single-window apps
  feel fine, multi-window load doesn't.
- Same rule for `Depends()` helpers in `app/auth.py`.

## 2. Long-running synchronous workflows → SSE

Anything with **≥3 natural sub-steps** that runs in-process (vs. async job)
gets streamed progress. The shape:

**Backend** — `app/services/sse.py` exposes `stream_with_progress(work_fn)`.
The work function takes a `progress_callback(pct: int, msg: str)` and
returns the final JSON-serializable result. The helper runs the work in a
daemon thread, queues `progress` events as the callback fires, emits a
terminal `result` (or `error`) frame, and inserts `: keepalive\n\n`
comments every ~20s so the Databricks Apps proxy doesn't drop the stream.

A route looks like:

```python
@router.post("/foo/stream")
def foo_stream(payload: FooReq, _: CurrentUserDep):
    def work(progress_cb):
        progress_cb(5, "Phase one")
        ...
        progress_cb(100, "Done")        # ← mandatory; without this the
        return {...}                     #   last stage stays "active"
    return StreamingResponse(
        stream_with_progress(work),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Service helpers accept `progress_callback=None` and a `pct_start/pct_end`
budget when called inside a larger pipeline (see `teddy.embed_cells`).

**Frontend** — `hooks/useSseMutation.ts` is the SSE-aware drop-in for
`useMutation`. It returns `{ start, reset, abort, setData, isPending,
progress, data, error }`. Render condition for the progress bar is
**`isPending` only** — not `isPending || progress` — otherwise the bar
lingers on the last stale event after the result arrives. See section 5.

`components/RealtimeProgress.tsx` renders the determinate bar + current
message + the **stage checklist** (✓ done / ⏳ active / ○ pending) when a
`stages` prop is supplied. Stage boundaries are `pctEnd` numbers that
mirror the backend's pct markers — keep them in sync.

### Where SSE applies
| Workflow                  | Multi-step?            | SSE                |
|---------------------------|------------------------|--------------------|
| SCimilarity annotation    | gene-order, embed×N, KNN×N, vote | ✅ |
| TEDDY annotation          | preflight, embed×N, KNN×N, vote  | ✅ |
| Cell Similarity search    | embed×N, KNN×N, aggregate        | ✅ |
| Perturbation prediction   | mean expr, scGPT, rank           | ✅ |
| Sequence Search           | embed, ANN, fetch, SW alignment×N | ✅ |
| Protein Design            | ESMFold, RFDiff×N, MPNN×N, ESMFold×N, align | ✅ |
| ESMFold / Boltz / InverseFolding | one endpoint round-trip   | ❌ — keep `WorkflowProgress` time bar |
| Sub-tab MLflow loaders (DE, Markers, Enrichment, Raw) | one MLflow read | ❌ — short enough that staging adds no value |

## 3. Async-job workflows (still useful pattern)

For >60s server work that would blow the Apps proxy timeout, use the
AlphaFold2 pattern: kick off a Databricks Job, poll job status, render
result when artifacts land. Don't try to keep one SSE connection open for
6 hours.

## 4. MLflow artifact persistence — IMPORTANT GOTCHA

Inside Databricks Apps, the runtime injects `MLFLOW_RUN_ID` into the
process. Calling `mlflow.start_run(run_id=...)` raises:
> Cannot start run with ID X because active run ID does not match environment run ID

**Workaround**: use `MlflowClient.log_artifact(run_id=..., local_path=...)`
directly. The MlflowClient API ignores the env var and takes `run_id` as
an explicit parameter. Mirror the storage shape Streamlit uses
(`{ "cluster_col": ..., "results": [...] }`) so older runs stay readable
across the migration. See `single_cell_runs.save_annotation`.

For round-trip persistence on a workflow:

1. **Save** at the end of the SSE work() function. Wrap in `try/except
   logger.warning(...)` so a save failure never breaks the user-visible
   response.
2. **Load** via a `GET /api/{module}/{thing}` route that returns the
   payload or `null`. Treat MLflow `"does not exist"` errors as a clean
   miss; only re-raise real auth/server errors.
3. **Restore on mount** — in the React component, fetch the saved payload
   on `runId` change and seed `mutation.setData(persisted)`.

## 5. SSE render condition + final 100%

Two related bugs caught during rollout:

1. **Render condition** must be `mutation.isPending` only. `isPending ||
   progress` keeps the bar visible after the result lands because
   `progress` still holds the last event.
2. **Every SSE work function must emit `progress_callback(100, ...)`** as
   its last action before returning the result. Without it the bar peaks
   at the second-to-last pct (e.g. 98%) and the last stage shows the
   active-spinner indefinitely.

## 6. Pre-flight asset checks

If a workflow depends on reference data that may not be deployed (TEDDY's
`teddy_cells` table + `teddy_cell_index` VS index), check existence
up-front and raise a single human-readable error. Otherwise downstream
per-cell calls fail silently and the UI shows zero-confidence "Unknown"
rows everywhere — exactly the bug shape that's hardest to debug. See
`teddy._check_teddy_assets_available()`.

## 7. Tab layout — submission vs. analysis

Standardised in Single Cell, applies to any module with these two concerns:

- **Submission / Browse Runs tab** — the form to start a new run + a
  filterable table of past runs.
  - Each row's `View` button opens a `Dialog` modal showing the run
    metadata + QC + Raw Data sub-tabs.
  - The modal closes via Esc / click-outside / ✕.
- **Analysis tab** — **one shared run picker at the top**, then sub-tabs
  that all operate against the selected run.
  - Beneath the picker: a bold two-line `Experiment: …` / `Mode: …`
    header, then the 4-metric stat cards (Total cells, Subsample loaded,
    Clusters, …) and a "Key MLflow metrics" card.
  - Empty-state nudge before a run is selected.
- **Deployed Models tab — always last** across every module page.

Components that drive sub-tab content take a `runId: string | null` (and
optionally `summary` from a parent-loaded run-summary query) prop. Each
renders its own empty-state when `runId` is null — don't embed a picker
inside.

## 8. Tables with a lot of columns

Show **3–4 headline columns + an `i` info button**. Clicking the info
button opens a Dialog with the long-tail fields. See `DeployedModelsTab`.
Per-column width hints via `meta: { thClass, tdClass }` on the column def
(picked up by `DataTable`).

## 9. React 19 + Plotly

`react-plotly.js` v2.6 uses removed legacy lifecycle methods that throw
**React error #130 ("Element type is invalid: got object")** under React
19. Use the custom `PlotlyChart` wrapper that imports `plotly.js`
directly and calls `Plotly.newPlot` / `Plotly.react` in `useEffect`.

## 10. Forms — `react-hook-form` + Zod + numeric inputs

`z.coerce.number()` collides with `zodResolver`'s type inference under
TS 5.6 ("Two different types"). Use `z.number()` plus
`{...register("field", { valueAsNumber: true })}` on the input. See
`RawProcessingTab.RunNewAnalysisForm` for a canonical example.

## 11. Auth split

- User OBO token (`X-Forwarded-Access-Token`) lacks the `model-serving`
  and most service-principal-only scopes.
- Per-request `WorkspaceClient(token=OBO_token, auth_type="pat")` via
  `Depends()` covers user-scoped operations.
- For serving-endpoint queries, VS index queries, and grants, use the
  app SP's `WorkspaceClient()` (no token arg).

## 12. Build + deploy cycle

```bash
cd modules/core/app-react/frontend && npm run build
cd .. && databricks sync . /Workspace/Users/$USER/genesis-workbench-react \
  --include "frontend/dist/**" --full
databricks apps deploy gwb-react \
  --source-code-path /Workspace/Users/$USER/genesis-workbench-react
```

`frontend/dist/` is gitignored by Vite's default — the `--include` flag is
required to ship the built bundle. Static assets in `frontend/public/`
(logo, favicons) are copied into `dist/` by Vite and served via the SPA
fallback in `main.py`.

## 13. NPM registry

The Databricks proxy at `https://npm-proxy.cloud.databricks.com/`
is mandatory inside the corporate network. `npm config set registry
<that URL>` once before any `npm install`.

## 14. Style — terse comments, no over-engineering

- Default to **no** code comments. Add one only when the *why* is
  non-obvious (a hidden constraint, a workaround for a specific bug, an
  invariant). Identifier names handle the *what*.
- Don't add error handling, fallbacks, or validation for scenarios that
  can't happen. Trust internal-code guarantees. Only validate at system
  boundaries (user input, external APIs).
- For new features in an existing tab, keep the visual language of the
  rest of the app: `text-xs uppercase tracking-wide text-muted-foreground`
  for labels, `rounded-md border border-border bg-background` for
  inputs, `rounded-md bg-primary` for primary actions.

## Quick checklist when porting a Streamlit page

1. Is the workflow multi-step? → SSE (`useSseMutation` +
   `RealtimeProgress` with `stages`).
2. Does it produce results worth re-reading on a later session? → MLflow
   artifact save (`MlflowClient.log_artifact`, NOT `start_run`) + load
   route + restore-on-mount.
3. Does it depend on reference data that might not be deployed? →
   pre-flight check that surfaces a clean error.
4. Does it browse past runs? → search table + modal Dialog for details,
   not an inner tab strip.
5. Does it analyze a selected run? → shared run picker at the top of an
   Analysis tab, sub-tabs receive `runId`.
6. Is the route `def`? Is `progress_callback(100, …)` the last call
   before `return`? Is the render condition `isPending` only?
7. Deployed Models tab moved to last position?
