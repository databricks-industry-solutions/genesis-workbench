# Genesis Workbench — React + FastAPI port

Phase 0 of the [Streamlit → React migration](../../docs/streamlit-to-react-migration-analysis.md).
Sibling Databricks App to the existing Streamlit `app/`. They run side-by-side until parity.

## Known follow-ups

- **Boltz sync inference can exceed the Databricks Apps proxy timeout (~60s) during cold start.** Returns 504 to the browser even though FastAPI eventually receives the prediction. Fix: move Boltz to the async-job pattern (Start → poll → result) like AlphaFold2, or pre-warm via Start All Endpoints. Tracked, deferred from Phase 4 slice 1.
- **Protein design has the same cold-start-vs-proxy-timeout risk as Boltz**, magnified — ≥5 sequential serving-endpoint calls per Generate. Same fix: move to async-job pattern.
- **Add a filter widget bound to `run_name` in `variant_annotation_dashboard.lvdash.json`** so the embedded Lakeview dashboard works inside the React Variant Annotation popup (or via URL-param binding). Without a widget, the parameter never binds and all `:run_name` queries fail with `UNBOUND_SQL_PARAMETER`. Today the popup keeps the inline pathogenic-variants table + an "Open dashboard ↗" link to the full dashboard. Requires re-registering the dashboard via the initial_setup job.
- **`batch_models` schema: add a `kind` column ('model' vs 'package')** so the Models & Packages drawer doesn't have to maintain a `WORKFLOW_PACKAGES` allow-list in `DeployedModelsTab.tsx`. Touches `initialize_*` core notebook (schema), every register notebook that writes a row, the `/api/models/batch` response shape, and the frontend.
- **Multi-app SP grants are ad-hoc on the current deployment.** Every orchestrator job (13 across disease_biology, protein_studies, single_cell, small_molecule, bionemo) + the four volumes the React backend touches (`enzyme_optimization`, `alphafold`, `teddy`, `scanpy_reference`) currently have the gwb-react SP granted by hand because the jobs/volumes were created before the React app existed in `DATABRICKS_APP_NAMES`. A planned destroy + redeploy with `DATABRICKS_APP_NAMES=genesis-workbench,gwb-react` set will make grants permanent — each register/initialize notebook now calls `set_app_permissions_for_job` AND `set_app_permissions_for_volume` with the multi-app list, so a clean install provisions both SPs end-to-end.

## Layout

```
app-react/
├── app.yml                # Databricks Apps launch (uvicorn)
├── requirements.txt       # Python deps installed by Databricks Apps
├── backend/
│   └── app/
│       ├── main.py        # FastAPI app + static mount of frontend/dist/
│       ├── auth.py        # Depends() that builds WorkspaceClient from X-Forwarded-Access-Token
│       └── routers/
│           ├── health.py
│           └── me.py
└── frontend/
    ├── .npmrc             # pinned to https://npm-proxy.cloud.databricks.com/
    ├── package.json
    ├── vite.config.ts     # dev proxy /api → :8000, build outputs to ./dist
    └── src/
        ├── main.tsx
        ├── App.tsx        # Router + QueryClient
        ├── components/Layout.tsx
        ├── pages/         # Home, Placeholder
        ├── stores/        # Zustand
        ├── api/           # openapi-fetch client (schema gen-able)
        ├── types/         # manual API types (to be replaced by codegen)
        └── lib/utils.ts
```

## Dev loop (local)

Two terminals:

```bash
# Terminal 1 — backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (Vite HMR, proxies /api/* to :8000)
cd frontend
npm install        # pinned to https://npm-proxy.cloud.databricks.com/ via .npmrc
npm run dev
```

Open http://localhost:5173. Auth and `/api/me` will fail locally (no `X-Forwarded-Access-Token`
header) — that's expected. Health and routing work without auth.

## Deploy (Databricks Apps)

```bash
# Build the frontend
cd frontend && npm run build && cd ..

# Sync + deploy. Note: --include "frontend/dist/**" is REQUIRED — Vite's
# auto-generated frontend/.gitignore excludes dist/, and `databricks sync`
# honors gitignores by default. Without the override, the built UI never
# reaches the workspace and the app responds with {"detail":"Not Found"}.
SRC=/Users/$(databricks current-user me | jq -r '.userName')/gwb-react-source
databricks sync --full \
  --include "frontend/dist/**" \
  --exclude "frontend/node_modules/**" \
  --exclude "**/__pycache__/**" \
  --exclude "backend/.venv/**" \
  --exclude "**/*.pyc" \
  . "$SRC"
databricks apps deploy gwb-react --source-code-path "/Workspace$SRC" --mode SNAPSHOT
```

## Codegen

After the backend is running locally, regenerate `src/api/schema.ts`:

```bash
cd frontend
npm run gen:api
```

The generated `schema.ts` replaces `src/types/api.ts` as the source of truth for typed
endpoints.
