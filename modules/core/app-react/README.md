# Genesis Workbench — React + FastAPI port

Phase 0 of the [Streamlit → React migration](../../docs/streamlit-to-react-migration-analysis.md).
Sibling Databricks App to the existing Streamlit `app/`. They run side-by-side until parity.

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
