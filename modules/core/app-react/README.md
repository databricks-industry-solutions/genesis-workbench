# Genesis Workbench — React + FastAPI port

Phase 0 of the [Streamlit → React migration](../../docs/streamlit-to-react-migration-analysis.md).
Sibling Databricks App to the existing Streamlit `app/`. They run side-by-side until parity.

## Known follow-ups

- **Boltz sync inference can exceed the Databricks Apps proxy timeout (~60s) during cold start.** Returns 504 to the browser even though FastAPI eventually receives the prediction. Fix: move Boltz to the async-job pattern (Start → poll → result) like AlphaFold2, or pre-warm via Start All Endpoints. Tracked, deferred from Phase 4 slice 1.
- **Protein design has the same cold-start-vs-proxy-timeout risk as Boltz**, magnified — ≥5 sequential serving-endpoint calls per Generate. Same fix: move to async-job pattern.
- **Protein Design page layout: convert to left-form / right-viewer split** (mirror the planned Small Molecules pattern). Today the layout stacks the form above and renders the result below; users on widescreen waste space.
- **Sequence Search returns short hits against long queries.** The ANN funnel surfaces low-quality candidates that pass through to Smith-Waterman — review `sequence_search.run_sequence_search` filtering: candidate min-length floor, identity threshold, or a query-length-relative length filter before alignment. The Streamlit version exhibits the same behavior; this is a quality bug in the funnel, not a port regression.
- **TEDDY reference in Cell Similarity + Raw Processing annotation panel.** Currently SCimilarity-only. Streamlit's `cell_similarity.py` and `_umap_fragment` in `processing.py` both run SCimilarity + TEDDY side-by-side. Needs `utils/teddy_tools.py` ported — embed via TEDDY endpoint, search a separate VS index.
- **Trajectory gene-along-pseudotime trendline.** Streamlit uses Plotly's `trendline="lowess"`. React ships raw scatter only — add a LOWESS smoother (client-side or server-side via `statsmodels`) for parity.
- **Per-run markers_flat parquet caching.** Each View Analysis sub-tab independently downloads the parquet from MLflow. For large runs that's ~5s × 7 = repeated cost. An in-process LRU keyed by `run_id` would let all sub-tabs share one parquet read.
- **Cleanup: remove the unmounted `CellTypeAnnotationTab` component** now that annotation lives inside the UMAP sub-tab. The standalone tab component is dead code.
- **Bundle code-splitting.** Plotly is ~3.5 MB and ships in the main chunk. A `React.lazy` around each workflow tab + a Plotly-only sub-chunk would cut initial load to ~150 KB. Pursue when bundle hits a real cold-start pain point.

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
