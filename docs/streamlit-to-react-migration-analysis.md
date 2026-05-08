# Streamlit → React Migration Feasibility Analysis

## Context

You're considering rewriting the Genesis Workbench Streamlit UI as a React (TypeScript) SPA backed by a FastAPI service. Stated motivations: **better UX, better performance, true multi-user concurrency.** This analysis scopes the effort, identifies the real blockers, and recommends an incremental path.

The rewrite is **technically very feasible** — the codebase is well structured, business logic is largely decoupled from Streamlit, and Databricks Apps natively supports React+FastAPI. The hard parts are not the framework swap; they are (a) the volume of screens, (b) the 3D molecular viewer integration, and (c) the auth model shift.

**Headline estimate:** ~7–11 weeks for one experienced full-stack engineer to reach a feature-equivalent v1 (core app). Two engineers in parallel: ~5–7 weeks. This excludes hardening, accessibility, and cross-browser QA.

---

## What You're Migrating

### Scope at a glance

| Surface | Count | Lines (approx) |
|---|---|---|
| Top-level pages (`views/*.py`) | 8 | ~2,000 |
| Workflow sub-views (`views/*_workflows/*.py`, `views/nvidia/*.py`) | 18 | ~4,100 |
| Custom 3D viewer integration (`utils/molstar_tools.py` + small_molecule_tools) | 1 component, used in ~6 views | ~370 |
| Util modules — pure Python (reusable behind FastAPI) | 10 of 11 | majority |
| Util modules — Streamlit-coupled | 1 (`utils/streamlit_helper.py`) | needs refactor |
| **Total UI Python LOC** | **26 files** | **~6,100** |

Static assets are minimal (logos, a couple of GIFs, one workflow diagram). No external CSS files — all styling is inline `st.markdown(unsafe_allow_html=True)`.

### Ecosystem (modules/ outside core/app)

The other modules (`single_cell/`, `disease_biology/`, `small_molecule/`, etc.) contain **job notebooks and deployment YAML**, not UI. They run on Databricks as jobs/model-serving endpoints. **The React rewrite does not touch these.** They continue to be invoked the same way (jobs.run_now, serving endpoint queries) — only the caller changes from a Streamlit page to a FastAPI route.

---

## What's Easy

These translate cleanly with standard React patterns and require no novel work:

- **Backend logic reuse (~90% of `utils/`).** 10 of 11 util files are pure Python — `enzyme_optimization_tools.py`, `disease_biology.py`, `sequence_search_tools.py`, `scimilarity_tools.py`, `small_molecule_tools.py`, `protein_design.py`, `protein_structure.py`, `structure_utils.py`, `single_cell_analysis.py`, `molstar_tools.py`. They wrap Databricks SDK, MLflow, SQL warehouse, and serving endpoint calls. Move them as-is into FastAPI routes.
- **Job dispatch + polling.** Pattern is already async-on-Databricks, sync-from-UI. FastAPI endpoints expose `POST /jobs/start` (returns `run_id`) and `GET /jobs/{run_id}/status`. React polls with React Query / TanStack Query.
- **Tabs and navigation.** `st.navigation` + `st.tabs` → React Router. Trivial.
- **Modal dialogs (~8 `@st.dialog` usages).** → Radix UI / Headless UI / shadcn Dialog.
- **Forms.** Even the 30+ `st.form` blocks become React Hook Form + Zod schemas. The forms ARE long, but each one is mechanical.
- **Most data tables (~50 `st.dataframe`).** TanStack Table handles sort/filter/select. Row-selection callbacks (currently `session_state[key].selection`) become standard `onRowSelect` handlers.
- **Plotly charts.** `react-plotly.js` is a drop-in. Generate the figure dict server-side (already happens), send JSON, render in React.
- **File uploads** (FASTA, PDB, h5ad, VCF). Standard multipart upload to FastAPI; the parsing helpers in `utils/protein_structure.py` and friends move server-side unchanged.
- **Settings, monitoring, user profile.** Small, list/form-shaped pages. ~1–2 days each.

---

## What's Medium

Real work, but no unknowns:

- **Cross-page state.** Streamlit's `st.session_state` carries `user_settings`, `deployed_modules`, `doc_index`, plus per-workflow caches (`available_*_models_df`, `*_run_search_result_df`, `selected_row_index`, `_last_assistant_query`, …). Total: 30+ keys. In React this becomes Zustand stores (or Redux Toolkit) + React Query for server state. Plan ~1 week to design this layer up front — getting it wrong cascades.
- **Conditional forms.** Several pages change fields based on prior selections (`processing.py` scanpy↔rapids, `structure_prediction.py` model selector → param schema, `small_molecules.py` sub-workflow routing). Doable with React Hook Form `watch()`, but each conditional form needs its own design pass.
- **Two heaviest screens.** `views/disease_biology.py` (995 lines, 5 tabs, 6+ dialogs, GWAS + variant-calling + VCF + annotation flows) and `views/single_cell_workflows/processing.py` (805 lines, scanpy/rapids dual mode, plotly results) — each is roughly a week on its own.
- **Auth model shift.** Currently `streamlit_helper.get_user_info()` reads `X-Forwarded-Access-Token` from the Streamlit request and instantiates `WorkspaceClient(token=..., auth_type="pat")`. Databricks Apps injects this header at the proxy. **A FastAPI service inside the same Databricks App still receives this header**, so the auth mechanism *can* be preserved — but you should redesign it as proper FastAPI middleware that validates the token, caches `WorkspaceClient` per request, and surfaces the user to route handlers via `Depends()`. Not hard, but must be done right once.
- **Multi-user concurrency (your stated goal).** Today, `st.session_state["__gwb_user_info"]` is per-Streamlit-session, but Streamlit's process model and rerun semantics serialize work and waste compute on every interaction. FastAPI gives you true per-request concurrency and lets you cache the `WorkspaceClient` per user. This is *the* win you're after — but it implies discipline: zero shared mutable state in FastAPI, all state lives in the React client or Lakebase.

---

## What's Hard

Three real risks. Plan budget here:

### 1. Mol* (Molstar) 3D viewer — the single biggest unknown

- Used by ~6 workflows (structure prediction, protein design, inverse folding, enzyme optimization, ligand binder design, motif scaffolding).
- Today: `utils/molstar_tools.py` generates HTML/JS, embedded via `st.components.v1.html()` in an iframe.
- For React: two options — (a) keep the iframe approach (fastest, works day 1, but limits React↔viewer interaction), or (b) integrate `molstar` as an npm dependency directly in React (better UX, but requires WebGL context lifecycle management, prop-driven structure loading, and event bubbling for selections).
- **Recommend (a) first**, upgrade to (b) only if you need bidirectional interaction (e.g., "click a residue in the viewer → highlight a row in the React table"). Budget: 2–3 days for (a); 1.5–2 weeks for (b).

### 2. Long-running job UX

Streamlit's "click button → spinner → user clicks Search later" works because reruns are cheap. In React, you need to design:

- Progress indication for jobs that take 5–60 minutes (ESMFold, Boltz, enzyme optimization).
- A "running jobs" tray / notifications so users can leave the page and come back.
- WebSocket or SSE for log streaming, OR polling with backoff (polling is simpler — start there).

This is genuinely better UX than Streamlit, but it's net-new design work. Budget 1 week.

### 3. The volume itself

26 view files × even 1–2 days each = 4–8 weeks of mechanical porting alone. There is no shortcut. AI assistance helps but does not eliminate this.

---

## What You'd Actually Build

```
Databricks App
├── frontend/  (React + TypeScript + Vite)
│   ├── src/
│   │   ├── routes/                      # React Router; mirrors views/
│   │   ├── features/
│   │   │   ├── single-cell/
│   │   │   ├── protein-studies/
│   │   │   ├── small-molecules/
│   │   │   ├── disease-biology/
│   │   │   └── settings/
│   │   ├── components/
│   │   │   ├── MolstarViewer.tsx        # Mol* wrapper
│   │   │   ├── JobStatusBadge.tsx
│   │   │   └── ...
│   │   ├── stores/                      # Zustand: user, deployedModules, …
│   │   ├── api/                         # generated client (openapi-typescript)
│   │   └── lib/
│   └── ...
└── backend/  (FastAPI + uvicorn)
    ├── app/
    │   ├── main.py
    │   ├── auth.py                      # OBO header → WorkspaceClient (Depends)
    │   ├── routers/
    │   │   ├── jobs.py
    │   │   ├── models.py
    │   │   ├── single_cell.py
    │   │   ├── protein_studies.py
    │   │   ├── small_molecules.py
    │   │   ├── disease_biology.py
    │   │   └── settings.py
    │   └── services/                    # ← copies of modules/core/app/utils/*.py (the pure-Python ones)
    └── requirements.txt
```

The Databricks Apps framework supports this exact shape. The on-behalf-of-user header continues to flow to FastAPI.

---

## Effort Breakdown

| Phase | Work | Calendar (1 eng) |
|---|---|---|
| 0. Foundation | Project scaffold (Vite+React+FastAPI), auth middleware, design system (shadcn/ui or MUI), Zustand stores, API client codegen, CI | 1.0 wk |
| 1. Easy pages | Settings, Monitoring, User Profile, Home | 0.5 wk |
| 2. Tab/launcher pages | Single Cell, Protein Studies, Small Molecules, Disease Biology shells + model selectors | 0.5 wk |
| 3. Mol* viewer (iframe approach) | Wrap `molstar_tools` HTML output in a React component | 0.5 wk |
| 4. Protein workflows (4) | structure_prediction, sequence_search, protein_design, inverse_folding | 1.5 wk |
| 5. Small-molecule workflows (5) | enzyme_optimization, ligand_binder_design, motif_scaffolding, admet_safety, binder_design | 1.5 wk |
| 6. Single-cell workflows (4) | processing (heavy), perturbation, cell_similarity, cell_type_annotation | 1.5 wk |
| 7. Disease biology (5 tabs) | Variant calling, GWAS, VCF ingest, annotation, settings | 1.5 wk |
| 8. NVIDIA / BioNeMo | bionemo_esm, parabricks | 0.3 wk |
| 9. Polish | Job-status tray, error states, loading skeletons, dark theme, accessibility pass, deploy + smoke test | 1.0 wk |
| **Total (1 engineer)** | | **~9.3 wk** |
| **Total (2 engineers in parallel after phase 0)** | | **~5.5 wk** |

Risk buffer: +20% for the heavy screens and integration debugging. Realistic: **7–11 weeks for one engineer, 5–7 for two.**

---

## Recommendation

**Go incrementally, keep Streamlit alive during the migration.**

1. Stand up the new React+FastAPI app **alongside** the existing Streamlit app as a sibling Databricks App. Don't try to replace Streamlit in-place.
2. Pick the **smallest valuable slice first**: Settings + User Profile + Home. Validate auth, stores, deploy, theming end-to-end on something low-risk. (~1.5 weeks including foundation.)
3. **Then port the workflow that has the most user pain** — likely one of the heavy ones (disease_biology or processing). Doing a hard one second forces all the architectural decisions early.
4. **Defer Mol* deep integration.** Use the iframe wrapper for v1. Only invest in native-React Mol* if usage data shows you need bidirectional interaction.
5. Migrate the rest workflow-by-workflow. Users can run both apps until parity is reached.

Why incrementally: a big-bang rewrite of 26 screens before any goes to users is the highest-risk path. The auth+state+job-polling architecture must be validated against real workflows early, and the scientific viewers will surface integration gotchas you can't predict from inspection alone.

---

## Critical Files Referenced

**Streamlit entry / nav:**
- `modules/core/app/home.py` (189 lines) — entry point, `st.navigation` setup
- `modules/core/app/app.yml` — launch config
- `modules/core/app/.streamlit/config.toml` — theme

**Heaviest screens (port last or split across people):**
- `modules/core/app/views/disease_biology.py` (995 lines)
- `modules/core/app/views/single_cell_workflows/processing.py` (805 lines)
- `modules/core/app/views/small_molecule_workflows/enzyme_optimization.py` (595 lines)

**Reusable utils (move to FastAPI services/ as-is):**
- `modules/core/app/utils/enzyme_optimization_tools.py`
- `modules/core/app/utils/single_cell_analysis.py`
- `modules/core/app/utils/disease_biology.py`
- `modules/core/app/utils/sequence_search_tools.py`
- `modules/core/app/utils/scimilarity_tools.py`
- `modules/core/app/utils/small_molecule_tools.py`
- `modules/core/app/utils/protein_design.py`
- `modules/core/app/utils/protein_structure.py`
- `modules/core/app/utils/structure_utils.py`
- `modules/core/app/utils/molstar_tools.py`

**Must refactor (do not copy as-is):**
- `modules/core/app/utils/streamlit_helper.py` — auth, dialogs, session state. Auth → FastAPI middleware; dialogs → React; session state → Zustand + React Query.

**Hardest UI integration:**
- Mol* viewer references in: `protein_studies_workflows/structure_prediction.py`, `protein_design.py`, `inverse_folding.py`; `small_molecule_workflows/enzyme_optimization.py`, `ligand_binder_design.py`, `motif_scaffolding.py`.

---

## Verification (if you proceed)

This analysis is a scoping exercise, not a code change — there's nothing to "test." But before committing to the rewrite, validate the two highest-risk assumptions with a 1–2 day spike:

1. **Auth spike.** Stand up a minimal FastAPI app inside Databricks Apps, confirm `X-Forwarded-Access-Token` arrives and `WorkspaceClient(token=...)` works for the current user. If yes, the auth migration is de-risked.
2. **Mol* spike.** Drop the existing `molstar_tools.html` output into a React `<iframe srcDoc={...}/>` component. If the structure renders and the iframe approach is acceptable visually, the hardest integration is de-risked. Only if this fails do you need to plan native-Mol* React integration.

If both spikes pass, the rest of the work is volume, not risk.
