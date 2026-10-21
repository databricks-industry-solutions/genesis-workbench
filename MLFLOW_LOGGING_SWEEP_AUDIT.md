# MLflow Logging Sweep — Audit & Fix Plan (defect #3)

**Status: AUDIT ONLY — no code changed.** Fixes deferred to the NEXT deploy (UI changes land first).
Goal of the sweep: *every workflow step and every serving-endpoint call is recorded in MLflow* so a
scientist/operator can see in Past Runs what a run actually did — and, critically, when an endpoint
**silently degraded** (a caught exception returned `0.0`/empty/`None`, masking a real failure as data).

Audited 2026-10-21 against: the shared executor (`genesis_workbench/executor.py`), the AI-canvas
orchestrator (`modules/core/notebooks/run_ai_canvas_workflow.py`), and the four batch orchestrators
(antibody, vaccine, enzyme, GenMol molecule-optimization) + their `notebooks/utils.py` helpers.

---

## What's already GOOD (don't touch)

- **AI-canvas orchestrator** (`run_ai_canvas_workflow.py`): one pre-created run; `job_status` lifecycle
  (started→running→complete/failed) + `failure_reason`; per-node status tags `node:<id>:status`
  (pending→running→complete/failed) + `:error` + `:job_run_id`; `node_count`/`edge_count` params;
  persists `results/workflow_results.json` (all node outputs) on success AND failure; captures the
  originating child-job error as `results/node_<id>_job_error.json`. **Step (node) coverage is strong.**
- **Batch orchestrators** (antibody/vaccine/enzyme): proper `start_run` (attach to pre-created
  `mlflow_run_id` or create), `job_status` lifecycle + `failure_reason` in an OUTER try/except,
  per-iteration summary metrics (max/mean reward), per-candidate reward axes logged as
  `<cand_id>/<axis>` metrics, final artifacts (CSV, FASTA, top-K PDBs, MHC burden, motif positions).
- **Executor job adapters** (`_job_*`): pre-create a per-node child MLflow run, dispatch the job INTO it
  (the job logs into that run), tag `job_run_id`, wait, and read declared output back.

---

## GAP FINDINGS

### A. Executor + AI-canvas (verified in-session)

- **A1 — per-endpoint-call telemetry is invisible (cross-cutting).** The chain functions
  (`_chain_ligand_binder_design`, `_chain_motif_scaffolding`, `_chain_protein_design`,
  `_chain_protein_binder_design`, `_chain_admet_screen`) call ESMFold / RFDiffusion / ProteinMPNN /
  Proteina-Complexa / DiffDock / Boltz directly and record nothing per call. Per-call failures are
  **swallowed with `logger.info`** (executor.py:329-330, 363-364, 370-371, 377-378, 425-426, 434-435,
  204-205) → a design silently missing its dock/affinity leaves no MLflow trace; only the node's final
  status is recorded. Same for the `kind=="endpoint"` node (`run_ai_canvas_workflow.py:215/218`) and
  `_query_endpoint` (executor.py:114).
  - Note: the executor is deliberately "presentation-agnostic — the caller adds MLflow on top"
    (executor.py:10-12, 244). Chains run **inside** the orchestrator's active run (synchronous), so
    `mlflow.active_run()` is available to a guarded tracer; `mlflow` is already a dep here (job adapters).
- **A2 — child-run failure not tagged.** `_dispatch_wait` (executor.py:587) raises on a failed job run
  but does NOT set the child run's `job_status=failed`/`failure_reason` → a job that crashes without
  self-tagging leaves its child run stuck at `job_status=submitted`. (Partially mitigated at the PARENT:
  the AI-canvas node catch tags the node failed + captures the child error artifact.)

### B. Batch orchestrators (from the notebook audit)

Common pattern across all four: lifecycle + per-candidate metrics good; **endpoint calls SILENT**;
**inner exceptions caught-and-degraded with NO MLflow record** (run "succeeds" with masked data).

- **Antibody** (`01_run_antibody_design.py`): generation block (RFD4 generate → VHH numbering/gate →
  CDR graft/ProteinMPNN → ESMFold refold, ~lines 408-481) entirely silent — a batch can shrink 8→0 at
  the VHH gate (~line 554) with no record why. Endpoint wrappers (Boltz/NetSolP/DeepSTABp/MHCflurry/
  HLAIIPred/PLTNUM/ESMFold/ProteinMPNN, ~lines 352-372, 455) return `0.0`/empty on failure, untagged.
- **Vaccine** (`01_run_vaccine_immunogen.py`): endpoint calls (~lines 307-314, 342, 360) silent;
  ProteinMPNN redesign success/fail per iteration unrecorded; raw per-candidate `motif_rmsd` not a
  discrete metric (only composite).
- **Enzyme** (`01_run_optimization.py`): BOTH generation paths untraced — `call_ame()` (Fast) and
  `run_ame_with_rewards()` (Accurate, in-process FK steering) opaque (~lines 520-537); ProteinMPNN +
  ESMFold redesign failures printed with ⚠️ but NOT MLflow-tagged (~lines 664-699).
- **GenMol molecule-optimization** (`03_run_molecule_optimization.py`): **BIGGEST single gap —
  per-candidate metrics are NOT logged as MLflow metrics** (only per-iteration summaries at ~line 358 +
  `top_k.json`/`trajectory.json` artifacts), unlike the protein workflows. Can't query "molecules with
  QED>0.7" in the MLflow UI. Endpoint calls (GenMol invoke ~line 147, ClinTox ~161, ESMFold ~206, ESM
  embed ~214, DiffDock ~230) silent; if ESMFold/ESM-embed fail, docking is disabled for the whole run
  with only a `print` (~line 223), no tag.

Endpoint wrappers live in each module's `notebooks/utils.py` (antibody/vaccine/enzyme); GenMol has inline
helpers (no shared utils). None log to MLflow today.

### Cross-notebook summary

| Workflow | Per-candidate metrics | Endpoint calls traced | Silent-degradation risk |
|---|---|---|---|
| Antibody | ✅ | ❌ | generation 8→0 unrecorded; masked `0.0` scores |
| Vaccine | ✅ | ❌ | ProteinMPNN/endpoint failures masked |
| Enzyme | ✅ | ❌ | both generation paths opaque; redesign fails only to stdout |
| GenMol opt | ❌ (artifacts only) | ❌ | per-candidate not searchable; docking-disabled untagged |
| AI-canvas/executor | n/a (node-level ✅) | ❌ (A1) | per-call failures swallowed; child run stuck "submitted" (A2) |

---

## PRIORITIZED FIX PLAN (next deploy)

**Tier 1 — the core of "every endpoint call recorded" (cross-cutting, highest value)**
1. **Traced endpoint-call helper + failure surfacing.** One lightweight wrapper that records, to the
   ACTIVE mlflow run (guard on `mlflow.active_run()`): per-endpoint call count, error count, and — on a
   caught-and-degraded failure — a visible metric/tag (e.g. `endpoint.<name>.calls`,
   `endpoint.<name>.errors`, and a `endpoints_degraded` tag naming the endpoint + which candidate/seq).
   Apply at the wrappers in the three `notebooks/utils.py` + GenMol's inline helpers (they run inside the
   orchestrator's active run) AND in the executor chains via a `_traced_query` wrapper. Key any running
   counters by `run_id` so the long-lived UI/MCP process doesn't accumulate across runs.
2. **GenMol per-candidate metrics.** Log discrete per-candidate metrics (qed, tox, dock_confidence,
   reward, feasible) as `<cand_id>/<axis>` like the protein workflows, so molecules are searchable.

**Tier 2 — step/generation coverage + error propagation**
3. **Generation-step tracing.** Log input batch size → output count for RFD4/AME/GenMol generation (so an
   8→0 gate shrink is recorded) + a `generation` phase tag; record VHH-numbering/gate drop counts.
4. **Child-run failure tagging.** In executor `_dispatch_wait`, on terminal non-success set the child
   run `job_status=failed` + `failure_reason` before raising.

**Tier 3 — nice-to-have**
5. Per-call latency metrics (perf observability).
6. Raw per-candidate `motif_rmsd` as discrete metrics (vaccine/enzyme).

**Risks / design notes:** keep the executor dep-light (the tracer must be a no-op when there's no active
run, and must not import anything beyond mlflow, already present); never let a logging failure break a
workflow (wrap all tracing in try/except); counters keyed per run to avoid cross-run bleed in the app
process. Validate on one workflow (GenMol opt — it has the most gaps) before propagating.
