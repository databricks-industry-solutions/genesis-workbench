# Workflow → Shared Capability Core: migration tracker

Tracks every user-facing GWB workflow and its migration to the shared capability
core (`genesis_workbench/capabilities.py` + `executor.py`), per `MCP_SERVER_APP_PLAN.md`.

Two independent dimensions:
- **Capability** — registered so **Vortex + MCP** can use it: `endpoint` (live in
  `model_deployments`), `job` (row in `prebuilt_workflows`), or `chain` (executor
  `_CHAINS`). 
- **UI delegates** — the UI service runs via `execute_capability`/`run_chain`
  instead of its own bespoke endpoint/chain code (Step 4). The UI keeps
  presentation (SSE progress, MLflow, Mol* viewer) on top via the `progress` callback.

Legend: ✅ done · 🟡 partial · ⬜ not yet.

## Large Molecule
| Workflow | Kind | Capability (Vortex/MCP) | UI delegates |
|---|---|---|---|
| ESMFold (structure) | endpoint | ✅ | ⬜ |
| Boltz (structure) | endpoint | ✅ | ⬜ |
| AlphaFold | job | ✅ (`run_alphafold`) | ⬜ (UI dispatches directly) |
| Inverse Folding | endpoint (ProteinMPNN) | ✅ (endpoint exists) | ⬜ |
| **Protein Design** | chain | ✅ | ✅ **(done — `make_designs` delegates)** |
| Protein Binder Design | chain (`protein_binder_design`) | ✅ **(built)** | ⬜ (Phase B) |
| Guided Enzyme Optimization | job | ✅ (`run_enzyme_optimization_gwb`) | ⬜ |

## Small Molecule
| Workflow | Kind | Capability (Vortex/MCP) | UI delegates |
|---|---|---|---|
| ADMET & Safety | chain (`admet_screen`) | ✅ | 🟡 (chain exists + has progress; ADMET tab still bespoke for per-predictor toggles) |
| DiffDock (docking) | endpoint | ✅ | ⬜ |
| Ligand Binder Design | chain (`ligand_binder_design`) | ✅ **(built)** | ⬜ (Phase B) |
| Motif Scaffolding | chain (`motif_scaffolding`) | ✅ **(built)** | ⬜ (Phase B) |
| GenMol Generate | endpoint | ✅ | ⬜ |
| Guided Molecule Optimization | job | ✅ (`run_molecule_optimization_gwb`) | ⬜ |

## Genomics
| Workflow | Kind | Capability (Vortex/MCP) | UI delegates |
|---|---|---|---|
| Variant Calling | job | ✅ (`gwas_parabricks_alignment`) | ⬜ |
| VCF Ingestion | job | ✅ (`vcf_ingestion_glow`) | ⬜ |
| Variant Annotation | job | ✅ (`variant_annotation_clinical`) | ⬜ |
| GWAS | job | ✅ (`gwas_glow_analysis`) | ⬜ |

## Single Cell
| Workflow | Kind | Capability (Vortex/MCP) | UI delegates |
|---|---|---|---|
| Single Cell Processing (scanpy) | job | ⬜ (not registered) | ⬜ |
| SCimilarity Annotation / Similarity | endpoint | 🟡 (some scimilarity endpoints registered) | ⬜ |
| TEDDY Annotation | endpoint | ✅ | ⬜ |
| scGPT Perturbation | endpoint | ✅ | ⬜ |
| DE / Enrichment / Trajectory / Dotplot | in-app compute (not an endpoint/job) | ⬜ (out of scope — UI analytics) | ⬜ |

## Summary
- **Fully converted (capability + UI delegates):** Protein Design.
- **Capability exists, UI not delegating:** all deployed endpoints + every seeded
  job (genomics ×4, AlphaFold, Enzyme/Molecule Opt, ESM2/KERMT fine-tune, KERMT
  deploy) + ADMET (chain). These are usable in Vortex/MCP already; the UI just
  hasn't been pointed at the core.
- **Chain built, UI not delegating (Phase B):** Protein Binder Design, Ligand Binder
  Design, Motif Scaffolding — all 5 chains now live in `executor._CHAINS` + the
  `prebuilt_workflows` registry, usable in Vortex/MCP. UI services still bespoke.
- **Not a capability yet:** Single Cell Processing (job, not registered).
- **Out of scope:** single-cell DE/enrichment/trajectory/dotplot + AI narratives —
  in-app analytics/LLM, not endpoint/job capabilities.

## Known limitations / future items
- **Batch-job output now flows into the canvas (DONE for all 10 jobs).** Each job
  has an output-collecting adapter in `executor._JOB_RUNNERS`: the orchestrator
  pre-creates a per-node child MLflow run, dispatches the job into it (run-id key
  `mlflow_run_id`, except AlphaFold = `run_id`), waits on the JOB-RUN state, then
  reads the job's output back — artifact (`top_k.json`, enzyme `results/topK_pdbs`),
  Volume PDB (AlphaFold `…/results/{run}/{run}/ranked_0.pdb`), run tag
  (`output_table`, `pathogenic_table`, `result_location`, `ft_id`), run param
  (`output_vcf`), or derived Delta table (gwas `gwas_results_{run}`). So a node wired
  downstream of a batch job receives the real value. Input parity to the UI is also
  complete (node params + MCP seed). kermt_deploy is trigger-and-wait (no MLflow
  handoff) and returns the endpoint/model name.
- **No parallel branches.** The orchestrator runs nodes in a single topological pass;
  independent branches execute sequentially, not concurrently.
- **Retry quirk (parked):** DAB drops `max_retries: 0` from the orchestrator job, so
  the deployed task has no retry cap and can auto-retry once on failure. Fix TBD.

## How to migrate one (the repeatable recipe)
1. **Register the capability** — add an endpoint contract (`capabilities._ENDPOINT_CONTRACTS`),
   a `prebuilt_workflows` row (job), or a new chain in `executor._CHAINS` (+ `RUNNABLE_CHAINS`).
2. **For chains:** implement the orchestration in `executor.py` with the `progress(pct, msg)` callback.
3. **Refactor the UI service** to call `execute_capability`/`run_chain` (passing its
   `progress_callback`), keeping MLflow + Mol* viewer + SSE on top.
4. Bump the wheel version (Apps reinstall), deploy, verify the live tab.
