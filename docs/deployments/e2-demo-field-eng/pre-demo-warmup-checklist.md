# Pre-demo warm-up checklist — GWB live demo

**Demo URL:** https://gwb-mmt-demo-1444828305810485.aws.databricksapps.com
**Workspace:** e2-demo-field-eng (profile `DEFAULT`)
**Warehouse:** `mmt_gwb_warehouse` (id `9b5370ee2ef1e248`)

**Use:** fillable checklist before each live demo. Tick items as you go. Total budget: ~30 min. Minimum: 15 min if everything was warm <1 hr ago.

---

## T-60 min — long-running prerequisites

These take long enough that they need to be *done*, not started, by the warm-up window.

- [ ] Verify Vector Search index sync is complete (1M-row sync = 30-60 min when newly created)
  ```bash
  databricks api get /api/2.0/vector-search/indexes/mmt_gwb.genesis_workbench.sequence_embedding_index --profile DEFAULT \
    | python3 -c "import sys, json; r = json.load(sys.stdin); print(r.get('status'))"
  ```
  Expect: `ready: True`, `detailed_state: ONLINE_NO_PENDING_UPDATE`

- [ ] Pre-run any long-running workflows that the demo will reference (results shown via "past runs" / "deployed models" tab):
  - Scanpy processing → Cell Type Annotation chain
  - GWAS Manhattan plot (if the existing screenshot in `gwb_app_imgs/` doesn't suffice)
  - Any specific BioNeMo fine-tune output you want to show
- [ ] Confirm slide deck is uploaded / shareable (Google Slides exported from `slide-deck-mvp.md`)
- [ ] `presenter-cheat-sheet.md` open on second screen / printed

---

## T-30 min — wake all serving infrastructure

- [ ] **Trigger "Start All Endpoints" job** — wakes every serving endpoint at once
  ```bash
  databricks jobs run-now 129075315342116 --profile DEFAULT
  ```
  *(job_id `129075315342116` = `dbx_gwb_start_all_endpoints` from `mmt_gwb.genesis_workbench.settings`)*

- [ ] **Wake the warehouse**
  ```bash
  databricks api post /api/2.0/sql/statements --profile DEFAULT --json \
    '{"warehouse_id":"9b5370ee2ef1e248","statement":"SELECT 1","wait_timeout":"10s"}'
  ```

- [ ] **Optional: trigger heartbeat manually** to bump app `update_time` + ping VS index/endpoint
  ```bash
  databricks jobs run-now 499481011575662 --profile DEFAULT
  ```
  *(buys ~3 days against FE auto-stop policy + refreshes VS activity)*

- [ ] Verify endpoint state — should see READY count climbing
  ```bash
  databricks serving-endpoints list --profile DEFAULT --output json | \
    python3 -c "
  import sys, json
  gwb = [e for e in json.load(sys.stdin) if 'gwb_mmt' in e.get('name','')]
  ready = sum(1 for e in gwb if e.get('state',{}).get('ready') == 'READY')
  print(f'READY: {ready}/{len(gwb)}')"
  ```
  Expect: 19/19 READY. If <19, wait 60-90s and re-check.

---

## T-20 min — Parabricks (only if demoing live)

⚠️ **If you're NOT live-demoing Parabricks, skip this section** — show pre-run screenshots instead. Parabricks' value prop is "4-6× faster than CPU"; you can communicate that without watching it run.

- [ ] Find and start the parabricks_cluster
  ```bash
  PB_CLUSTER_ID=$(databricks api get "/api/2.0/clusters/list?filter_by.cluster_states=TERMINATED" --profile DEFAULT \
    | python3 -c "import sys, json; r = json.load(sys.stdin); c = next(x for x in r.get('clusters',[]) if x.get('cluster_name')=='parabricks_cluster'); print(c['cluster_id'])")
  databricks api post /api/2.0/clusters/start --profile DEFAULT --json "{\"cluster_id\":\"$PB_CLUSTER_ID\"}"
  ```
- [ ] Cluster start polls — wait until RUNNING (10-15 min cold)

---

## T-10 min — UI + browser sanity

- [ ] Open https://gwb-mmt-demo-1444828305810485.aws.databricksapps.com in browser
- [ ] Click each tab once to make sure UI renders (no 500s, no broken images):
  - [ ] Home
  - [ ] Settings (verify SQL Warehouse ID = `9b5370ee2ef1e248`, all `<module>_deployed=true`)
  - [ ] Protein Studies (Structure Prediction / Protein Design / Inverse Folding / Sequence Search subtabs)
  - [ ] Single Cell (Cell Type Annotation / Cell Similarity / Perturbation / Processing)
  - [ ] Small Molecule (ADMET / Ligand Binder / Binder Design / Motif Scaffolding)
  - [ ] Disease Biology (VCF Ingestion / Variant Annotation / GWAS)
  - [ ] BioNeMo (ESM2 Finetune + ESM2 Inference)
  - [ ] Parabricks (Germline)
- [ ] If any tab errors → 10 min to triage; otherwise move on

---

## T-5 min — final smoke tests

- [ ] **ESMFold smoke test** (5-second flagship demo):
  - Protein Studies → Structure Prediction
  - Paste: `MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK`
  - Select ESMFold → Run
  - Confirm 3D structure renders in Mol* viewer
- [ ] **Sequence Search smoke test** (proves VS index is queryable):
  - Protein Studies → Sequence Search
  - Paste a known protein sequence → Search
  - Confirm results return (5-15 sec)
- [ ] **Chemprop ADMET smoke test** (NEW today):
  - Small Molecule → ADMET & Safety
  - Paste aspirin SMILES: `CC(=O)Oc1ccccc1C(=O)O`
  - Click Run → confirm BBBP / ClinTox / ADMET cards return
- [ ] Pull up Home tab in clean browser window — Claude AI assistant box ready for opener

---

## T-1 min — final state check

- [ ] App is showing "no errors" on Home tab
- [ ] At least 19/19 endpoints READY (or whatever count is canonical for the demo)
- [ ] Cheat sheet visible on second screen
- [ ] Slide deck open on share screen
- [ ] Live demo browser window separate from share screen, not yet shared

---

## Demo execution — fallback plan

If a live demo step fails mid-presentation:
- **Endpoint cold (60-90s spinner):** narrate the warm-up — "this is the platform allocating GPU capacity from zero. In production you'd warm them with the Start All Endpoints job before a meeting."
- **Whole tab errored:** pivot to screenshots from `gwb_app_imgs/` (have folder open in another window)
- **Vector Search returned empty:** fall back to "the index is still indexing the embeddings — let me show you a pre-run version" + screenshot
- **Parabricks job hangs:** never run live; show pre-run output

If a tab is fundamentally broken pre-demo: skip that part of the flow, don't try to hot-fix in front of the audience.

---

## Post-demo (minor housekeeping)

- [ ] Capture any new screenshots taken during demo into `gwb_app_imgs/` with `<Module>_<Workflow>_<Step>.png` naming
- [ ] Note any issues in `testing-log-2026-04-26.md` (or new dated log)
- [ ] If audience asked questions you didn't expect, add to `presenter-cheat-sheet.md` Q&A section for next time
- [ ] Stop the parabricks_cluster (saves ~$ idle)
  ```bash
  databricks api post /api/2.0/clusters/delete --profile DEFAULT --json "{\"cluster_id\":\"$PB_CLUSTER_ID\"}"
  ```
  (Note: this terminates, not deletes — name+config persist for next start)

---

## Reference — cold-start times + when to wake

| Resource | Cold-start time | When to wake | Verify signal |
|---|---|---|---|
| **App `gwb-mmt-demo`** | already RUNNING | n/a | `apps get` → `app_status.state == RUNNING` |
| **SQL warehouse** `mmt_gwb_warehouse` (2X-Small serverless, auto_stop=10 min) | 10–20 sec | **T-2 min** — hit with `SELECT 1` | statement returns SUCCEEDED |
| **Serving endpoints** (19 total, scale-to-zero) | 60–90 sec each | **T-15 min** — trigger Start All Endpoints job (`129075315342116`) | `state.ready == READY` for all |
| **Vector Search index** `sequence_embedding_index` | 30–60 min for initial 1M-row sync (post-create only) | **T-60 min** if newly created — verify `ready=true` before demo | `status.ready == true`, `detailed_state == ONLINE_NO_PENDING_UPDATE` |
| **Parabricks cluster** (currently TERMINATED) | 10–15 min cold start | **T-20 min** IF demoing live (otherwise SKIP — use pre-run screenshots) | `state == RUNNING` |
| **BioNeMo job cluster** (on-demand) | 10–15 min cold | **DON'T demo live** — show pre-run results | task `RUNNING` |
| **Long-running workflows** (Scanpy processing, GWAS, BioNeMo finetune) | 10–60 min each | **T-60+ min** pre-run BEFORE warm-up window; show results via Past Runs | output table populated |
| **App auto-stop** | 3 days idle (FE workspace policy) | heartbeat M/W/F bumps `update_time` | description carries `heartbeat: <recent-utc>` |
| **App auto-delete** | 7 days idle (FE workspace policy) | same heartbeat | same |

---

## Reference — known-canonical resource IDs (mmt_gwb on e2-demo-field-eng, as of 2026-04-26 reconstruction)

```
warehouse_id                    9b5370ee2ef1e248
app_name                        gwb-mmt-demo
heartbeat_job_id                499481011575662
start_all_endpoints_job_id      129075315342116
deploy_model_job_id             508511447468379
catalog                         mmt_gwb
schema                          mmt_gwb.genesis_workbench
vs_endpoint                     gwb_sequence_search_vs_endpoint
vs_index                        mmt_gwb.genesis_workbench.sequence_embedding_index
```

---

## Iteration

- v0 (2026-04-27 morning): initial draft for SA HUNTER demo. Refine after first run with notes on what was over-prepped vs. under-prepped.
