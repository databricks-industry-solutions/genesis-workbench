---
name: gwb-discover-resources
description: Discover what Genesis Workbench offers in a Databricks workspace from a notebook — which model serving endpoints, Vector Search indices, and batch-workflow jobs are available (and awake), plus the GWB metadata tables and the MCP server. Use at the start of any GWB life-sciences workflow to see what you can call before writing code. Triggers on "what models are available", "what can I use in Genesis Workbench", "list the GWB endpoints / indices / jobs", "is <model> deployed", "discover Genesis Workbench".
---

# Discover Genesis Workbench resources

Run this first to see what's installed and awake, then use the specific skill for the task.

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
CATALOG, SCHEMA = "main", "genesis_workbench"

# 1) Serving endpoints (models you can call synchronously — see gwb-serving-endpoints)
print("— serving endpoints —")
for e in w.serving_endpoints.list():
    if e.name.startswith("gwb_"):
        ready = getattr(getattr(e, "state", None), "ready", None)
        print(f"  {e.name:50} {ready}")

# 2) Vector Search endpoints + indices (similarity / annotation — see gwb-vector-search)
print("— vector search endpoints —")
for ep in (w.vector_search_endpoints.list_endpoints() or []):
    print(f"  {ep.name}: {getattr(getattr(ep,'endpoint_status',None),'state',None)}")
for idx in ("sequence_embedding_index", "gene_sequence_embedding_index",
            "scimilarity_cell_index", "teddy_cell_index"):
    try:
        s = w.vector_search_indexes.get_index(f"{CATALOG}.{SCHEMA}.{idx}").status
        print(f"  {idx}: ready={s.ready} rows={s.indexed_row_count}")
    except Exception as ex:
        print(f"  {idx}: not found ({ex})")

# 3) Batch-workflow jobs (long/heavy — see gwb-batch-jobs)
print("— jobs —")
for j in w.jobs.list():
    n = j.settings.name
    if n and (n.startswith("register_") or "gwb" in n or n in (
            "sequence_search_workflow","parabricks_alignment","vcf_ingestion_glow","gwas_glow_analysis")):
        print(f"  {j.job_id}  {n}")
```

## Also available
- **Metadata tables** — `{CATALOG}.{SCHEMA}.model_deployments` and `.settings` record what GWB has
  registered/deployed and the job IDs its app dispatches. Query with Spark/SQL to see versions + wiring.
- **MCP server** — the `mcp-genesis-workbench` app exposes GWB capabilities as tools for AI agents (an
  alternative to calling endpoints directly from code).

## Notes
- **Reaping:** GWB garbage-collects idle **serving endpoints, VS endpoints, and apps** (jobs, models,
  tables, and Volumes survive). If a model/index shows missing or asleep, (re)deploy or start it before
  use; a fresh `update.sh` / endpoint start brings them back.
- Endpoint names carry an install-specific prefix (`gwb_<prefix>_<slug>_endpoint`) — match by the model
  **slug**, never the full literal name.
