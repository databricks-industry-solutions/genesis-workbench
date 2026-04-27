---
name: genesis-workbench-reconstruction
description: Recover a Genesis Workbench deployment when workspace sweepers or accidental deletes have removed SQL warehouse, serving endpoints, or other non-bundle-managed resources. Diagnoses what survived, rebuilds endpoints from existing UC models without re-registration, mass-tags resources with RemoveAfter, and hardens heartbeat against future sweeps. Use when the app 500s with `ResourceDoesNotExist`, when serving endpoints have vanished, or when the user says "the app is wonky", "warehouse is gone", "endpoints disappeared", "rebuild after sweep".
---

# Genesis Workbench Reconstruction Skill

Diagnose and rebuild a partially-swept GWB deployment without redoing model registration or re-running expensive async jobs.

## When to use

- App returns `databricks.sdk.errors.platform.ResourceDoesNotExist: SQL warehouse <id> has been deleted` on init
- `databricks serving-endpoints list` shows fewer endpoints than the GWB `model_deployments` table claims are active
- Resources have vanished from a workspace despite tags / despite working last week
- User says "the app is wonky", "the deploy got swept", "endpoints disappeared", "warehouse is gone"
- Before re-deploying anything, confirm whether you need module-level redeploy or just resource recreation

Do NOT use for: initial deployment setup (`genesis-workbench-deploy-wizard`), live deploy monitoring (`genesis-workbench-deploy-monitor`), or static error recipes (`genesis-workbench-troubleshooting`).

## Related skills (when to switch)

- **From-zero deploy on a new workspace** → `genesis-workbench-deploy-wizard` (or `SKILL_GENESIS_WORKBENCH_INSTALLATION.md`). That walks pre-flight (DCS toggle, secret scope, catalog create, Docker creds) → `./deploy.sh core <cloud>` → per-module deploys. This skill assumes core+modules already deployed at least once.
- **A specific deploy step is failing right now** → `genesis-workbench-troubleshooting` for known error recipes (numpy/accelerate, scGPT dtype, SCimilarity request size, etc.).
- **Watching a fresh deploy** → `genesis-workbench-deploy-monitor`.
- **You think the workspace got swept and resources are missing** → stay here.

## The two-axis sweep model

Genesis Workbench resources sit in three tiers, with different sweep behaviour:

| Tier | Resources | Sweep risk | Bundle-managed? |
|---|---|---|---|
| **A. UC catalog data** | Catalog, schema, volumes, registered UC models | Low (but possible) | No (created at init time) |
| **B. Bundle-managed infra** | App, jobs, job clusters | Medium (managed by terraform; redeploy fixes drift) | Yes |
| **C. Lifecycle-bound resources** | SQL warehouse, serving endpoints, standalone clusters | **HIGH** — most common sweep targets | No (created post-deploy) |

When a sweep happens, **C is hit first**. UC models (A) usually survive. The bundle-managed jobs/app (B) survive but their state may diverge from terraform.

## Pre-flight: damage assessment (read-only)

Run this BEFORE proposing any fix. The choices in the recovery plan depend on what survived.

```bash
PROFILE=DEFAULT
CATALOG=mmt_gwb
SCHEMA=genesis_workbench
APP_NAME=gwb-mmt-demo

# 1. Catalog / schema / app survival
databricks catalogs get $CATALOG --profile $PROFILE --output json | jq '.name, .owner'
databricks apps get $APP_NAME --profile $PROFILE --output json | jq '.app_status, .url'

# 2. UC models — these are what we need to keep
databricks api get "/api/2.1/unity-catalog/models?catalog_name=$CATALOG&schema_name=$SCHEMA&max_results=50" \
  --profile $PROFILE | jq '.registered_models[] | .name'

# 3. Serving endpoints — what's currently alive
databricks serving-endpoints list --profile $PROFILE --output json | \
  jq '[.[] | select(.name | contains("gwb"))] | length, [.[] | select(.name | contains("gwb")) | .name]'

# 4. GWB's view of what SHOULD be deployed (from app's tracking table)
WH_ID=$(grep '^sql_warehouse_id=' /path/to/genesis-workbench/application.env | cut -d= -f2)
databricks api post /api/2.0/sql/statements --profile $PROFILE --json '{
  "warehouse_id": "'"$WH_ID"'",
  "statement": "SELECT model_endpoint_name, deploy_model_uc_name, is_active FROM '"$CATALOG.$SCHEMA"'.model_deployments WHERE is_active = true",
  "wait_timeout": "20s"
}' | jq '.result.data_array'
```

Compare endpoints alive (#3) vs endpoints GWB thinks are deployed (#4). The diff = the recovery scope.

## Pattern 1: SQL warehouse swept

**Symptom:** App 500s with `SQL warehouse <id> has been deleted`.

**Recovery:**

```bash
# Create new warehouse with same spec — note --tags is NOT a CLI flag, must use --json
WAREHOUSE_JSON=$(databricks warehouses create --profile $PROFILE --output json --json '{
  "name": "mmt_gwb_warehouse",
  "cluster_size": "2X-Small",
  "warehouse_type": "PRO",
  "enable_serverless_compute": true,
  "auto_stop_mins": 10,
  "min_num_clusters": 1,
  "max_num_clusters": 1,
  "tags": {"custom_tags": [
    {"key": "RemoveAfter", "value": "2027-12-31"},
    {"key": "application", "value": "genesis_workbench"},
    {"key": "created_by", "value": "may.merkletan"}
  ]}
}')
NEW_ID=$(echo "$WAREHOUSE_JSON" | jq -r '.id')

# Update application.env with new ID
sed -i.preedit "s/^sql_warehouse_id=.*/sql_warehouse_id=$NEW_ID/" application.env

# Bundle deploy core — this updates the app's terraform-managed sql_warehouse
# resource binding via ${var.sql_warehouse_id} → ${var.app_name}.resources.sql_warehouse.id
# App auto-restarts; .deployed marker prevents init job re-run.
./deploy.sh core aws
```

**Critical CLI gotchas:**
- `databricks warehouses create --tags` does NOT exist. Tags only via `--json`.
- `min_num_clusters` and `max_num_clusters` are required (>=1).
- Old warehouse ID stays in `mmt_gwb.genesis_workbench.settings` until next bundle deploy of core (which re-runs `initialize_core_job` IF `.deployed` is missing — it normally isn't, so settings table just keeps reading from env var `SQL_WAREHOUSE` injected via app resource binding).

**Verify:**
```bash
# App should restart and log group fetch — that proves db_connect() succeeded
databricks apps logs $APP_NAME --profile $PROFILE | grep -E 'Initializing|User belongs to following groups'
```

## Pattern 2: Serving endpoints swept (UC models survived)

**Symptom:** App loads but module pages fail when invoking models. `serving-endpoints list` shows 1 of N expected endpoints.

**Recovery — DO NOT run `./deploy.sh <module>`** unless UC models are also gone. Module redeploy re-runs registration (15-30 min per module) and can create v2 of UC models for no reason. Instead, use GWB's internal `deploy_model_job`:

```bash
# 1. Get the missing endpoints from GWB's tracking table
SQL="SELECT m.model_id, m.model_name, m.model_uc_name, d.deployment_name,
            d.deployment_description, d.model_endpoint_name
     FROM mmt_gwb.genesis_workbench.models m
     JOIN mmt_gwb.genesis_workbench.model_deployments d ON d.model_id = m.model_id
     WHERE d.is_active = true"
# (Run via api post /api/2.0/sql/statements)

# 2. For each missing endpoint, run deploy_model_job
# Per-endpoint workload params (record from memory or from prior deploy logs):
#   scimilarity_search_nearest: workload_type=CPU, workload_size=Large
#   scimilarity_get_embedding:  workload_type=GPU_SMALL, workload_size=Small
#   scimilarity_gene_order:     workload_type=CPU, workload_size=Small
#   default for others:         workload_type=GPU_SMALL, workload_size=Small

for_each_missing_endpoint() {
  databricks bundle run deploy_model_job --target prod_aws --profile $PROFILE \
    --params "gwb_model_id=$MODEL_ID,deployment_name=$NAME,deployment_description=$DESC,workload_type=$WTYPE,workload_size=$WSIZE,deploy_user=$USER_EMAIL"
}
```

**Verify each endpoint reaches READY:** `databricks serving-endpoints get <name>` → `state.ready == "READY"`.

## Pattern 3: Mass-tag everything with RemoveAfter

After a sweep, the only durable protection is tags the workspace sweeper actually respects. RemoveAfter on a single resource (the warehouse) is NOT enough — the sweeper checks tags on each resource individually.

**Two-step process:**

```bash
# A. Bake into bundle so future deploys auto-tag (modifies all variables.yml)
python3 << 'PYEOF'
import re, os
for root, _, files in os.walk('modules'):
    if '.databricks' in root or '.bundle' in root: continue
    if 'variables.yml' not in files: continue
    p = os.path.join(root, 'variables.yml')
    text = open(p).read()
    if 'RemoveAfter' in text: continue
    pat = re.compile(
        r'(common_resource_tags:.*?\n\s*default:\s*\n(?:\s*\w+:\s*[^\n]+\n)+?(\s*)created_by:\s*\$\{[^}]+\}\s*\n)',
        re.DOTALL)
    m = pat.search(text)
    if not m: continue
    indent = m.group(2)
    new = text[:m.end()] + f'{indent}RemoveAfter: "2027-12-31"\n' + text[m.end():]
    open(p, 'w').write(new)
    print('OK:', p)
PYEOF

# B. Tag existing resources via API (immediate; bundle deploy from edited variables.yml
# will keep these in sync going forward)
# Resources to tag:
#   - All UC registered models (ALTER MODEL ... SET TAGS)
#   - Catalog, schema, volumes (ALTER CATALOG/SCHEMA/VOLUME ... SET TAGS)
#   - All GWB jobs (jobs/update with new_settings.tags)
#   - Standalone clusters (parabricks_cluster) (clusters/edit with custom_tags)
#   - Surviving serving endpoints (serving-endpoints/<name>/tags PATCH with add_tags)
# Skip:
#   - Secret scopes (not taggable in Databricks)
#   - Apps (custom_tags not exposed via API; rely on description for tag-protection)
```

The full mass-tagger script lives at `scripts/mass_tag_remove_after.py` (TODO if not yet present — author from this skill).

## Pattern 4: Heartbeat hardening for warehouse sweep protection

The default `scripts/heartbeat_app.py` only PATCHes the app description. That keeps the APP alive (the FE workspace's auto-stop is based on app `update_time`) but does NOT keep the warehouse alive. The warehouse gets swept based on its own activity timer.

**Patch:** add a Statement Execution `SELECT 1` against the warehouse BEFORE the app PATCH. This refreshes the warehouse's `last_query_time`.

```python
# In scripts/heartbeat_app.py, before the apps.update PATCH:

# Find current warehouse by name (resilient to ID changes after sweep+recreate)
warehouses = w.warehouses.list()
target = next((wh for wh in warehouses if wh.name == "mmt_gwb_warehouse"), None)
if target:
    # Statement Execution API hits the warehouse and updates last_query_time
    result = w.statement_execution.execute_statement(
        warehouse_id=target.id,
        statement="SELECT 1",
        wait_timeout="10s",
    )
    print(f"Warehouse ping: {result.status.state} (warehouse {target.id})")
else:
    print("WARN: mmt_gwb_warehouse not found — sweeper may have struck again")
```

Trade-off: if every GWB job is desired to refresh the warehouse, the cleaner pattern is for each job's *natural* work to query the warehouse (which happens organically for `deploy_model_job`, `initialize_core_job`, etc. — they read settings table). Only `gwb_mmt_demo_heartbeat` lacks a natural warehouse touch and needs explicit treatment.

## Anti-patterns (avoid)

| Tempting shortcut | Why it's wrong |
|---|---|
| `./deploy.sh single_cell aws` to "fix endpoints" | Re-runs registration → 15-30 min wait → creates v2 of UC models → no benefit when models already exist |
| `databricks apps update --resources` to rebind warehouse | Creates drift between app config and bundle terraform state — next bundle deploy reverts |
| Tagging only the warehouse with RemoveAfter | Sweeper may still take other resources; tag everything |
| Editing terraform state directly | Use bundle deploy for canonical state updates |
| `set -e` + bash subshell capture without `if` guard | Failed warehouse-create silently produces empty `$NEW_ID` → corrupts application.env via sed. Always validate captured IDs before passing to sed/edits |
| Heredoc `<<PYEOF` with `${...}` shell expansion | Use `<<'PYEOF'` (quoted) when Python string contains shell variable syntax |

## Pre-deploy verification checklist

Before declaring "reconstruction complete":

- [ ] `databricks apps logs gwb-mmt-demo` shows successful init (User belongs to following groups...)
- [ ] All endpoints from `model_deployments` table have matching live serving endpoints
- [ ] Each endpoint's `state.ready == READY`
- [ ] Surviving warehouse + all jobs + UC models + parabricks_cluster have `RemoveAfter=2027-12-31` tag
- [ ] Heartbeat job tested manually + unpaused
- [ ] application.env baseline backed up to `.preedit` so emergency rollback is one `mv` away

## Future-proofing

After successful reconstruction, consider:

1. Add a `warehouse_ping` SQL last task to bundle-managed jobs that don't naturally query the warehouse (heartbeat especially)
2. Move from `srijitnair254/parabricks_dbx_amd64:0.1` (Srijit's docker namespace) to your own controlled namespace so docker access doesn't depend on someone else's tokens
3. Bake `RemoveAfter` into a top-level shared variables include (currently duplicated across 21 files) — single source of truth
4. Add a one-line check at start of each module `deploy.sh` that fails loudly if `RemoveAfter` is missing from variables.yml
5. Document workspace sweeper cadence/criteria in `docs/deployments/<workspace>/SESSION-NOTES.md` once observed
