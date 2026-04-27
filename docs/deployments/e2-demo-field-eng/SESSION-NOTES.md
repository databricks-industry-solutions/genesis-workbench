# GWB Deploy — e2-demo-field-eng — Session Notes

Third GWB deployment target after fevm-mmt-aws-usw2 (sandbox) and fe-vm-hls-amer.
Chosen because Container Services is enabled here — unblocks Parabricks + BioNeMo
modules that the fevm-mmt-aws-usw2 sandbox can't run.

## Context

- **Workspace:** https://e2-demo-field-eng.cloud.databricks.com (o=1444828305810485)
- **Region:** AWS (us-west-2 metastore `unity-catalog-demo`)
- **Shared field-eng workspace** — 299 apps, 4383 catalogs, many concurrent users
- **Local branch:** started on `mmt/ver_pin_sandbox_setup`; new branch planned for e2fe-specific env files

## Probe results (2026-04-21)

All prerequisites verified before committing to a full deploy:

| Check | Result |
|---|---|
| Databricks CLI profile `DEFAULT` | ✓ OAuth valid after `databricks auth login --profile DEFAULT` |
| May in `admins` workspace group | ✓ |
| Container Services (`enableDcs`) | ✓ `true` |
| GPU instance types available | ✓ g4dn.* and g5.* (xlarge through 48xlarge) |
| Cluster-create w/ docker image | ✓ `srijitnair254/parabricks_dbx_amd64:0.1` on g5.xlarge reached RUNNING at t+401s (~6:40) |
| Catalog-create permission | ✓ created + deleted `mmt_gwb_permcheck` |
| App name `genesis-workbench` free | ✓ no collision among 299 apps |
| LLM endpoint `databricks-claude-sonnet-4-6` | ✓ READY |

## Resources provisioned (2026-04-21)

| Resource | Name / ID | State | Notes |
|---|---|---|---|
| Catalog | `mmt_gwb` | active | Created by May. `isolation_mode=OPEN`, `MANAGED_CATALOG`. Comment: "Genesis Workbench for FE demos \| RemoveAfter: 2027-12-31". |
| Secret scope | `mmt` | active | Pre-existing (had 6 non-GWB keys). Added 4 GWB docker keys: `gwb_parabricks_docker_{user,token}`, `gwb_bionemo_docker_{user,token}`. All 4 use Srijit's Docker Hub (`srijitnair254`). |
| SQL warehouse | `mmt_gwb_warehouse` id=`1a9a54ac772b8798` | RUNNING | 2X-Small, serverless PRO, auto_stop=10 min. Tagged `RemoveAfter=2027-12-31`, `purpose=mmt_gwb_deploy`. |
| Probe app | `gwb-mmt-probe` | compute STOPPED | URL: https://gwb-mmt-probe-1444828305810485.aws.databricksapps.com. No code deployed. Description contains `RemoveAfter: 2027-12-31`. Keep until new app deploy lands (app-quota safety). |

## Planned deploy config (not yet applied)

Once branch switch + env file edits are done:

**`application.env` (planned):**
```
workspace_url=https://e2-demo-field-eng.cloud.databricks.com
databricks_profile=DEFAULT
core_catalog_name=mmt_gwb
core_schema_name=genesis_workbench
sql_warehouse_id=1a9a54ac772b8798
```

**`modules/core/module.env` (planned):**
```
dev_user_prefix=mmt
app_name=gwb-mmt-demo             # matches May's `gwb-mmt-*` pattern on other workspaces
secret_scope_name=mmt
llm_endpoint_name=databricks-claude-sonnet-4-6
```

**Module docker refs (already use secret scope, no change needed):**
- `modules/parabricks/module.env` → `{{secrets/mmt/gwb_parabricks_docker_{user,token}}}`
- `modules/bionemo/module.env` → `{{secrets/mmt/gwb_bionemo_docker_{user,token}}}`
- `modules/disease_biology/module.env` → reuses parabricks refs (same image)

## Decisions + rationale

- **Own SQL warehouse (not shared):** the shared `dbdemos-shared-endpoint` is X-Large and RUNNING but has 299-app workload concurrency. Own 2X-Small is $0 when auto-stopped, predictable latency. GWB warehouse load is CRUD-only (~40 jobs run on their own clusters, not the SQL warehouse).
- **`mmt_gwb` catalog (not `mmt_demos2`):** clean isolation; `DROP CATALOG mmt_gwb CASCADE` is an easy nuke path after SA HUNTER session if the deploy gets retired.
- **Srijit's Docker Hub PAT for both BioNeMo + Parabricks:** Docker Hub PATs are user-scoped — one PAT covers all `srijitnair254/*` images.
- **Keep `gwb-mmt-probe` until new app is up:** field-eng shared workspaces may cap apps per user; deleting the probe *before* creating the new one risks being stuck without either.

## Backup / restore path

Before swapping env files off sandbox target, backups were made (2026-04-21):

- `application.env.fevm-mmt-aws-usw2.bak`
- `modules/core/module.env.fevm-mmt-aws-usw2.bak`
- `modules/parabricks/module.env.fevm-mmt-aws-usw2.bak`
- `modules/bionemo/module.env.fevm-mmt-aws-usw2.bak`
- `modules/disease_biology/module.env.fevm-mmt-aws-usw2.bak`

To restore sandbox config: `cp <file>.fevm-mmt-aws-usw2.bak <file>`. (Note: `application.env` + `module.env` files are gitignored, so git branch switching does NOT swap their contents — `.bak` files are the only restore path.)

### Bundle state backup

First `./deploy.sh core aws` on e2-demo-field-eng failed on `resources.dashboards.genesis_workbench` — stale dashboard ID from the sandbox deploy was in `.databricks/bundle/prod_aws/terraform/terraform.tfstate`. DAB reuses the same `prod_aws` target name across workspaces, so local state from the prior deploy was being applied to the new workspace.

Fix: renamed all 18 `.databricks/bundle/prod_aws/` dirs to `prod_aws.fevm-mmt-aws-usw2/` and tarballed for durable backup:

```
docs/deployments/fevm-mmt-aws-usw2/bundle-state-backup-20260421-1724.tar.gz   (116K, 147 state files)
```

(Excludes `.terraform/` provider cache which re-downloads on deploy.)

To restore sandbox bundle state (e.g., to re-run on fevm-mmt-aws-usw2):
```bash
tar xzf docs/deployments/fevm-mmt-aws-usw2/bundle-state-backup-20260421-1724.tar.gz
# then: for each prod_aws.fevm-mmt-aws-usw2 dir, rename back to prod_aws
find modules -name "prod_aws.fevm-mmt-aws-usw2" -type d | while read d; do mv "$d" "${d%.fevm-mmt-aws-usw2}"; done
```

## Deploy runbook — end-to-end

The actual command sequence used for this deploy, in order. Time estimates are wall-clock for this workspace (e2-demo-field-eng, AWS us-west-2); your mileage on others may vary.

### 1. Pre-flight — workspace resources (one-time, ~2 min)

```bash
# Auth (browser OAuth flow)
databricks auth login --profile DEFAULT

# Catalog
databricks catalogs create mmt_gwb \
  --comment "Genesis Workbench for FE demos | RemoveAfter: 2027-12-31" \
  --profile DEFAULT

# Docker creds → secret scope (reused existing `mmt` scope)
databricks secrets put-secret mmt gwb_parabricks_docker_user  --string-value "srijitnair254"                        --profile DEFAULT
databricks secrets put-secret mmt gwb_parabricks_docker_token --string-value "<Srijit Docker Hub PAT>"              --profile DEFAULT
databricks secrets put-secret mmt gwb_bionemo_docker_user     --string-value "srijitnair254"                        --profile DEFAULT
databricks secrets put-secret mmt gwb_bionemo_docker_token    --string-value "<same PAT — Docker Hub user-scoped>"  --profile DEFAULT

# SQL warehouse (own, 2X-Small serverless, auto-stop 10 min)
cat > /tmp/wh.json <<'EOF'
{
  "name": "mmt_gwb_warehouse",
  "cluster_size": "2X-Small",
  "auto_stop_mins": 10,
  "enable_serverless_compute": true,
  "warehouse_type": "PRO",
  "min_num_clusters": 1,
  "max_num_clusters": 1,
  "tags": {"custom_tags": [{"key":"RemoveAfter","value":"2027-12-31"},{"key":"purpose","value":"mmt_gwb_deploy"}]}
}
EOF
databricks warehouses create --json @/tmp/wh.json --profile DEFAULT
```

### 2. State hygiene before switching workspaces (CRITICAL — ~30 sec)

`.databricks/bundle/prod_aws/` and `modules/*/.deployed` markers are local state from prior deploys. When pointing the same repo at a new workspace, **both must be moved aside** or the deploy will (a) try to reconcile against stale resource IDs, and (b) skip the `initialize_core_job` that creates the `settings` table.

```bash
# Backup + clear per-module bundle state
for d in $(find modules -name "prod_aws" -type d -path "*/.databricks/bundle/*"); do
  mv "$d" "${d}.<prior-workspace-suffix>"
done

# Backup + clear .deployed markers
for f in $(find modules -name ".deployed" -not -path "*/node_modules/*"); do
  mv "$f" "${f}.<prior-workspace-suffix>"
done

# Optional: durable tarball (since .databricks/ is gitignored)
tar czf docs/deployments/<prior-workspace>/bundle-state-backup-$(date +%Y%m%d-%H%M).tar.gz \
  --exclude='.terraform' --exclude='.terraform.lock.hcl' --exclude='plan' \
  $(find modules -name "prod_aws.<prior-workspace-suffix>" -type d)
```

### 3. Env files (~5 min, local edits)

```bash
# Backup existing (sandbox) env files before overwriting
cp application.env                     application.env.<prior>.bak
cp modules/core/module.env            modules/core/module.env.<prior>.bak
# (optional) cp modules/parabricks/module.env + modules/bionemo/module.env + modules/disease_biology/module.env

# Write new application.env pointing at target workspace
cat > application.env <<'EOF'
workspace_url=https://e2-demo-field-eng.cloud.databricks.com
databricks_profile=DEFAULT
core_catalog_name=mmt_gwb
core_schema_name=genesis_workbench
sql_warehouse_id=<your-warehouse-id>
EOF

# Update modules/core/module.env (mostly: app_name unique per workspace)
sed -i '' 's/app_name=.*/app_name=gwb-mmt-demo/' modules/core/module.env
```

Module env files for parabricks / bionemo / disease_biology **don't change** if you use the same secret scope name + key naming convention (`{{secrets/<scope>/gwb_<module>_docker_{user,token}}}`).

### 4. Core deploy (~7-10 min — bundle + init_core + app start)

```bash
./deploy.sh core aws
```

Expected phases:
1. Secret scope check → already exists
2. Library build (Poetry)
3. Schema create (`mmt_gwb.genesis_workbench`)
4. Bundle validate + deploy
5. **`initialize_core_job`** — creates 6 tables: `settings`, `batch_models`, `app_permissions`, `model_deployments`, `models`, `user_settings`
6. App `gwb-mmt-demo` start (3-5 min cold start)
7. Grant app SP on catalog + schema
8. `initialize_module_job` — MERGE `core_deployed=true` into settings
9. `.deployed` marker written

### 5. Module deploys — one at a time, ~2-5 min each

Per CLAUDE.md: serialize to surface quota/docker issues one at a time.

```bash
./deploy.sh parabricks aws       # docker-dependent; ~2-3 min
./deploy.sh bionemo aws          # docker-dependent; ~2-5 min (if weight-download job fires)
./deploy.sh protein_studies aws  # no docker, several sub-models (ESMFold/Boltz/AlphaFold/etc.)
./deploy.sh single_cell aws
./deploy.sh small_molecule aws
./deploy.sh disease_biology aws  # reuses parabricks docker image for GWAS
```

Between each, verify via:
```bash
databricks api post "/api/2.0/sql/statements" --profile DEFAULT --json '{
  "warehouse_id":"<id>",
  "statement":"SELECT module, COUNT(*) FROM <catalog>.<schema>.settings GROUP BY module ORDER BY module",
  "wait_timeout":"10s"
}'
```

### 6. Post-deploy cleanup

After `gwb-mmt-demo` is ACTIVE and smoke-tested, delete the probe app to return an app-quota slot:

```bash
databricks apps delete gwb-mmt-probe --profile DEFAULT
```

## Teardown checklist (when this deploy retires)

In order — least to most destructive:

1. `databricks apps delete gwb-mmt-demo --profile DEFAULT`
2. `databricks apps delete gwb-mmt-probe --profile DEFAULT` (if still there)
3. `databricks warehouses delete 9b5370ee2ef1e248 --profile DEFAULT` (was `1a9a54ac772b8798` before 2026-04-26 sweep+recreate)
4. `databricks catalogs delete mmt_gwb --force --profile DEFAULT`
5. Leave `mmt` secret scope in place (has non-GWB keys). Optionally `databricks secrets delete-secret mmt gwb_<*>_docker_{user,token}` for the 4 GWB-added keys.
6. `git checkout mmt/ver_pin_sandbox_setup` or restore `.bak`s to roll env files back.

---

## Reconstruction session 2026-04-26 — sweep + recovery

**Incident:** Sun afternoon the app started 500'ing with `ResourceDoesNotExist: SQL warehouse 1a9a54ac772b8798 has been deleted`. Workspace sweeper removed the warehouse + 10 of 11 serving endpoints despite the warehouse description containing `RemoveAfter: 2027-12-31`. UC models, catalog, schema, volumes, all 14 jobs, parabricks_cluster, app, secret scope all SURVIVED.

**Root cause:** Sweeper checks `custom_tags` (not description). Original warehouse + endpoints had `{application, created_by}` from the bundle's `common_resource_tags` default — no `RemoveAfter`. SCimilarity gene_order endpoint survived (the only one that did) for unclear reasons; same tag set but CPU/Small workload vs. GPU on the others.

**Recovery sequence (1.5 hours wall clock):**

1. Recreate warehouse via `databricks warehouses create --json '{...}'` — `--tags` is NOT a CLI flag (only via `--json`); also requires `min_num_clusters`/`max_num_clusters >= 1`. New ID: `9b5370ee2ef1e248`.
2. Update `application.env` `sql_warehouse_id` (use `sed -i.preedit` for auto-backup; `set -e` doesn't catch empty-capture-then-sed corruption — always validate captured ID before passing to sed).
3. `./deploy.sh core aws` — terraform reconciles app's `sql_warehouse` resource binding via `${var.sql_warehouse_id}`; app auto-restarts; `.deployed` marker correctly skips init.
4. **Bake `RemoveAfter: "2027-12-31"`** into 21 modules' `variables.yml` `common_resource_tags.default` (commit `191db70`). Future bundle deploys auto-tag terraform-managed resources.
5. **API mass-tag existing resources** with `RemoveAfter`+`application`:
   - SQL `ALTER CATALOG/SCHEMA/VOLUME ... SET TAGS` for catalog/schema/15 volumes — works.
   - Jobs API `/api/2.1/jobs/update` with `new_settings.tags` for 14 jobs — preserve existing via `{**existing, RemoveAfter, application}`.
   - Clusters API `/api/2.0/clusters/edit` for parabricks_cluster — preserve `ResourceClass: SingleNode`.
   - Serving-endpoints API `PATCH /api/2.0/serving-endpoints/<name>/tags` with `add_tags` for surviving endpoint.
   - **UC registered models gap:** `ALTER MODEL ... SET TAGS` SQL fails with `PARSE_SYNTAX_ERROR`. MLflow `set-tag` API rejects dotted UC names ("Invalid name … cannot contain periods"). UC `securable-tag-assignments` endpoint not present on this workspace's Databricks version. Fallback: `PATCH /api/2.1/unity-catalog/models/<full_name>` with `comment: "RemoveAfter:2027-12-31 | application:..."`. Catalog/schema-level tags should inherit downward for sweep purposes; revisit on newer DBR.
   - Secret scopes: not taggable (skip).
6. Re-deploy core so `deploy_model_job` and the app pick up the new tag default for future endpoint creates.
7. **Recreate 10 missing serving endpoints — DO NOT use `bundle run deploy_model_job`.** First attempt failed: 8 of 10 with `ModuleNotFoundError: No module named 'dbboltz'` (and analogous per-model). Root cause: `deploy_model.py`'s `process_model_with_adapters` imports per-model packages (`modules/protein_studies/boltz/boltz_1/dbboltz/`, etc.) at module-load time. These packages are bundled INTO the MLflow model artifacts (install on serving container via conda_env's `/model/artifacts/<package>`) but NOT installed in the deploy_model.py serverless runtime. The `./deploy.sh <module>` flow side-steps this via `%pip install ../<dbpkg>` before invoking deploy_model. Calling `deploy_model_job` out-of-context fails.
   - **Working pattern:** direct `POST /api/2.0/serving-endpoints` with served entity pointing at existing UC model. Serving container installs per-model package from `/model/artifacts` automatically. No deploy_model_job needed.
   - Per-endpoint workload sizing:
     - `scimilarity_gene_order`: CPU/Small (already alive)
     - `scimilarity_get_embedding`: GPU_SMALL/Small (notebook default was MULTIGPU_MEDIUM; downsized at memory-recommended GPU/Small)
     - `scimilarity_search_nearest`: CPU/Large (was MULTIGPU_MEDIUM, OOM'd; per memory)
     - **`esmfold`: `Medium` not `Small`** — first attempt with Small got `DEPLOYMENT_FAILED: Failed to load the model. Exit code 1`. Retry with Medium succeeded. **Update register notebook hardcoded `workload_size="Small"` to `Medium` for ESMFold.**
8. Heartbeat hardening (commit `75986d3`): `scripts/heartbeat_app.py` now does Statement Execution `SELECT 1` against `mmt_gwb_warehouse` (resolved by NAME — resilient to ID changes from sweep+recreate) BEFORE the app description PATCH. Wrapped in try/except so warehouse-ping failure doesn't block app PATCH. Schedule unpaused (had been paused since 2026-04-22).
9. Deployed previously-pending modules: `small_molecule` (chemprop, diffdock, proteina_complexa) + `disease_biology` (gwas, vcf_ingestion, variant_annotation). Bundle deploys ✓; per-model registration sub-jobs (15-30 min each) running async.

**Skill captured:** `claude_skills/SKILL_GENESIS_WORKBENCH_RECONSTRUCTION.md` (commits `75986d3` + `4e30880`) — recovery-only playbook with sweep model, four recovery patterns, and anti-patterns table.

**Still-open / nice-to-have (post-demo):**
- `./deploy.sh bionemo / parabricks / protein_studies / single_cell aws` to refresh terraform state with new RemoveAfter from variables.yml (currently API-tagged; harmless drift)
- Verify sweeper cadence: leave the new warehouse alone for ~5+ days, watch whether the new `RemoveAfter` tag actually protects it
- When Databricks gets a UC `securable-tag-assignments` endpoint, replace the comment-field fallback with proper UC tags on registered models
- Consider hoisting `common_resource_tags` to a single shared variables include — currently duplicated across 21 files
- Update `register_esmfold.py` to hardcode `workload_size="Medium"` (currently "Small" → fails)
- Push branch commits to origin (currently local-only on `mmt/e2fe_gwb_deploy`)

---

## Third sweep class found 2026-04-26 ~23:00 UTC — Vector Search

While clickthrough-testing the Sequence Search workflow, "Search failed: Unity Catalog entity `mmt_gwb.genesis_workbench.sequence_embedding_index` does not exist."

**Diagnosis:**
- `mmt_gwb.genesis_workbench.sequence_embeddings` table: ALIVE (1,000,000 rows) — the upstream embedding pipeline ran successfully at some point
- `gwb_sequence_search_vs_endpoint` Vector Search endpoint: GONE (sweeper)
- `mmt_gwb.genesis_workbench.sequence_embedding_index` Delta Sync index: GONE (depends on endpoint)

**Recovery (mirrors `04_create_vector_index.py` notebook logic):**
1. `POST /api/2.0/vector-search/endpoints` → `gwb_sequence_search_vs_endpoint` (STANDARD type) — went ONLINE in seconds (e2fe has VS infra pre-warmed)
2. SQL `ALTER TABLE mmt_gwb.genesis_workbench.sequence_embeddings SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` — required for Delta Sync index
3. `POST /api/2.0/vector-search/indexes` with primary_key=`seq_id`, embedding_dim=1280, pipeline_type=TRIGGERED, columns_to_sync=[`seq_id`]
4. Initial sync of 1M rows takes ~30-60 min; search is partially queryable earlier

**Sweep pattern is now confirmed across THREE resource classes:**
| Class | Bundle-managed? | Swept? |
|---|---|---|
| SQL warehouse | NO | YES |
| Serving endpoints | NO | YES (10/11) |
| Vector Search endpoint + index | NO | YES |
| Catalog / schema / volumes / app / jobs / parabricks_cluster / UC models | bundle or UC-managed | NO |

The sweeper's pattern is clear: **non-bundle-managed, workspace-level lifecycle resources** are the targets. Bundle terraform state + UC-governance-protected resources all survived. This is the same pattern as the warehouse sweep earlier today.

**Action items added:**
- Add `gwb_sequence_search_vs_endpoint` + `sequence_embedding_index` to whatever post-demo resource-tagging sweep we do (VS endpoints support tags via separate API)
- Update `claude_skills/SKILL_GENESIS_WORKBENCH_RECONSTRUCTION.md` to add **Pattern 3: Vector Search swept** with the recovery sequence above (next session)
- Heartbeat enhancement candidate: include a `vector-search-indexes get` ping for known indexes to refresh activity timestamps (similar to the SELECT 1 warehouse ping pattern)
