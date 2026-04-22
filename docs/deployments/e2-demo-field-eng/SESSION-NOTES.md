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
3. `databricks warehouses delete 1a9a54ac772b8798 --profile DEFAULT`
4. `databricks catalogs delete mmt_gwb --force --profile DEFAULT`
5. Leave `mmt` secret scope in place (has non-GWB keys). Optionally `databricks secrets delete-secret mmt gwb_<*>_docker_{user,token}` for the 4 GWB-added keys.
6. `git checkout mmt/ver_pin_sandbox_setup` or restore `.bak`s to roll env files back.
