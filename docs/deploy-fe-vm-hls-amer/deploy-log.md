# Genesis Workbench Deployment to fe-vm-hls-amer

Deployment log and code changes for deploying Genesis Workbench to
`https://fe-vm-hls-amer.cloud.databricks.com` (AWS, workspace ID `1602460480284688`).

Date: 2026-03-11
Deployer: may.merkletan@databricks.com

---

## Target Environment

| Property | Value |
|----------|-------|
| Workspace | https://fe-vm-hls-amer.cloud.databricks.com |
| Cloud | AWS |
| Catalog | `hls_amer_catalog` (shared, pre-existing) |
| Schema | `mmt_genesis_workbench` (created by deploy) |
| SQL Warehouse | `18ca4fa4ce58f74c` |
| App Name | `gwb-mmt-app` |
| Secret Scope | `gwb_mmt_PAT` |
| App URL | https://gwb-mmt-app-1602460480284688.aws.databricksapps.com |

## Previous Environment (Azure)

| Property | Value |
|----------|-------|
| Workspace | https://adb-830292400663869.9.azuredatabricks.net |
| Catalog | `genesis_workbench` |
| Schema | `mmt_gwb_demo` |
| SQL Warehouse | `1ced721a01ee2aff` |

---

## Pre-Deployment Steps

### 1. CLI Auth Setup

```bash
databricks auth login --host https://fe-vm-hls-amer.cloud.databricks.com --profile fe-vm-hls-amer
```

### 2. Pre-flight Checks

Verified:
- User is workspace admin (groups: admins, hls-ssa, users)
- SQL warehouses available
- Apps supported on workspace (29+ already deployed)
- No `genesis_workbench` catalog exists (user lacks catalog creation permissions)
- Decided to use existing `hls_amer_catalog` instead (shared catalog with `ALL_PRIVILEGES` granted to `account users`)

### 3. Create Groups

Genesis Workbench requires two workspace groups for its internal permissions system.
The deploy scripts reference these groups but **do not create them**.

```bash
databricks groups create --display-name genesis-admin-group --profile fe-vm-hls-amer
databricks groups create --display-name genesis-users --profile fe-vm-hls-amer
```

Add deployer to admin group (use deployer's user ID):
```bash
databricks groups patch <GROUP_ID> --json '{
  "Operations": [{"op": "add", "path": "members", "value": [{"value": "<USER_ID>"}]}],
  "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"]
}' --profile fe-vm-hls-amer
```

### 4. Clean Old Deployment State

Old Azure deployment left `.databricks/` dirs (terraform state) and `.deployed` markers
that would conflict with the new workspace.

```bash
# 11 .databricks/ dirs and 4 .deployed markers
find modules -name ".databricks" -type d -exec rm -rf {} +
find modules -name ".deployed" -delete
```

---

## Configuration Changes

### `application.env`

Changed from Azure to AWS target. **No comments allowed** in this file — `deploy.sh` uses
`paste -sd,` to concatenate all lines into a comma-separated string for bundle variables.
Comments, blank lines, and trailing whitespace all break the bundle validation.

```env
workspace_url=https://fe-vm-hls-amer.cloud.databricks.com/
core_catalog_name=hls_amer_catalog
core_schema_name=mmt_genesis_workbench
sql_warehouse_id=18ca4fa4ce58f74c
```

**Key decision:** Using `hls_amer_catalog` (pre-existing shared catalog) instead of a
dedicated `genesis_workbench` catalog. The deploy scripts only create the schema, not the
catalog. This works because:
- `account users` already has `ALL_PRIVILEGES` on `hls_amer_catalog`
- Schema `mmt_genesis_workbench` is created automatically by the deploy script
- No other config changes needed — `core_catalog_name` propagates everywhere via secrets

### No changes to `aws.env` or `modules/core/module.env`

These were already correct for AWS deployments.

---

## Code Changes

### `modules/core/notebooks/initialize_core.py` (line 145-146)

**Problem:** The init notebook runs:
```python
spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog} TO `{app.service_principal_client_id}`")
spark.sql(f"GRANT ALL PRIVILEGES ON SCHEMA {catalog}.{schema} TO `{app.service_principal_client_id}`")
```
This requires `MANAGE` permission on the catalog. When using a shared catalog where the
deployer is not the catalog owner, this fails with `PERMISSION_DENIED`.

**Root cause:** On `hls_amer_catalog`, `account users` already has `ALL_PRIVILEGES`,
so the app's service principal already has access. The GRANT is redundant but the
permission to issue GRANTs is not available.

**Fix:** Wrap in try/except so a permission error doesn't block the rest of initialization:
```python
try:
    spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog} TO `{app.service_principal_client_id}`")
    spark.sql(f"GRANT ALL PRIVILEGES ON SCHEMA {catalog}.{schema} TO `{app.service_principal_client_id}`")
except Exception as e:
    print(f"⚠️ Grant failed (may already be inherited): {e}")
    print("Continuing — verify SP has access to catalog/schema.")
```

**When to apply this change:** Any time deploying to a shared catalog where the deployer
does not have `MANAGE` on the catalog, but the app SP already has access via inherited
privileges (e.g., `account users` has `ALL_PRIVILEGES`).

**When NOT to apply:** If deploying to a dedicated catalog where the deployer is the
owner, the original code works fine.

---

## Deployment Execution

### Deploy Core

```bash
export DATABRICKS_HOST=https://fe-vm-hls-amer.cloud.databricks.com
export DATABRICKS_PROFILE=fe-vm-hls-amer
./deploy.sh core aws
```

**Flow:** Creates secret scope -> builds Poetry wheel -> creates schema -> bundle deploy ->
runs init job -> deploys Streamlit app -> copies wheel to Volumes.

**Issue encountered:** First run failed at init job due to GRANT permission (see code
change above). After applying the fix:
1. Re-deployed bundle to upload modified notebook
2. Re-ran init job — succeeded

### Wheel Copy Issue

The core `deploy.sh` copies the genesis_workbench wheel to UC Volumes at:
`/Volumes/{catalog}/{schema}/libraries/genesis_workbench-0.1.0-py3-none-any.whl`

The `libraries` volume is created by the bundle deployment (defined in `databricks.yml`),
but the wheel copy step (`databricks fs cp`) happened during the first (failed) deploy run.
The volume existed but was empty after the failure.

**Manual fix:**
```bash
databricks fs cp modules/core/library/genesis_workbench/dist/genesis_workbench-0.1.0-py3-none-any.whl \
  dbfs:/Volumes/hls_amer_catalog/mmt_genesis_workbench/libraries/genesis_workbench-0.1.0-py3-none-any.whl \
  --overwrite --profile fe-vm-hls-amer
```

Sub-module registration notebooks (`register_scanpy_job.py`, etc.) search for this wheel
at runtime via `dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/libraries")` and `%pip install`
it. Without the wheel, all sub-module registration jobs fail with
`ModuleNotFoundError: No module named 'genesis_workbench'`.

### Deploy Single Cell

```bash
./deploy.sh single_cell aws
```

**Sub-modules deployed:**
| Sub-module | Status | Notes |
|------------|--------|-------|
| scanpy | Deployed + registered | Gene references downloaded |
| rapidssinglecell | Deployed + registered | Gene references downloaded |
| scGPT | Deployed, background job | Model download/registration (long-running) |
| SCimilarity | Deployed, background job | Model download/registration (long-running) |

### Post-Deploy: Tag Untagged Job

The `download_gene_references_gwb` job (ID `599617698721092`) was created without tags,
unlike other GWB jobs which have `application`, `created_by`, and `module` tags.

```bash
databricks jobs update --json '{
  "job_id": 599617698721092,
  "new_settings": {
    "tags": {
      "application": "genesis_workbench",
      "created_by": "may_merkletan",
      "module": "single_cell"
    }
  }
}' --profile fe-vm-hls-amer
```

---

## Deploy Protein Studies

```bash
./deploy.sh protein_studies aws
```

**Sub-modules deployed (all SUCCESS as of 2026-03-13):**
| Sub-module | Status | Notes |
|------------|--------|-------|
| alphafold | DEPLOYED | 7/7 download tasks succeeded after 5 attempts (see AlphaFold section below) |
| esmfold | DEPLOYED | Re-run with ON_DEMAND succeeded |
| boltz | DEPLOYED | Re-run with ON_DEMAND succeeded |
| rfdiffusion | DEPLOYED | Model registration succeeded |
| protein_mpnn | DEPLOYED | Model registration succeeded |

### Spot Instance Fix (2026-03-12)

Jobs `register_esmfold`, `register_boltz`, and `alphafold_register_and_downloads` failed
due to AWS spot instance preemption. Added `aws_attributes: availability: ON_DEMAND` to
all 16 job clusters across 10 YAML files and redeployed the 3 affected sub-modules.

See [redeploy-failed-jobs.md](redeploy-failed-jobs.md) for targeted redeploy commands.

### Redeploy Attempt 2 (2026-03-12)

After adding ON_DEMAND and redeploying, esmfold failed again with "driver is lost"
(not spot preemption — possible OOM on `g4dn.4xlarge`). Also discovered
`download_uniprot_cluster` in alphafold was missing `aws_attributes` — fixed and
redeployed. All 3 jobs re-triggered:

```bash
databricks jobs run-now 747548824052399 --profile fe-vm-hls-amer  # esmfold
databricks jobs run-now 970641084894060 --profile fe-vm-hls-amer  # boltz
databricks jobs run-now 151110797461064 --profile fe-vm-hls-amer  # alphafold
```

### Results (2026-03-13)

- **Boltz:** SUCCESS — ON_DEMAND fix resolved it (previous failure was spot preemption)
- **ESMFold:** SUCCESS — ON_DEMAND fix resolved it (previous "driver is lost" was spot preemption, not OOM as suspected)
- **AlphaFold:** 6/7 tasks SUCCESS, `pdb_mmcif` completed download of 250,359 structures and copying to Volume. See `alphafold-debug-summary-slack.md` for the condensed writeup.

**Auth note:** The default CLI profile points to `e2-demo-field-eng`, not `fe-vm-hls-amer`.
Always pass `--profile fe-vm-hls-amer` to all `databricks` commands, or re-auth will
silently target the wrong workspace.

### AlphaFold Download Failures (2026-03-12)

The `alphafold_register_and_downloads` job partially succeeded. 4 of 7 tasks completed:

| Task | Status | Error |
|------|--------|-------|
| register_run_job | SUCCESS | |
| common_files | SUCCESS | |
| download_mgnifyPLUS | SUCCESS | |
| download_params | SUCCESS | |
| download_uniprot | FAILED | `download_uniprot.sh` exit code 2 — likely network/disk issue during large download |
| download_unirefs | FAILED | `download_uniref90.sh` exit code 2 — likely network/disk issue during large download |
| pdb_mmcif | FAILED | `download_pdb_mmcif.sh` exit code 10 (aria2c error) — download failure from wwpdb.org |

**Root cause:** AlphaFold v2.3.2 download scripts use FTP URLs (`ftp://ftp.ebi.ac.uk`,
`ftp://ftp.uniprot.org`) and rsync (port 33444), both of which are blocked on AWS
Databricks clusters. The same files are available over HTTPS.

**Fix applied in `download_setup.py`:**
1. `sed` commands patch FTP→HTTPS in `download_uniprot.sh`, `download_uniref90.sh`,
   `download_pdb_mmcif.sh`, and `download_pdb_seqres.sh`
2. A Python cell creates `download_pdb_mmcif_https.sh` — replaces the rsync-based bulk
   download with `aria2c` against the EBI HTTPS mirror
   (`https://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/divided/mmCIF/`).
3. `download_pdb_mmcif.py` updated to call the HTTPS script instead of the original

**Four distinct issues, each only visible after fixing the previous one:**

| Layer | Issue | Symptom | Fix |
|-------|-------|---------|-----|
| 1. Infra | Spot instance preemption | `Cluster terminated because driver node is a spot instance` | Added `aws_attributes: availability: ON_DEMAND` to all job clusters |
| 2. Network | FTP/rsync blocked on AWS VPC | aria2c `0B/0B` for 5 min then `errorCode=2 Timeout` on `ftp://` URLs; rsync `Connection timed out` on port 33444 | `sed` patches FTP→HTTPS in download scripts; HTTPS replacement script for rsync |
| 3. Notebook format | Heredocs inside `# MAGIC %sh` cells are malformed | Script created by heredoc had quoting/escaping errors, exit code 1 | Moved script creation to a Python cell (`with open(...) as f: f.write(script)`) |
| 4. Script bugs | `grep`/`sed` left trailing `"` in subdir names (`aa"` not `aa`); `wget -r` silently failed on HTTPS dir listings | wget hit `BASE_URL/aa"/` (nonexistent), downloaded 0 files, `find -empty -delete` removed `raw/` dir — all hidden by `|| true` + `2>/dev/null` | Replaced wget with aria2c: parse actual `.cif.gz` links per subdir, batch download with 16 parallel connections. Added file count verification (exit on zero). |
| 5. HTML parsing | `sed 's/[^a-z0-9]//g'` kept "href" prefix (alphanumeric) | All 1119 subdirs parsed as `href0a` instead of `0a`, every dir showed `(0 files)`, verification caught 0 total and exited with error | Replaced sed with `cut -d'"' -f2 \| tr -d '/'` to extract value between quotes |

**Key lessons:**
- Never use `|| true` + `2>/dev/null` together — it hides all evidence of failure
- `wget -r` on HTTPS directory listings is unreliable; `aria2c` with explicit URLs is better
- Always add a verification step after bulk downloads (count files, exit on zero)
- When debugging layered failures, each fix may expose a different underlying issue
- When stripping non-alphanumeric chars, remember the field name itself (e.g. "href") may be alphanumeric — use positional extraction (`cut`, field splitting) instead of character-class deletion
- The HTTPS replacement is slower than rsync (per-subdir `curl` to list files) but provides clear progress: aria2c logs each file with speed, and the outer loop shows `[N/1119] Downloading XX/ (M files)...`

The downloads are idempotent (guarded by `if [ ! -d "$MODEL_VOLUME/datasets/..." ]`),
so re-running the job will skip completed downloads and retry only the missing ones.

**Note on retries:** The alphafold job provisions all 6 clusters on every run, even if
most tasks skip immediately (data already exists). This wastes ~$0.50/cluster on
provisioning clusters that aren't needed. For targeted retries of individual tasks,
consider running the specific notebook on an all-purpose cluster instead.

**To retry the full job:** Redeploy alphafold bundle first (to upload modified notebooks), then re-trigger:
```bash
cd modules/protein_studies/alphafold/alphafold_v2.3.2
databricks bundle deploy --profile fe-vm-hls-amer \
  --var="$(paste -sd, ../../../../application.env),$(paste -sd, ../../../../aws.env)"
databricks jobs run-now 151110797461064 --profile fe-vm-hls-amer
```

### ON_DEMAND Rationale

All job clusters use `aws_attributes: availability: ON_DEMAND` instead of spot instances.
GWB jobs are long-running (hours for model downloads/registration), non-checkpointed,
and partially non-idempotent. Spot preemption wastes all compute up to the preemption
point. The ~30-40% cost premium for ON_DEMAND is negligible compared to wasted spot
compute and manual re-triggering overhead.

---

## Modules Not Yet Deployed

| Module | Notes |
|--------|-------|
| `bionemo` | Requires valid Docker/NGC credentials in `modules/bionemo/module.env` |

---

## Known Issues & Workarounds

### 1. `application.env` does not support comments

`deploy.sh` concatenates all lines with `paste -sd,`. Any non-key=value lines (comments,
blanks, trailing whitespace) are passed as bundle variables and cause validation errors.

**Workaround:** Keep `application.env` strictly as key=value pairs with no blank lines,
comments, or trailing whitespace.

**Potential fix for repo:** Update `deploy.sh` to filter comments and blank lines:
```bash
EXTRA_PARAMS_GENERAL=$(grep -v '^\s*#' ../../application.env | grep -v '^\s*$' | sed 's/[[:space:]]*$//' | paste -sd,)
```

### 2. Shared catalog requires MANAGE for GRANT

When deploying to a catalog the deployer doesn't own, the init notebook fails on
`GRANT USE CATALOG`.

**Workaround:** Try/except around the GRANT (see code change above).

**Potential fix for repo:** Check if privileges are already inherited before attempting
GRANT, or make the GRANT step non-fatal by default.

### 3. Wheel not copied on partial deploy failure

If `deploy.sh core` fails after bundle deploy but before the wheel copy step, sub-modules
will fail with `ModuleNotFoundError`. The wheel copy is near the end of the script.

**Workaround:** Manually copy the wheel (see command above).

**Potential fix for repo:** Move wheel copy earlier in the script, or add a pre-check
in sub-module deploy scripts.

### 4. `download_gene_references_gwb` job missing tags

This job is created without the standard GWB tags (`application`, `created_by`, `module`),
making it harder to identify and manage alongside other GWB jobs.

**Workaround:** Manually tag after deployment (see command above).

**Potential fix for repo:** Add tags to the job definition in the relevant
`resources/job_definition.yml`.

---

## Verification Checklist

After deployment, verify:

- [ ] App accessible at https://gwb-mmt-app-1602460480284688.aws.databricksapps.com
- [ ] Schema `hls_amer_catalog.mmt_genesis_workbench` contains tables: `models`, `model_deployments`, `settings`, `user_settings`
- [ ] Secret scope `gwb_mmt_PAT` exists with expected keys
- [ ] Wheel exists at `/Volumes/hls_amer_catalog/mmt_genesis_workbench/libraries/genesis_workbench-0.1.0-py3-none-any.whl`
- [ ] All GWB jobs tagged with `application: genesis_workbench`
- [ ] scGPT background model registration job completes
- [ ] SCimilarity background model registration job completes
- [ ] (After protein_studies deploy) All protein sub-module jobs registered
