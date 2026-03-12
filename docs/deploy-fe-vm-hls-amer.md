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

## Modules Not Yet Deployed

| Module | Notes |
|--------|-------|
| `protein_studies` | Ready to deploy: `./deploy.sh protein_studies aws` |
| `bionemo` | Requires valid Docker/NGC credentials in `modules/core/env.env` |

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
