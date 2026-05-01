---
name: genesis-workbench-deploy-wizard
description: Interactive guided deployment of Genesis Workbench to a Databricks workspace. Walks the user through cloud/catalog/schema/warehouse selection, writes the required env files, runs ./deploy.sh module-by-module, and auto-fixes common failures (expired Terraform PGP key, missing catalog, wrong DEFAULT profile, Python version mismatch).
---

# Genesis Workbench Deploy Wizard

Drive a first-time (or repeat) deployment of [Genesis Workbench](https://github.com/databricks-industry-solutions/genesis-workbench) to a Databricks workspace through an interactive, validated conversational flow. This skill asks the user one short question at a time, checks each answer against the live workspace with the `databricks` CLI, writes the `.env` files, and invokes `./deploy.sh` in the correct order.

Use this skill whenever the user says things like "deploy Genesis Workbench", "install GWB on a new workspace", "set up genesis workbench", or is sitting in a cloned `genesis-workbench` repo and asks about deployment.

## Pre-flight checks (run first, in parallel)

```bash
python3 --version                        # warn if < 3.11
poetry --version
jq --version
databricks --version
which terraform && terraform version     # used to bypass expired HashiCorp PGP key
databricks auth profiles                 # confirm DEFAULT exists and is valid
databricks current-user me               # confirms auth works
```

If any required tool is missing:
- **databricks CLI**: https://docs.databricks.com/aws/en/dev-tools/cli/install
- **terraform**: `brew install terraform` (macOS) — required to work around the expired-PGP-key issue below.
- **poetry**: `deploy.sh` installs it via `pip install poetry`; no action needed.
- **jq**: `brew install jq`.

If Python 3.10 is detected, warn but don't block — `Installation.md` recommends 3.11 but most deploys still work. If Python < 3.10, stop and require an upgrade.

## Conversational flow

Ask these in order, using `AskUserQuestion` for anything with a finite choice set (cloud, catalog from list, warehouse from list, yes/no). For free-text values (workspace URL, custom catalog name, schema, app name), ask in plain text.

### 1. Target cloud
`aws` or `azure`. Drives which of `aws.env` / `azure.env` is consumed by `deploy.sh`. GCP is not supported by current deploy scripts.

### 2. Workspace URL
Read the current `DEFAULT` profile from `databricks auth profiles`. Ask the user whether that's the intended target.
- If yes, continue.
- If no, instruct the user: run `databricks auth login --host <url>` in their own terminal (interactive; do **not** auto-run). Wait for them to confirm, then re-check `databricks current-user me`.

### 3. Catalog
Run `databricks catalogs list` and show the user the `MANAGED_CATALOG` rows. Ask: use an existing one (pick from list) or create a new one (type name).
- If create: confirm first, then `databricks catalogs create <name>`. Catalog creation is cheap but visible in the metastore — always confirm before creating.
- Validate the chosen catalog exists with `databricks catalogs get <name>` before writing it to `application.env`.

### 4. Schema
Default: `genesis_workbench`. Remind the user it must be *dedicated* to GWB (the deploy process writes many tables there). Don't validate its existence — `deploy.sh` creates it.

### 5. SQL warehouse
Run `databricks warehouses list` and show warehouses — it's fine if they're `STOPPED`, they start on demand.

**Always re-validate** `sql_warehouse_id` even if `application.env` already exists. Warehouse IDs go stale between sessions (deleted, recreated, workspace swapped). Run `databricks warehouses get <id>` before trusting the file; if it 404s, prompt the user to pick a fresh one.

- If they need to create one, link them to https://docs.databricks.com/aws/en/compute/sql-warehouse/create and wait.
- Capture the warehouse ID, validate with `databricks warehouses get <id>`.

### 6. Core module settings
Read current `modules/core/module.env` (if present) and offer each field with the current value as a default the user can accept or override:
- `dev_user_prefix` — e.g. `demo`. Namespaces dev resources.
- `app_name` — Databricks App name, must be unique per workspace (e.g. `genesis-workbench`).
- `secret_scope_name` — e.g. `genesis_workbench_secret_scope`. Created by the deploy if missing.
- `llm_endpoint_name` — default `databricks-claude-sonnet-4-6`. Validate it exists: `databricks serving-endpoints get <name>`.

### 7. Docker-backed modules (BioNeMo, Parabricks, disease_biology) — creds required
**`bionemo`, `parabricks`, AND `disease_biology`** all require docker-registry creds in a `module.env`. The bundle YAML declares these as required vars (see `modules/bionemo/variables.yml`, `modules/parabricks/variables.yml`, and each of `modules/disease_biology/{gwas,variant_annotation,vcf_ingestion}/*/variables.yml`). Missing them → `bundle validate` fails with `no value assigned to required variable parabricks_docker_userid` (or equivalent).

**disease_biology reuses the parabricks creds** — its GWAS pipeline uses parabricks for alignment, and all three sub-bundles (gwas, variant_annotation, vcf_ingestion) import the same var set. Write the same three values to BOTH `modules/parabricks/module.env` AND `modules/disease_biology/module.env`.

For each module the user opts into, collect:
- `<module>_docker_userid` (for disease_biology, use `parabricks_docker_userid`)
- `<module>_docker_token` (secret — handle with care, don't echo back)
- `<module>_docker_image` — can be Docker Hub (`<user>/<image>:<tag>`), NGC (`nvcr.io/...`), or any other registry. Must be pre-built and pushed (see `modules/<module>/docker/build_docker.sh` where present).

If user is pushing to Docker Hub, the token is usually a Docker Hub PAT (`dckr_pat_*`). If NGC, the userid is typically `$oauthtoken` and the token is the NGC API key.

### 7a. Secret-scope setup (recommended for Docker creds)

Instead of storing Docker PATs as plaintext in `module.env`, use a Databricks secret scope and reference the secrets via `{{secrets/<scope>/<key>}}` in `module.env`.

**Automated setup** — repo ships a script that creates the scope + populates the four Docker credentials:

```bash
./scripts/setup_secret_scope.sh <profile> [scope-name]
```

It prompts for the four credentials (bionemo user/token, parabricks user/token — tokens read with `-s` so they don't echo), creates the scope, puts the secrets with the `gwb_<module>_docker_{user,token}` naming convention, and prints the `module.env` reference template to copy-paste.

**Module.env references after setup:**
```
bionemo_docker_userid={{secrets/<scope>/gwb_bionemo_docker_user}}
bionemo_docker_token={{secrets/<scope>/gwb_bionemo_docker_token}}
parabricks_docker_userid={{secrets/<scope>/gwb_parabricks_docker_user}}
parabricks_docker_token={{secrets/<scope>/gwb_parabricks_docker_token}}
```

Full convention + rationale: `docs/deployments/docker-secrets-convention.md`.

### 7b. Init jobs need user_email on first run

Several per-module `dbx_<module>_initial_setup` jobs widget-default `user_email=a@b.com` — trigger them with the actual user email to avoid `ResourceDoesNotExist: Principal UserName(a@b.com)` failures:

```bash
databricks jobs run-now \
  --json '{"job_id":<job-id>,"job_parameters":{"user_email":"<user@databricks.com>"}}' \
  --profile <profile>
```

Affects bionemo, parabricks, and any destroy_module run. Logged as UX-GAPS #15.

### 8. Which additional modules to deploy
After `core`, ask the user to pick from: `protein_studies`, `single_cell`, `small_molecule`, `disease_biology`, `parabricks`, `bionemo`. Deploy one at a time; each triggers long-running background jobs.

**Treat the approved module order as a contract.** Once the user confirms the list in step 8, deploy in exactly that order. If a module is blocked (e.g., waiting on docker creds from step 7) do NOT jump over it to a later unblocked module without first asking the user to explicitly re-approve the swap. Past user feedback: silent reordering has been rejected.

## Writing the env files

Use the `Write` tool — **no comments, no blank lines** (the `paste -sd,` used in `deploy.sh:44-45` flattens comments into the bundle variable string and breaks).

**`application.env`** (repo root):
```
workspace_url=<url>
core_catalog_name=<catalog>
core_schema_name=<schema>
sql_warehouse_id=<warehouse_id>
```

**`modules/core/module.env`**:
```
dev_user_prefix=<prefix>
app_name=<app_name>
secret_scope_name=<secret_scope_name>
llm_endpoint_name=<llm_endpoint_name>
```

**`modules/bionemo/module.env`** (only if deploying BioNeMo):
```
bionemo_docker_userid=<userid>
bionemo_docker_token=<token>
bionemo_docker_image=<image>
```

**`modules/parabricks/module.env`** (only if deploying Parabricks — **also required**, not just BioNeMo):
```
parabricks_docker_userid=<userid>
parabricks_docker_token=<token>
parabricks_docker_image=<image>
```

`aws.env` / `azure.env` have sensible defaults; only overwrite if the user asks for non-default node types.

## Auto-patch for expired Terraform PGP key

Databricks CLI tries to download Terraform and verify HashiCorp's PGP signature, which has expired. Symptom:

```
Error: error downloading Terraform: unable to verify checksums signature: openpgp: key expired
```

Before running any deploy, check whether `deploy.sh` already exports `DATABRICKS_TF_EXEC_PATH` and `DATABRICKS_TF_VERSION`. If not, inject these two lines immediately after `set -e`:

```bash
export DATABRICKS_TF_EXEC_PATH=$(which terraform)
export DATABRICKS_TF_VERSION=$(terraform version -json | jq -r .terraform_version)
```

(Use the user's actual locally-installed Terraform binary + version, not hardcoded paths.) This tells the Databricks CLI to use the local Terraform and skip the signed download.

**If the exports already exist but with hardcoded paths** (e.g., `DATABRICKS_TF_EXEC_PATH=/opt/homebrew/bin/terraform`), verify the hardcoded path actually exists on the current machine: `[ -x /opt/homebrew/bin/terraform ]`. If not, rewrite the lines to use `$(which terraform)` / `$(terraform version -json | jq -r .terraform_version)`.

## Running the deploy

Always run `core` first:
```bash
./deploy.sh core <cloud>
```

After it completes, verify the lock file:
```bash
ls modules/core/.deployed
```

Then loop through the modules the user chose in step 8, **in the exact order the user approved**:
```bash
./deploy.sh <module> <cloud>
```

Wait for each `deploy.sh` to return (it drives `databricks bundle deploy` + an `initialize_module_job` run). That's fast (minutes). What runs *after* is module-specific:

- `small_molecule`, `protein_studies`, `single_cell`, `disease_biology` — spawn multiple `register_*` jobs against GPU clusters. These are the ones that can hit quota at cluster-create time.
- `bionemo` — spawns `dbx_bionemo_initial_setup`, then registers on-demand finetune/inference jobs (no `register_*` jobs).
- `parabricks` — primarily builds a docker-backed cluster template; actual compute runs on-demand from the app.

**Between modules, poll until the first post-deploy job run spawned by the module reaches `RUNNING` or a terminal state** (not just `PENDING`). This is the quota gate. Use:
```bash
databricks jobs list --limit 50 | grep -iE "<module-keyword>"
databricks jobs list-runs --job-id <id> --limit 1
databricks jobs get-run <run-id> | jq '.state'
```
Only advance to the next module once the predecessor's first job is past `PENDING`. This serializes GPU cluster-create and surfaces quota issues one module at a time.

Watch the Jobs UI at `<workspace_url>/jobs` throughout.

## Error auto-handlers

| Failure signal | Action |
|---|---|
| `openpgp: key expired` | Apply the Terraform env-var patch above. If `terraform` isn't installed locally, tell the user to `brew install terraform` and retry. |
| `Catalog '<name>' does not exist` | Offer to create it via `databricks catalogs create`. Confirm first. |
| `databricks current-user me` fails / 401 | Tell the user to re-auth: `databricks auth login --host <workspace_url>` in their own terminal. |
| `./deploy.sh` exits non-zero before `.deployed` is written | Surface the last ~30 lines of output, check for known patterns (catalog, warehouse, secret scope), and point at `SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md` for anything uncovered. |
| Python < 3.11 detected | Warn once; recommend a 3.11 venv; continue unless < 3.10. |
| `app_name` collision | Databricks Apps names are workspace-unique. If the deploy fails with a name conflict, ask for a new `app_name`, rewrite `modules/core/module.env`, and retry. |
| LLM endpoint not found | Default `databricks-claude-sonnet-4-6` may not exist in every workspace. Offer to pick an existing serving endpoint from `databricks serving-endpoints list`. |

## Post-deploy

When `modules/core/.deployed` exists, print:
- Databricks App URL: `<workspace_url>/apps/<app_name>`
- Jobs UI (to track background registration): `<workspace_url>/jobs`
- Reminder: model-registration jobs for some modules (AlphaFold, Parabricks, BioNeMo) can run for many hours.

Offer the user the next module to deploy, or stop.

## When to use this skill

- User says "deploy Genesis Workbench", "install GWB", "set up Genesis Workbench on a new workspace", "run deploy.sh".
- User is in a cloned `genesis-workbench` repo and asks about deployment steps.
- User hits a deploy failure and asks for guided recovery — resume at the appropriate step.

## When NOT to use this skill

- User is destroying / tearing down (see `destroy.sh`; a separate skill would be appropriate).
- User is developing a new module — use `genesis-workbench-development` instead.
- User is troubleshooting a UI workflow — use `genesis-workbench-workflows` or `_troubleshooting`.

## Cross-workspace state hygiene (required pre-step when switching workspaces)

**When pointing the same local repo at a new workspace** (e.g., switching from sandbox to a new deploy target), clear these two kinds of local state or `./deploy.sh core` will silently skip `initialize_core_job` and the deploy fails downstream with `TABLE_OR_VIEW_NOT_FOUND`:

```bash
# Backup + clear .deployed markers
for f in $(find modules -name ".deployed" -not -path "*/node_modules/*"); do
  mv "$f" "${f}.<prior-workspace-suffix>"
done

# Backup + clear bundle state dirs (per-target terraform state)
for d in $(find modules -name "prod_aws" -type d -path "*/.databricks/bundle/*"); do
  mv "$d" "${d}.<prior-workspace-suffix>"
done

# Optional durable backup (since .databricks/ is gitignored)
tar czf docs/deployments/<prior-ws>/bundle-state-backup-$(date +%Y%m%d-%H%M).tar.gz \
  --exclude='.terraform' --exclude='.terraform.lock.hcl' --exclude='plan' \
  $(find modules -name "prod_aws.<prior-ws>" -type d)
```

Also back up the env files themselves (they're gitignored, so branch switching doesn't swap them):
```bash
cp application.env                   application.env.<prior-ws>.bak
cp modules/core/module.env          modules/core/module.env.<prior-ws>.bak
cp modules/bionemo/module.env       modules/bionemo/module.env.<prior-ws>.bak
cp modules/parabricks/module.env    modules/parabricks/module.env.<prior-ws>.bak
cp modules/disease_biology/module.env  modules/disease_biology/module.env.<prior-ws>.bak
```

## Shared-workspace caveats (e.g., e2-demo-field-eng)

- **300-app workspace cap.** Deploy fails at terraform's `create app` step if workspace is near the cap. Keep a placeholder app alive UNTIL the real app lands, then delete placeholder.
- **App-name uniqueness.** `genesis-workbench` may be taken — prefix (e.g., `gwb-<owner>-<target>`).
- **`enableDcs` toggle.** Confirm Container Services is on before deploying bionemo/parabricks/disease_biology:
  ```bash
  # Check (expect {"enableDcs":"true"}; null or "false" means off)
  databricks api get "/api/2.0/workspace-conf?keys=enableDcs" --profile <profile>

  # If off AND you're in the `admins` workspace group, enable via CLI:
  databricks api patch "/api/2.0/workspace-conf" --json '{"enableDcs":"true"}' --profile <profile>
  ```
  Docs: https://docs.databricks.com/aws/en/compute/custom-containers#enable-container-services
- **Catalog create permission.** Test by creating a throwaway catalog (then delete with `--force` — catalogs auto-create a default schema so need force). Most admins group members can.
- **Async registration jobs (scGPT / SCimilarity).** `./deploy.sh single_cell` returns ✅ SUCCESS before scGPT + SCimilarity registration sub-jobs complete (15-30 min each). UI workflows on those tabs aren't usable until async jobs finish. See DEPLOY_MONITOR skill for how to check.

## Related skills

- `genesis-workbench` — overview of modules and workflows.
- `genesis-workbench-installation` — reference documentation for the deployment process.
- `genesis-workbench-troubleshooting` — recipes for common post-deploy failures.
- `genesis-workbench-deploy-monitor` — for watching jobs + diagnosing failures during/after deploy (includes cross-workspace state hygiene details).
- `databricks-authentication` — use this if the `DEFAULT` profile needs to be reset.
