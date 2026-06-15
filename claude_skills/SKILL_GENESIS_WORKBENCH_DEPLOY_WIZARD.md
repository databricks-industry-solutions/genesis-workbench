---
name: genesis-workbench-deploy-wizard
description: Interactive, guided deployment of Genesis Workbench to a Databricks workspace. A consistent wizard that (1) confirms the TARGET WORKSPACE, (2) verifies the databricks CLI is installed and pointed at that workspace, (3) asks which modules to install, (4) validates/creates only the env files those modules need, then (5) runs ./deploy.sh module-by-module and auto-fixes common failures.
---

# Genesis Workbench Deploy Wizard

Drive a deployment of [Genesis Workbench](https://github.com/databricks-industry-solutions/genesis-workbench) (GWB) to a Databricks workspace through an interactive, validated conversational flow that feels the same for every user. Ask **one short question at a time**, check each answer against the live workspace with the `databricks` CLI, write only the `.env` files the chosen modules need, then invoke `./deploy.sh` in the correct order.

**Trigger this skill** whenever the user says "deploy", "I want to deploy", "deploy Genesis Workbench", "install GWB", "set up genesis workbench on a workspace", or runs/asks about `./deploy.sh`.

> **The wizard runs in a fixed phase order. Do not skip ahead, do not reorder.** The order below is the contract — it confirms the *where* before the *what*, and only collects configuration for modules the user actually wants. Each phase gates the next.

## Wizard banner (print this first, verbatim, before Phase 0)

Open every deploy session by printing the Genesis Workbench banner ("Genesis Workbench" + a DNA helix accent), then the "Deploy Wizard" subtitle:

```
   ╔═╗╔═╗╔╗╔╔═╗╔═╗╦╔═╗   ╦ ╦╔═╗╦═╗╦╔═╔═╗╔═╗╔╗╔╔═╗╦ ╦    ╲ ╱
   ║ ╦║╣ ║║║║╣ ╚═╗║╚═╗   ║║║║ ║╠╦╝╠╩╗╠╩╗║╣ ║║║║  ╠═╣      ╳
   ╚═╝╚═╝╝╚╝╚═╝╚═╝╩╚═╝   ╚╩╝╚═╝╩╚═╩ ╩╚═╝╚═╝╝╚╝╚═╝╩ ╩    ╱ ╲
                   Genesis Workbench — Deploy Wizard
```

---

## Phase 0 — Confirm the TARGET WORKSPACE (always first)

Never assume the workspace. Before anything else, establish and **explicitly confirm** which Databricks workspace this deploy targets.

1. Discover the current target:
   ```bash
   databricks auth profiles                 # lists profiles + hosts + validity
   databricks current-user me 2>&1          # resolves the ACTIVE default identity + host
   ```
2. Show the user the resolved host (workspace URL) and the user/email it authenticated as, then confirm with `AskUserQuestion`:
   > "Deploy Genesis Workbench to **`<host>`** (authenticated as `<email>`)? "
   Options: **Yes, this workspace** · **No, a different workspace**.
3. If the user picks a different workspace, ask for the target workspace URL (free text), then have them authenticate **in their own terminal** (interactive — never auto-run a login):
   - Suggest they type: `! databricks auth login --host <url>` (the `!` runs it in this session so output lands here), **or** select an existing profile.
   - Re-run `databricks current-user me` and re-confirm the host before continuing.

Do not leave Phase 0 until the user has affirmatively confirmed the exact workspace host.

---

## Phase 1 — CLI installed + configured to target that workspace as the default

The deploy scripts shell out to `databricks` using the **default profile** unless `DATABRICKS_CONFIG_PROFILE` is set. Make the target workspace the one those commands will actually hit.

1. **Installed?** `databricks --version`. If missing → stop and link https://docs.databricks.com/aws/en/dev-tools/cli/install.
2. **Default points at the confirmed workspace?** The host from `databricks current-user me` (Phase 0) is what the *default* profile resolves to. Confirm it equals the workspace the user approved.
   - If the user authenticated via a **named profile** (not `DEFAULT`), you have two consistent options — pick one and use it for **every** `databricks` and `./deploy.sh` command for the rest of the session:
     - **(a)** Have the user make it the default, or
     - **(b)** Prefix every command with `DATABRICKS_CONFIG_PROFILE=<profile>` (e.g. `DATABRICKS_CONFIG_PROFILE=ci-demo ./deploy.sh core aws`).
   - State plainly which one you're using so it's reproducible. Do not silently switch profiles mid-flow.
3. **Auth valid?** `databricks current-user me` must return a user (not a 401). If it 401s, send the user back to `databricks auth login`.
4. **Other pre-flight tools** (run in parallel; warn, don't all block):
   ```bash
   python3 --version          # < 3.10 → stop and require upgrade; 3.10 → warn only
   jq --version               # brew install jq
   which terraform && terraform version   # needed for the PGP-key workaround (see Auto-patch)
   poetry --version           # deploy.sh installs it via pip if absent — no action needed
   ```

---

## Phase 2 — Cloud

Determine the cloud (drives which `*.env` `deploy.sh` consumes and the bundle target). **Auto-detect from the workspace host, then confirm:**
- `*.cloud.databricks.com` → **aws**
- `*.azuredatabricks.net` → **azure**
- `*.gcp.databricks.com` → **gcp**

Confirm with `AskUserQuestion` (`aws` / `azure` / `gcp`). All three are supported by `deploy.sh` (`prod_aws` / `prod_azure` / `prod_gcp`). The matching `aws.env` / `azure.env` / `gcp.env` already ship with sensible node-type defaults — only touch them if the user explicitly wants non-default compute.

---

## Phase 3 — Which modules to install (multi-select, EARLY)

This selection **drives the rest of the wizard** — you only validate/create env files for the modules chosen here.

1. **`core` is mandatory and always first.** Check if it's already installed:
   ```bash
   ls modules/core/.deployed 2>/dev/null
   ```
   - **`.deployed` exists** → this is an existing install. ⚠️ See the hard rule below: refresh core with `update.sh`, never `./deploy.sh core`.
   - **absent** → fresh install; `./deploy.sh core <cloud>` is the first step.
2. Ask, with `AskUserQuestion` (`multiSelect: true`), which **additional** modules to deploy beyond core:
   - `large_molecule` — AlphaFold, Boltz, ESMFold, ProteinMPNN, RFdiffusion, enzyme optimization, …
   - `small_molecule` — ChemProp, DiffDock, GenMol, KERMT, NetSolP, …
   - `single_cell` — scGPT, SCimilarity, scanpy, rapids-singlecell, …
   - `genomics` — GWAS, variant annotation, VCF ingestion, **parabricks** (GPU, needs docker creds)
   - `bionemo` — NVIDIA BioNeMo ESM-2 finetune/inference (container-only, needs docker creds)
3. **Treat the approved list + order as a contract.** Deploy in exactly the order confirmed. If a module is blocked (e.g. waiting on docker creds) do NOT silently jump to a later one — ask the user to re-approve any reordering. (Past user feedback: silent reordering was rejected.)

---

## Phase 4 — `application.env` (core-level config; always required)

This file lives at the **repo root** and is consumed by every module deploy. Read the existing file if present, show current values, validate each against the live workspace, and let the user accept or override.

| Key | How to validate / source |
|---|---|
| `workspace_url` | The confirmed host from Phase 0 (no trailing path). |
| `core_catalog_name` | `databricks catalogs list` → show `MANAGED_CATALOG` rows. Existing → `databricks catalogs get <name>`. New → **confirm**, then `databricks catalogs create <name>`. |
| `core_schema_name` | Default `genesis_workbench`. Must be **dedicated** to GWB (deploy writes many tables). `deploy.sh` creates it — don't pre-validate. |
| `sql_warehouse_id` | `databricks warehouses list` → show `HEALTHY` ones. Validate the pick with `databricks warehouses get <id>`. If none, link https://docs.databricks.com/aws/en/compute/sql-warehouse/create and wait. |

---

## Phase 5 — `modules/core/module.env` (always, since core always deploys)

Read the current file if present; offer each field with its current value as the default.

| Key | Notes |
|---|---|
| `dev_user_prefix` | Namespaces dev resources, e.g. `demo`. May be blank for a clean prod-style install. |
| `app_name` | Databricks App name — **workspace-unique**. Default `genesis-workbench`. |
| `secret_scope_name` | e.g. `genesis_workbench_secret_scope`. Created by the deploy if missing. |
| `llm_endpoint_name` | Default `databricks-claude-sonnet-4-6`. Validate: `databricks serving-endpoints get <name>`; if absent, offer to pick from `databricks serving-endpoints list`. |

---

## Phase 6 — Module-specific `module.env` — ONLY for selected modules

Only `core` (Phase 5), `bionemo`, and `genomics` need a `module.env`. **`large_molecule`, `single_cell`, and `small_molecule` need NO `module.env`** — they run off `application.env` + the cloud env file. So:

- **If `bionemo` was selected** → `modules/bionemo/module.env`:
  ```
  bionemo_docker_userid=<userid>
  bionemo_docker_token=<token>
  bionemo_docker_image=<image>
  ```
  Remind the user the BioNeMo container must be pre-built + pushed (`modules/bionemo/docker/build_docker.sh`). Treat the token as a secret — don't echo it back.

- **If `genomics` was selected AND the parabricks submodule is wanted** → `modules/genomics/module.env`:
  ```
  parabricks_docker_userid=<userid>
  parabricks_docker_token=<token>
  parabricks_docker_image=<image>
  ```
  (parabricks is a genomics submodule; the GWAS / variant-annotation / VCF-ingestion submodules don't need docker creds.)

- **If only `large_molecule` / `single_cell` / `small_molecule` were selected** → skip Phase 6 entirely; do **not** prompt for or create any extra `module.env`.

### Env-file writing rule (critical)
Write env files with the `Write` tool — **no comments, no blank lines, `key=value` only**. `deploy.sh` flattens each file with `paste -sd,` into the bundle's `--var` string; a comment or blank line corrupts the variable list and breaks the deploy.

---

## Phase 7 — Confirm the plan, then automate

Echo back a concise plan and get a final go-ahead:
> Workspace `<host>` · cloud `<cloud>` · profile `<default|name>` · catalog `<cat>` · schema `<schema>` · modules: **core → <approved order>**.

### ⚠️ Hard rule: never run `./deploy.sh core` on a workspace with a live install
`./deploy.sh core <cloud>` re-runs `initialize_module_job`, which **re-creates the GWB schema tables** — including `settings` / `model_deployments` / `user_profiles` — wiping live app state (confirmed user-reported regression). For an app/library refresh on an existing install:
```bash
cd modules/core
./update.sh <cloud>          # bundle deploy + wheel + app SP grants + publishes node_catalog; no destructive init
```
`./deploy.sh core <cloud>` is correct **only on a brand-new workspace** (no `.deployed`, no state to lose).

### Run order
1. **core first** (fresh workspace only):
   ```bash
   ./deploy.sh core <cloud>
   ls modules/core/.deployed     # verify the lock was written
   ```
2. Then each selected module, **in the approved order, one at a time**:
   ```bash
   ./deploy.sh <module> <cloud>
   ```
   (Prefix with `DATABRICKS_CONFIG_PROFILE=<name>` if you chose option (b) in Phase 1.)

`deploy.sh` returns in minutes (it runs `databricks bundle deploy` + `initialize_module_job`). What runs *after* is module-specific and can be long:
- `small_molecule` / `large_molecule` / `single_cell` / `genomics` → spawn `register_*` jobs on GPU clusters (these hit quota at cluster-create).
- `bionemo` → `dbx_bionemo_initial_setup`, then on-demand finetune/inference jobs.

**Between modules, poll until the predecessor's first post-deploy job reaches `RUNNING` (or terminal), not just `PENDING`:**
```bash
databricks jobs list --limit 50 | grep -iE "<module-keyword>"
databricks jobs list-runs --job-id <id> --limit 1
databricks jobs get-run <run-id> | jq '.state'
```
This serializes GPU cluster-create and surfaces quota issues one module at a time. Watch `<workspace_url>/jobs` throughout.

### Per-submodule deploys
Large modules support `--only-submodule <path>` to deploy one piece at a time (mirrors how submodules are listed in each module's `deploy.sh ALL_SUBMODULES`), e.g.:
```bash
./deploy.sh small_molecule <cloud> --only-submodule kermt/kermt_v2
```
Use this when iterating on a single model rather than the whole module.

---

## Auto-patch for the expired Terraform PGP key

The Databricks CLI downloads Terraform and verifies HashiCorp's (expired) PGP signature:
```
Error: error downloading Terraform: unable to verify checksums signature: openpgp: key expired
```
`deploy.sh` already exports `DATABRICKS_TF_EXEC_PATH` + `DATABRICKS_TF_VERSION` near the top to use a local Terraform and skip the signed download. **Verify the path it points at actually exists on this machine:**
```bash
grep -n "DATABRICKS_TF_EXEC_PATH" deploy.sh
[ -x "<that path>" ] && echo ok
```
If the hardcoded path is wrong for this machine, rewrite those two lines to derive from the local install:
```bash
export DATABRICKS_TF_EXEC_PATH=$(which terraform)
export DATABRICKS_TF_VERSION=$(terraform version -json | jq -r .terraform_version)
```
If `terraform` isn't installed: `brew install terraform` (macOS) and retry.

---

## Error auto-handlers

| Failure signal | Action |
|---|---|
| `openpgp: key expired` | Apply the Terraform env-var patch above; `brew install terraform` if missing. |
| `Catalog '<name>' does not exist` | Offer to `databricks catalogs create <name>` — confirm first. |
| `databricks current-user me` 401 | Re-auth: `databricks auth login --host <workspace_url>` in the user's terminal; re-confirm host. |
| App name collision | Apps names are workspace-unique. Ask for a new `app_name`, rewrite `modules/core/module.env`, retry. |
| LLM endpoint not found | Offer to pick an existing one from `databricks serving-endpoints list`. |
| `./deploy.sh` exits non-zero before `.deployed` | Surface the last ~30 lines; match against catalog/warehouse/secret-scope patterns; hand off to `SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md` for anything else. |
| Python < 3.11 | Warn once, recommend a 3.11 venv; continue unless < 3.10 (then stop). |

---

## Post-deploy

When `modules/core/.deployed` exists, print:
- Databricks App URL: `<workspace_url>/apps/<app_name>`
- Jobs UI (track background registration): `<workspace_url>/jobs`
- Reminder: registration jobs for some models (AlphaFold, Parabricks, BioNeMo) can run for hours.

Then offer the next module in the approved list, or stop.

---

## When to use / not use

**Use** when the user wants to deploy/install GWB, is in a cloned `genesis-workbench` repo asking about deployment, or is recovering from a deploy failure (resume at the relevant phase).

**Don't use** for: tearing down (→ `genesis-workbench-destroy-wizard`), developing a new module (→ `genesis-workbench-development`), or UI workflow questions (→ `genesis-workbench-workflows` / `_troubleshooting`).

## Related skills
- `genesis-workbench-destroy-wizard` — the mirror-image teardown wizard (shares Phases 0–3).
- `genesis-workbench-installation` — reference docs for deployment mechanics.
- `genesis-workbench-troubleshooting` — recipes for post-deploy failures.
- `genesis-workbench` — module/architecture overview.
