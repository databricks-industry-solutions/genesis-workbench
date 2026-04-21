# GWB `version_pinning` — UX + deploy gaps

Running log of UX frustrations, deploy-time gotchas, unclear knobs, and "things a new user would trip on" encountered during sandbox deploy + playbook walkthrough.

**Audience:**
- (a) Informs the SA HUNTER playbook ("known caveats" callouts)
- (b) Actionable input to the `version_pinning` branch author for improvement PRs
- (c) Future-May reference if redeploying later

Format: short entry, reproducibility, proposed fix, severity.

---

## 1. Hardcoded Terraform path + version in top-level `deploy.sh`

**Location:** `deploy.sh:7-8`
```bash
export DATABRICKS_TF_EXEC_PATH=/opt/homebrew/bin/terraform
export DATABRICKS_TF_VERSION=1.3.9
```

**Gap:** Two problems for a new user —
- Path `/opt/homebrew/bin/terraform` assumes Apple Silicon macOS + Homebrew. Breaks silently on Intel Mac (`/usr/local/...`), Linux, or non-Homebrew installs.
- Version pinned to `1.3.9` (from ~2023). A fresh `brew install terraform` gets something much newer (1.7.x / 1.9.x). May cause DAB version mismatch warnings or failures.

**How a new user hits it:** They `brew install terraform` per `Installation.md`, run `./deploy.sh`, and either get a "file not found" (wrong path) or a version mismatch.

**Fix (small PR back to `version_pinning`):**
```bash
export DATABRICKS_TF_EXEC_PATH=$(which terraform)
export DATABRICKS_TF_VERSION=$(terraform version -json | jq -r .terraform_version)
```
Also worth adding a preflight check that `which terraform` returns a path — fail fast with a clear message if not.

**Severity:** medium — doesn't block anyone who reads `deploy.sh`, but silently wastes 15-30 min of debugging for anyone who doesn't.

---

## 2. DiffDock's `local/` Dockerfiles are dev-only but look deploy-critical

**Location:** `modules/small_molecule/diffdock/diffdock_v1/local/Dockerfile.esm` + `Dockerfile.scoring` + `test_build.sh`

**Gap:** A first-time reader (Claude included — I made this exact mistake) sees two Dockerfiles and assumes they need to be built + pushed for the DiffDock deploy. In reality, they're a developer sanity-check tool (per `test_build.sh` comment: *"validates that all DiffDock dependencies install correctly in an environment matching the model serving container"*). The actual deploy runs a notebook on a GPU cluster — no Docker.

**How a new user hits it:** They add "build DiffDock images" to their prep checklist, budget hours for it, maybe even try to set up a container registry they don't need.

**Fix (tiny PR):** Add a README at `modules/small_molecule/diffdock/diffdock_v1/local/README.md` with one sentence: *"These Dockerfiles are for local dependency validation only. The deploy runs a notebook on a Databricks GPU cluster — no image build required."*

**Severity:** low, but high surprise-factor.

---

## 3. Databricks CLI minimum version not surfaced in preflight docs

**Location constraint:** `modules/core/databricks.yml:3`
```yaml
databricks_cli_version: ">=0.295.*"
```

**Gap:** Neither `Installation.md` nor the deploy wizard's preflight checks list a minimum CLI version. Users with an older CLI (0.285.x, installed via brew many weeks ago) hit the constraint mid-`bundle validate` instead of at setup time. The failure message is clear once you see it, but wastes a cycle.

**How a new user hits it:** They run `./deploy.sh core aws`, which eventually calls `databricks bundle …`, which fails with:
```
Error: Databricks CLI version constraint not satisfied. Required: >=0.295.*, current: 0.285.0
```

**Fix (two complementary PRs):**
1. `Installation.md`: add "Databricks CLI **≥ 0.295** (`brew upgrade databricks` if older)".
2. `claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md`: augment the preflight block to also check `databricks --version` output against the minimum from `modules/core/databricks.yml`.

**Severity:** medium — fast fix once observed, but catches you out of flow.

---

## 4. Terraform not installable via plain `brew install terraform`

**Discovery:** 2026-04-21 during sandbox preflight. `brew install terraform` returns "formula not found" because Homebrew removed Terraform from core formulae after HashiCorp's license change (MPL → BSL, ~2023).

**Fix for a user running into this:**
```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

This pulls Terraform from HashiCorp's official tap (currently installs v1.14.9 as of 2026-04-21).

**Related gap:** the version gap between what brew installs (1.14.9) and the pin in `deploy.sh:8` (`DATABRICKS_TF_VERSION=1.3.9`) is now ~8 major versions. DAB may or may not complain. Reinforces the case for Gap #1's dynamic-version fix.

**Fix (add to `Installation.md`):** update the terraform install instructions to use the hashicorp tap explicitly.

**Severity:** medium — users with existing HashiCorp-tap terraform (common for long-time HCL users) won't hit this; new users or those who uninstalled Terraform after license change will.

---

## 5. "Start All Endpoints" only in version_pinning, not basic

**Discovery:** 2026-04-21 while drafting playbook Tab 3 (Basic app).

**Gap:** The `Start All Endpoints` keep-alive job (Settings → Endpoint Management) is a version_pinning addition — it isn't in the basic app (gwb-mmt-app, deployed March 2026). Users upgrading from basic to version_pinning may not realise it's now available; users on basic can't take advantage of it even if they've read about it in newer docs.

**How it surfaces:** the app view file (`modules/core/app/views/settings.py`) surfaces "Start All Endpoints" panel. The corresponding Databricks job + env var `START_ALL_ENDPOINTS_JOB_ID` were added as part of `version_pinning`. Basic app deployed from an earlier commit lacks the job.

**Fix (documentation, not code):** `CHANGELOG.md` entry calling out the feature addition + a "What's new" section in `SKILL_GENESIS_WORKBENCH.md` would prevent confusion.

**Severity:** low — but a common "what's changed?" question from customers upgrading.

---

## 6. Basic app has `small_molecules.py` view but no backing module

**Discovery:** 2026-04-21 verifying deploy/fe-vm-hls-amer branch.

**Gap:** `modules/core/app/views/small_molecules.py` exists in the basic app (fe-vm-hls-amer, March 2026 deploy), but there is NO corresponding `modules/small_molecule/` directory. The view renders a near-empty placeholder (just a title). Users clicking the Small Molecules sidebar tab in the basic app see an empty page with no explanation.

**How a user hits it:** They install basic GWB, see a "Small Molecules" tab, click it, see nothing, assume the app is broken.

**Fix:** either (a) remove the view file from basic deploys so the tab doesn't show; (b) add an informative placeholder ("Small Molecules module requires version_pinning or later — see docs"); (c) gate the view's visibility on whether the backing module is deployed.

**Severity:** medium — leaves users with broken-looking UI.

---

## 7. `deploy.sh:35` uses `pip` instead of `pip3`

**Discovery:** 2026-04-21 sandbox deploy attempt failed silently. Investigated logs → pip 20.3.4 from Python 2.7 was attempting to install poetry and failed.

**Gap:** `deploy.sh:35` reads `pip install poetry`. On macOS systems where `/usr/local/bin/python` still resolves to legacy Python 2.7 (a common MacOS artifact from `Python.framework`), the `pip` command is Python 2's pip. Pypi rejects Python 2.7 installs since mid-2021 → poetry install fails → deploy.sh exits non-zero from that pip call but the overall wrapper still reports exit 0 making it look like the deploy succeeded.

**Exact error (Python 2.7's pip):**
```
ERROR: Could not find a version that satisfies the requirement poetry
ERROR: No matching distribution found for poetry
```
Plus a series of `Connection refused` retries because pypi rejects the Py2.7 TLS cipher suite.

**Fix:** change line 35 to `pip3 install poetry`. Alternative: uncomment line 33 to use the official poetry installer (`curl -sSL https://install.python-poetry.org | python3 -`) which is more robust.

**Severity:** HIGH — silently masks deploy failure. Anyone on macOS without a clean pip-to-pip3 shim hits this.

**Compounds with PEP 668 issue:** even after `pip` → `pip3`, modern Homebrew-managed Python (3.11+) declares itself "externally managed" per PEP 668 and refuses system-wide installs. `pip3 install poetry` fails with:
```
error: externally-managed-environment
× This environment is externally managed
```
Under `set -e`, deploy.sh exits at that line. No `.deployed` marker gets created, but stderr gets lost in the noise and the wrapper command may still appear to exit 0 if piped.

**Better fix (adopted 2026-04-21 locally):** use the official `curl -sSL https://install.python-poetry.org | python3 -` installer (the commented-out line 33). Robust regardless of PEP 668 / Homebrew / venvs. Upstream deploy.sh should:

```bash
if ! command -v poetry >/dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org | python3 -
fi
export PATH="$HOME/.local/bin:$PATH"
command -v poetry || { echo "❌ poetry install failed"; exit 1; }
```

(Fixes both pip-vs-pip3 AND PEP 668 AND gives an explicit failure message if poetry really isn't installable.)

**Compounds with:** stale `modules/core/.deployed` markers from prior deploys — when the poetry step fails, the wrapper doesn't clean up, so the next attempt sees `.deployed` present and may short-circuit the dependency check.

---

## 8. `deploy.sh` silently uses DEFAULT Databricks CLI profile

**Discovery:** 2026-04-21 sandbox deploy. With multiple CLI profiles configured, `deploy.sh` fell back to the `DEFAULT` profile (e2-demo-field-eng in our case) — whose refresh token had expired — and failed with auth errors that looked unrelated to the deploy.

**Gap:** `application.env` has `workspace_url=` but no way to specify which auth profile to use. The CLI's default-profile behavior is surprising when the user wants a specific workspace.

**Fix (applied locally 2026-04-21):**

- Add `databricks_profile=<profile-name>` to `application.env`
- `deploy.sh` sources `application.env` early and exports `DATABRICKS_CONFIG_PROFILE=${databricks_profile:-DEFAULT}` before any `databricks` CLI calls
- Falls back to DEFAULT if unset (preserves existing behavior for users who rely on DEFAULT)

**Severity:** medium — surprising failure mode; once diagnosed, simple.

---

## 9. `paste -sd, application.env` floods DAB with undeclared variables

**Discovery:** 2026-04-21 sandbox deploy, v5 attempt. After adding `databricks_profile=fevm-mmt-aws-usw2` to `application.env` (per Gap #8), the DAB bundle validation failed with:
```
Error: variable databricks_profile has not been defined
```

**Root cause:** Every `modules/<module>/deploy.sh` and the outer `deploy.sh` use:
```bash
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")
...
databricks bundle validate --target $TARGET --var="$EXTRA_PARAMS"
```

This pipes EVERY line of `application.env` into DAB as `--var=` args. DAB strictly validates: any `--var=foo=bar` whose `foo` isn't declared in a `variables:` block raises an error. Adding a new entry to `application.env` effectively requires adding a matching DAB variable declaration across all module bundles, OR the deploy explodes.

**Fix (applied locally 2026-04-21 to outer deploy.sh + all 7 module deploy.sh files):**
```bash
EXTRA_PARAMS_GENERAL=$(grep -v '^databricks_profile=' ../../application.env | tr '\n' ',' | sed 's/,$//')
```
Filters out `databricks_profile` before piping. Also swapped `paste -sd,` for `tr + sed` because macOS `paste` rejects stdin with `-s -`.

**Files touched by the patch:** `deploy.sh` (outer), `modules/core/deploy.sh`, `modules/bionemo/deploy.sh`, `modules/disease_biology/deploy.sh`, `modules/parabricks/deploy.sh`, `modules/protein_studies/deploy.sh`, `modules/single_cell/deploy.sh`, `modules/small_molecule/deploy.sh`.

**Better long-term fix for the upstream PR:** either (a) move `databricks_profile` out of `application.env` entirely (into a separate untracked `deploy.profile` file), or (b) declare `databricks_profile` as a DAB variable in every bundle. (a) keeps DAB semantics clean; (b) would formalize the profile-per-deploy idea. (a) is easier, (b) is more future-proof.

**Severity:** MEDIUM — blocks the deploy completely but the error message is clear once you read it.

---

## 10. No single-grant "GWB access" for co-users on a shared workspace

**Discovery:** 2026-04-21, end-of-deploy discussion. Raised by May:

> *"if someone like Srijit sets up his demo it's hard for others to use if they don't give permissions to everyone needing to access different parts of it — maybe should be a global permission setting"*

**Gap:** When one person deploys GWB on a shared workspace (e.g., Srijit on fe-vm-hls-amer), anyone else wanting to USE the app has to be granted permissions on a long list of resources separately:

- Databricks App (run permission)
- Catalog + schema (USE_CATALOG, USE_SCHEMA, SELECT, MODIFY on the schema)
- Secret scope (READ on the scope)
- MLflow experiments (shared + per-user paths)
- Serving endpoints (CAN_QUERY per endpoint)
- Databricks jobs (CAN_VIEW / CAN_RUN per job)
- UC Volumes (READ on sample data volumes)

Right now the deployer has to know every resource + manually grant per user. It's >10 separate grants per added user, spread across Workspace, Catalog, App, Jobs, MLflow, and Secrets UIs. Zero-click default is "I deployed → only I can use the UI properly."

**Existing partial surface:** Settings → Access Management tab exists in the version_pinning app (see `modules/core/app/views/settings.py` — we haven't deep-read it yet; worth a pass). May already handle some of this, unclear if full.

**Ideation for better fix (product-level):**
- **Option A — Group-based:** deploy creates a `gwb_<prefix>_users` Databricks group; every resource grants access to that group; adding a user = add to group. Single pane of glass.
- **Option B — UI-driven:** the Access Management tab lets the deployer add a user by email; the app iterates through all GWB resources and grants the right levels. One click per user.
- **Option C — UC + App-level delegation:** leverage UC's shared-access patterns + app-level run_as to have the app act on behalf of the user for endpoint/job calls, bypassing most direct permission grants.

**Nuance (May, 2026-04-21):** In practice most users have self-contained catalogs (each user/team deploys into their own catalog). That collapses most of the UC permission pain — schema/catalog/volume grants are self-scoped because nobody else is in your catalog. So the permission friction is narrower than full multi-tenancy; it mostly shows up in:
- **App sharing** — someone wants to USE a colleague's already-deployed app (run permission, endpoint CAN_QUERY)
- **First-time Profile setup** — even in your own catalog, each user has to manually grant the app SP "Can Manage" on their MLflow experiment folder (friction at day-1)
- **Shared Docker creds / NGC access** — if team members want to deploy to their own catalogs using shared images

**Severity:** MEDIUM — not a blocker for most SA/POC demos (one deploy = one catalog), but HIGH friction the moment you need collaborative usage. Worth acknowledging in the playbook caveats section.

**For the playbook:** call out that GWB assumes a one-user/one-deploy model by default. If customers want multi-user access, plan for the setup overhead upfront.

---

## 11. Catalogs are metastore-scoped — no cross-region reuse

**Discovery:** 2026-04-21, while planning sandbox-vs-hls-amer deploys. May noted:

> *"my aws uswest2 catalog won't play nice on the hls-amer workspace (in east?)"*

Verified: fevm-mmt-aws-usw2 and fe-vm-hls-amer have different Unity Catalog metastore IDs (`616f89c2-...` vs `c0da88f1-...`). Metastores are region-scoped; a catalog provisioned in one region's metastore is NOT accessible from workspaces attached to a different region's metastore.

**Implication for GWB deploys:**
- Each workspace → needs a catalog provisioned in THAT workspace's metastore
- No cross-region catalog reuse (unless Delta Sharing / Lakehouse Federation is set up, which is a separate workstream)
- Silver lining: accidental cross-workspace catalog collision is impossible — metastores provide clean isolation

**For the playbook (Setup & Caveats tab):** explicitly note that GWB deploy is per-workspace + per-metastore; customers with multi-region footprints need a separate deploy per region unless they accept cross-region data access via Delta Sharing.

**Severity:** LOW as a GWB-specific issue (it's a platform constraint, not a bug). MEDIUM as a "customer expectation" item to cover upfront — multi-region customers will ask.

---

## 12. Parabricks deploy requires "Custom containers" workspace-admin toggle

**Discovery:** 2026-04-21, parabricks module deploy on fevm-mmt-aws-usw2 sandbox. Error:
```
Error: cannot create cluster: Custom containers is turned off for your deployment.
Please contact your workspace administrator to use this feature.
```

**Gap:** The Parabricks module provisions a classic GPU cluster with a custom Docker container (`srijitnair254/parabricks_dbx_amd64:0.1`). Databricks Custom Container support is a workspace-level setting that's OFF by default and requires admin toggle (Admin Settings → Advanced → Custom Containers). Not a GWB bug — a platform prerequisite that isn't called out in GWB's install docs.

**Note — BioNeMo deployed fine on the same workspace.** BioNeMo uses a different compute path (serverless/classic GPU without custom container by default), which is why it bypassed the gate. Parabricks and BioNeMo LOOK similar (both NVIDIA, both Docker-backed) but only one of them needs the Custom Containers feature flag.

**Fix for user:**
- ~~Workspace admin: Admin Settings → Advanced → enable "Custom Containers"~~ *(not possible on fevm-mmt-aws-usw2 — verified 2026-04-21: the toggle is NOT visible on any of Compute, Advanced, Previews tabs at workspace-admin level. Gated at account admin level in this org.)*
- Escalation: request via R2-DB / Opal / Databricks account admin: "enable Databricks Container Services on workspace fevm-mmt-aws-usw2"
- Reference workspace that DOES have it enabled: fe-vm-hls-amer (Srijit's deploy works there — likely was enabled during provisioning)
- Then `rm modules/parabricks/.deployed` (if a stale marker exists) and retry `./deploy.sh parabricks aws`

**Workaround for users without the toggle:**
- Skip `parabricks` module deploy entirely
- Skip Disease Biology → Variant Calling tab (uses parabricks alignment) — pivot to pre-called VCFs for GWAS / VCF Ingestion / Variant Annotation (those use Glow, not Parabricks)
- Bionemo finetune/inference jobs may hit the same gate at runtime (embedded `new_cluster` with Docker image) — untested on gated workspaces

**Architecture note — could parabricks refactor to bionemo's per-job-cluster pattern?**

Structurally yes. Parabricks' `resources/parabricks_cluster.yml` declares a STANDALONE cluster materialized at bundle-deploy time; that's what hits the Container Services gate at `terraform apply`. Refactoring to bionemo-style (cluster inlined as `new_cluster:` per-job) would move cluster creation to job-run time.

BUT: Databricks Container Services gates ANY cluster using a custom `docker_image`, whether standalone or ephemeral. So the refactor just moves the failure from deploy-time to run-time. Under the most common workspace policy, the end state is the same: you can't actually USE parabricks without the feature enabled.

**Speculative maybe:** some org policies gate only persistent/interactive clusters and allow ephemeral job clusters. Worth testing post-Monday as a data point; not a reliable fix.

**Refactor effort (if ever done):** ~1-2 hours YAML + affected jobs (parabricks internal + disease_biology's `gwas_parabricks_alignment`). Candidate improvement if combined with a workspace that confirms the "ephemeral-only" policy hypothesis.

**Fix for upstream (Installation.md / deploy wizard):** add a preflight check that verifies Custom Containers is enabled when the user opts into Parabricks. Early-fail with a clear message instead of mid-deploy cluster-create error.

**Severity:** MEDIUM — blocks parabricks-specific deploys; disease_biology's `gwas_parabricks_alignment` job will likely also fail at runtime for the same reason if the setting stays off.

---

## 13. BioNeMo module deploy doesn't trigger `dbx_bionemo_initial_setup` — missing `bionemo_weights` table

**Discovery:** 2026-04-21 sandbox deploy. After `./deploy.sh bionemo aws` reported ✅ SUCCESS, opening the app's NVIDIA → BioNeMo → Inference tab threw:

```
databricks.sql.exc.ServerOperationError: [TABLE_OR_VIEW_NOT_FOUND]
The table or view `mmt_aws_usw2`.`genesis_workbench`.`bionemo_weights` cannot be found.
```

Traceback bottoms out at `views/nvidia/bionemo_esm.py:179` calling `list_finetuned_weights()` → `SELECT * FROM {catalog}.{schema}.bionemo_weights`.

**Root cause:** The `bionemo_weights` Delta table is created by `modules/bionemo/notebooks/initialize.py` (line 63-66: DROP TABLE IF EXISTS + CREATE TABLE). That notebook is the task in the `dbx_bionemo_initial_setup` Databricks job. The job gets *created* during deploy but isn't *run* by `./deploy.sh bionemo aws` — only `dbx_gwb_initialize_module_job` (which registers the module in the core `settings` table) runs during deploy.

So the table never gets created until someone manually triggers the init job, which breaks the Inference tab on any fresh deploy.

**Other modules likely have the same pattern.** protein_studies, single_cell, small_molecule, disease_biology all have their own `initialize.py` notebooks that create module-specific tables. They may or may not run as part of their deploy. Worth auditing.

**Fix for user (one-time):** trigger the init job manually via CLI or UI:
```bash
databricks jobs run-now <job-id> --profile <profile>
```
Find the job id via `databricks jobs list` (look for `dbx_<module>_initial_setup`).

**Fix for upstream:** `./deploy.sh <module> <cloud>` should call `databricks bundle run dbx_<module>_initial_setup_job` AFTER `dbx_gwb_initialize_module_job` so the module-specific table setup happens automatically. Same pattern as `initialize_core_job` is called inside `modules/core/deploy.sh`.

**Severity:** HIGH — every GWB deploy hits this silently; users experience as "BioNeMo tab is broken" without any indication that a job needs to be run. Each module may need similar audit.

---

## 14. `deploy.sh` lacks post-deploy success check across downstream jobs

**Discovery:** 2026-04-21, while checking failure emails after the cascade deploy. The `./deploy.sh <module> <cloud>` wrapper reports:
```
✅ SUCCESS! Deployment complete.
```
as soon as the initial `bundle deploy` + `initialize_module_job` complete. But the per-module deploy creates MANY downstream jobs (init notebooks, model registrations) — these run asynchronously and can fail silently. The user only discovers failures via email-on-failure notifications (which land much later) or by opening the app and hitting a broken tab.

**Concrete example from 2026-04-21 sandbox deploy:** the wrapper reported SUCCESS for bionemo, but `dbx_bionemo_initial_setup` failed a later step (the `bionemo_weights` table was created but a subsequent cell in initialize.py errored). No signal in deploy output; only the on-failure email alerted.

**Fix (for upstream):** after bundle run `dbx_gwb_initialize_module_job`, the outer `deploy.sh` should either:
- Poll `dbx_<module>_initial_setup_job` AND the module's registration jobs until terminal state, reporting aggregate status, OR
- At minimum, print the Databricks Jobs UI URL + a list of "jobs triggered by this deploy" so the user knows where to look

Could be additive: leave the "✅ SUCCESS" claim as "deploy script complete" but also report "Jobs still running in background: N. Check {workspace}/jobs to monitor."

**Current hit list on the sandbox (2026-04-21):** 3 failures (dbx_bionemo_initial_setup, register_proteina_complexa_gwb, dbx_variant_annotation_initial_setup), 9 still running, 13 success, 13 never triggered. The user had to manually query to discover all of this.

**Severity:** MEDIUM — doesn't block deploy, but "silent success followed by email-triggered failure" is a bad UX and creates confusion about what's actually working. Fix is pure tooling improvement.

**Related:** UX-GAPS #13 (per-module init jobs don't trigger automatically). The "check that init job succeeded" would also have caught #13.

---

## 15. `a@b.com` placeholder for user_email across multiple init/destroy notebooks

**Discovery:** 2026-04-21, diagnosing bionemo init failure:
```
ResourceDoesNotExist: Principal: UserName(a@b.com) does not exist
```

**Gap:** Multiple init/destroy notebooks use `a@b.com` as the default value for a `user_email` widget. When the notebook runs WITHOUT the `user_email` parameter passed in, it defaults to `a@b.com` and tries to grant permissions to that nonexistent principal → job fails.

**Affected files:**
- `modules/parabricks/notebooks/initialize.py:9`
- `modules/core/notebooks/destroy_module.py:6`
- `modules/core/resources/jobs/destroy_module.yml:33`
- `modules/core/library/genesis_workbench/src/genesis_workbench/bionemo.py:56,89`
- (bionemo's initialize.py may pull from one of the above — inferred since the failure surfaces there)

**Fix for user (immediate workaround):** when manually triggering an init job:
```
databricks jobs run-now <job-id> --json '{"job_parameters":{"user_email":"<your-email>"}}'
```

**Fix for upstream:** either (a) make user_email a required parameter (no default) so missing it fails fast with a clear message, or (b) default to pulling the current user from the Databricks context (`dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName()`).

**Ties to UX-GAPS #13 (per-module init not auto-triggered) and #14 (no post-deploy success check).** If the outer `deploy.sh` auto-triggered these init jobs AND passed the right user_email, this whole class of failures disappears.

**Severity:** HIGH — bionemo is unusable until this is either fixed upstream or manually worked around.

## 16. `proteina_complexa` registration notebook: pip install resolves inconsistent versions

**Discovery:** 2026-04-21, `register_proteina_complexa_gwb` job failed with:
```
CalledProcessError: Command '['pip', 'install', '-q', '--find-links',
'https://data.pyg.org/whl/torch-2.7.0+cu126.html',
'torch==2.7.1', 'torch_geometric==2.7.0', 'torch_scatter==2.1.2',
'torch_sparse==0.6.18', 'torch_cluster==1.6.3',
'atomworks', 'jax', 'colabdesign', 'jaxtyping', 'loguru', 'biopandas', ...]'
returned non-zero exit status.
```

**Root cause (probable):** torch 2.7.1 + torch_geometric 2.7.0 + torch_scatter 2.1.2 + torch_cluster 1.6.3 — these version combinations and the cu126 find-links target may not all have available wheels for the runtime's Python/CUDA. `atomworks` + `colabdesign` are research libraries that aren't always in pypi mainline.

**Affected files:**
- `modules/small_molecule/proteina_complexa/proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:42-51` (inline %pip install)
- `modules/small_molecule/proteina_complexa/proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:131` (second install)
- `modules/small_molecule/proteina_complexa/proteina_complexa_v1/notebooks/01_register_proteina_complexa.py:657,673` (colabdesign, torch_scatter versions pinned)

**Fix:** needs upstream version-pinning audit against the actual GPU runtime (14.3 LTS ML GPU). Iterate until a consistent set resolves. This is the kind of thing `version_pinning` branch was supposed to address — likely still in flight.

**Severity:** HIGH for Small Molecule — proteina_complexa is a dependency for multiple Binder Design workflows. Without this model registered, Protein Binder Design / Ligand Binder Design / Motif Scaffolding tabs won't work.

## 17. variant_annotation init has too-short timeout (600s)

**Discovery:** 2026-04-21, `dbx_variant_annotation_initial_setup` failed with "Run timed out".

**Gap:** `modules/disease_biology/variant_annotation/variant_annotation_v1/resources/initial_setup.yml:25` sets `timeout_seconds: 600` (10 minutes). The init probably downloads ClinVar reference data (several GB) — 10 minutes is marginal.

**Fix (one-line YAML):** bump to 3600 or remove the explicit timeout:
```yaml
timeout_seconds: 3600   # was 600
```

**Severity:** MEDIUM — predictable failure on first deploy; re-triggering with a longer timeout (via `--timeout_seconds` override or YAML patch) resolves.

---

## 18. BioNeMo Finetune inputs have no pre-populated example paths

**Discovery:** 2026-04-21, walking NVIDIA → BioNeMo → Finetune tab on the sandbox app. User looking at "Train Data" + "Evaluation Data" inputs — no example paths shown, not obvious what to paste.

**Gap:** In `modules/core/app/views/nvidia/bionemo_esm.py`:
```python
train_data = st.text_input("Train Data (UC Volume Path *.csv):", " ",
    help="A CSV file with `sequence` column and a `target` column")
evaluation_data = st.text_input("Evaluation Data (UC Volume Path *.csv):",
    help="A CSV file with `sequence` column and a `target` column")
```

Train defaults to a LITERAL SPACE CHARACTER (useless); evaluation has no default at all. Help text describes the SCHEMA but doesn't give a PATH.

Compare to other workflows in the same app that pre-populate useful defaults:
- `processing.py` (Scanpy/Rapids) → default h5ad path at `/Volumes/{catalog}/{schema}/raw_h5ad/...` pre-staged
- `admet_safety.py` → 5 example SMILES pre-populated in textarea
- `small_molecules.py` (DiffDock) → example SMILES + example PDB
- `structure_prediction.py` → example protein sequence

BioNeMo Finetune is the outlier. New users literally have no idea where to point the input.

**Fix suggestion (upstream):**
1. Pre-populate sample paths:
   ```python
   _default_train = f"/Volumes/{os.environ.get('CORE_CATALOG_NAME','catalog')}/{os.environ.get('CORE_SCHEMA_NAME','schema')}/bionemo_examples/esm2_train_sample.csv"
   _default_eval  = f"/Volumes/{os.environ.get('CORE_CATALOG_NAME','catalog')}/{os.environ.get('CORE_SCHEMA_NAME','schema')}/bionemo_examples/esm2_eval_sample.csv"
   ```
2. Ship a tiny real `esm2_train_sample.csv` + `esm2_eval_sample.csv` with the module, staged during `dbx_bionemo_initial_setup` (same pattern as Scanpy's CellxGene sample).

So clicking "Start Finetuning" on defaults actually runs end-to-end — customers see a working demo before they touch their own data.

**Update 2026-04-21 — the Inference tab has the same issue, worse:**

```python
inf_data_location        = st.text_input("Data Location:(UC Volume Path *.csv):", "")
inf_sequence_column_name = st.text_input("Sequence Column Name:", "")
inf_result_location      = st.text_input("Result Location: (UC Volume Folder)", "",
                            help="...Please make sure the folder exists.")
```

- `inf_data_location` — empty
- `inf_sequence_column_name` — empty (should default to `"sequence"` since help text says so)
- `inf_result_location` — empty, AND user has to manually pre-create the folder ("Please make sure the folder exists")

**Expanded fix for upstream (both Finetune + Inference):**

1. Ship sample CSVs at `dbx_bionemo_initial_setup` time:
   - `/Volumes/{catalog}/{schema}/bionemo_examples/esm2_train_sample.csv`
   - `/Volumes/{catalog}/{schema}/bionemo_examples/esm2_eval_sample.csv`
   - `/Volumes/{catalog}/{schema}/bionemo_examples/esm2_inference_input.csv`
2. Pre-create the results folder so "folder must exist" is not a user burden:
   - `/Volumes/{catalog}/{schema}/bionemo_examples/results/`
3. Update the Streamlit form defaults:
   ```python
   _catalog = os.environ.get('CORE_CATALOG_NAME', 'catalog')
   _schema  = os.environ.get('CORE_SCHEMA_NAME', 'schema')
   _base    = f"/Volumes/{_catalog}/{_schema}/bionemo_examples"
   
   # Finetune
   train_data       = st.text_input(..., value=f"{_base}/esm2_train_sample.csv", ...)
   evaluation_data  = st.text_input(..., value=f"{_base}/esm2_eval_sample.csv", ...)
   
   # Inference
   inf_data_location        = st.text_input(..., value=f"{_base}/esm2_inference_input.csv", ...)
   inf_sequence_column_name = st.text_input(..., value="sequence", ...)
   inf_result_location      = st.text_input(..., value=f"{_base}/results", ...)
   ```

Result: a new user can click through a finetune AND an inference run on defaults without ever needing to know anything about paths, data format, or column names. Same UX bar as Scanpy/ADMET/DiffDock.

**Severity:** MEDIUM (unchanged) — covers BOTH Finetune and Inference tabs now.

**2026-04-21 refinement — DATA IS STAGED, UI DOESN'T KNOW.** Verified on sandbox: `modules/bionemo/notebooks/initialize.py:134-137` DOWNLOADS the BLAT_ECOLX benchmark dataset during `dbx_bionemo_initial_setup`:
```
/Volumes/{catalog}/{schema}/bionemo/esm2/ft_data/BLAT_ECOLX_Tenaillon2013_metadata_train.csv
/Volumes/{catalog}/{schema}/bionemo/esm2/ft_data/BLAT_ECOLX_Tenaillon2013_metadata_eval.csv
```
Both confirmed present on `mmt_aws_usw2.genesis_workbench.bionemo.esm2.ft_data` after init.

The fix is even simpler than I suggested above — the data exists, the UI just needs to **know where to find it**. One-liner defaults in `bionemo_esm.py`:

```python
_cat  = os.environ.get('CORE_CATALOG_NAME', 'catalog')
_sch  = os.environ.get('CORE_SCHEMA_NAME', 'schema')
_base = f"/Volumes/{_cat}/{_sch}/bionemo/esm2/ft_data"

# Finetune
train_data = st.text_input(..., value=f"{_base}/BLAT_ECOLX_Tenaillon2013_metadata_train.csv", ...)
evaluation_data = st.text_input(..., value=f"{_base}/BLAT_ECOLX_Tenaillon2013_metadata_eval.csv", ...)

# Inference (reuse eval CSV as inference input in demo mode)
inf_data_location = st.text_input(..., value=f"{_base}/BLAT_ECOLX_Tenaillon2013_metadata_eval.csv", ...)
inf_sequence_column_name = st.text_input(..., value="sequence", ...)
inf_result_location = st.text_input(..., value=f"/Volumes/{_cat}/{_sch}/bionemo/esm2/results", ...)
```

Plus `initialize.py` should `os.makedirs(f"/Volumes/{_cat}/{_sch}/bionemo/esm2/results", exist_ok=True)` so the result folder exists out of the box.

No new data shipping required — just UI defaults reflecting what init already staged.

---

*(further entries as we find them)*
