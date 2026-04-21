---
name: genesis-workbench-deploy-monitor
description: Live monitor Genesis Workbench job runs during or after a deploy. Polls all GWB-tagged Databricks jobs, surfaces failures with diagnostic traces as they happen, and matches common failure patterns to proposed fixes. Use when the user says "is my deploy working?", "check job status", "why did deploy fail?", "watch jobs", or after any `./deploy.sh` run.
---

# Genesis Workbench Deploy Monitor Skill

Watch GWB job runs in real time, diagnose failures as they surface, and suggest fixes based on known patterns.

## When to use

- User kicked off `./deploy.sh <module> <cloud>` and wants ongoing visibility
- User reports a failure email and wants to understand what broke
- After a deploy, user wants to confirm all downstream jobs actually succeeded (not just that `deploy.sh` reported ✅)
- Before a live demo, user wants to verify all serving endpoints + registration jobs are healthy
- User asks any variant of "is it working?", "what's stuck?", "which jobs failed?"

Do NOT use for initial deployment setup — that's `genesis-workbench-deploy-wizard`. Do NOT use for static troubleshooting recipes — that's `genesis-workbench-troubleshooting`.

## Pre-flight (run first)

```bash
# Verify CLI profile is valid + GWB jobs exist
databricks current-user me --profile <profile>
databricks jobs list --profile <profile> --output json | \
  python3 -c "import json,sys; print(sum(1 for j in json.load(sys.stdin) if j.get('settings',{}).get('tags',{}).get('application')=='genesis_workbench'))"
```

Report the count back to the user before starting the watch.

## Invocation patterns

**Continuous polling (default — leaves a terminal running):**
```bash
python3 scripts/watch_gwb_jobs.py --profile <profile> --interval 30
```

**One-shot snapshot (for a quick status check):**
```bash
python3 scripts/watch_gwb_jobs.py --profile <profile> --once
```

**Scoped to one module:**
```bash
python3 scripts/watch_gwb_jobs.py --profile <profile> --module bionemo --once
```

**With full error traces on failures (default shows summaries):**
```bash
python3 scripts/watch_gwb_jobs.py --profile <profile> --once --verbose
```

## Output format

The script emits a snapshot per poll, with three sections:

1. **Header** — timestamp + counts by state (🟢 success / 🟡 running / 🔴 failed / ⚪ no-run-yet)
2. **Transitions** — state changes since the previous poll (new successes, new failures, new RUNNING jobs)
3. **Details** — failure diagnostics (error + last 5 trace lines per failed task) and currently-running jobs

Exit codes: 0 if no failures, 1 if any failures, 2 if CLI/auth error.

## Common failure patterns + auto-handlers

When the monitor surfaces a failure, match the error message to one of these known patterns and propose the corresponding fix. Cross-reference `docs/deployments/<workspace>/UX-GAPS.md` if present.

### `ResourceDoesNotExist: Principal: UserName(a@b.com) does not exist`

**Cause:** Init/destroy notebook defaults `user_email` to `a@b.com` placeholder when not passed. Affects bionemo/parabricks/destroy jobs. See UX-GAPS entry #15.

**Fix:** Retrigger the job with the actual user_email:
```bash
databricks jobs run-now \
  --json '{"job_id":<job-id>,"job_parameters":{"user_email":"<user@databricks.com>"}}' \
  --profile <profile>
```

### `TABLE_OR_VIEW_NOT_FOUND: \`<catalog>\`.\`<schema>\`.\`bionemo_weights\``

**Cause:** `dbx_bionemo_initial_setup` job (which runs `initialize.py` to create the table) didn't run as part of the main deploy. See UX-GAPS entry #13.

**Fix:** Trigger the init job explicitly with user_email param (combined with a@b.com fix above). Same pattern applies to any per-module init whose tables are missing — check `modules/<module>/notebooks/initialize.py` for DDL + for which `dbx_<module>_initial_setup` job runs it.

### `cannot create cluster: Custom containers is turned off for your deployment`

**Cause:** Workspace-admin setting blocks custom-container clusters. Parabricks hits this at bundle-deploy (standalone cluster); BioNeMo finetune/inference would hit it at runtime. See UX-GAPS entry #12.

**Fix:** Workspace admin must enable Databricks Container Services at Admin Settings → Compute → Container Services. After the toggle:
```bash
rm modules/parabricks/.deployed
./deploy.sh parabricks <cloud>
```

### `Run timed out`

**Cause:** Task `timeout_seconds` is too short for the work (e.g., `dbx_variant_annotation_initial_setup` defaults to 600s but downloads ClinVar which takes longer). See UX-GAPS entry #17.

**Fix:** Bump timeout in the module's YAML (e.g., `modules/disease_biology/variant_annotation/variant_annotation_v1/resources/initial_setup.yml:25` from `timeout_seconds: 600` to `3600`), redeploy the module, retrigger the init.

### `CalledProcessError: Command '['pip', 'install', ...]' returned non-zero exit status`

**Cause:** Version-pinning issues in a registration notebook's `%pip install` stanza. Common in proteina_complexa + other ML-heavy registration notebooks. See UX-GAPS entry #16.

**Fix:** Not trivially fixable locally — requires upstream version-pin audit. Workaround: skip that specific model / demo using a different workflow. Log as improvement PR candidate.

### `openpgp: key expired`

**Cause:** Databricks CLI's embedded Terraform signature verification fails on expired HashiCorp PGP key.

**Fix:** Install local Terraform (`brew tap hashicorp/tap && brew install hashicorp/tap/terraform`), ensure `deploy.sh` exports `DATABRICKS_TF_EXEC_PATH=$(command -v terraform)` + `DATABRICKS_TF_VERSION=$(terraform version -json | jq -r .terraform_version)`. See UX-GAPS entry #1.

### `Databricks CLI version constraint not satisfied. Required: >=0.295.*, current: 0.X.Y`

**Fix:** Upgrade CLI: `brew upgrade databricks`. See UX-GAPS entry #3.

### `externally-managed-environment` / `PEP 668`

**Cause:** Homebrew Python 3.11+ rejects `pip install poetry` in system context. Affects `deploy.sh` if not already using curl-installer. See UX-GAPS entry #7.

**Fix:** Replace `pip install poetry` in deploy.sh with:
```bash
if ! command -v poetry >/dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org | python3 -
fi
export PATH="$HOME/.local/bin:$PATH"
```

## When no match is found

If an error doesn't match the known patterns above:

1. Fetch the full trace: `databricks jobs get-run-output <task-run-id> --profile <profile>`
2. Add a new UX-GAPS entry in `docs/deployments/<workspace>/UX-GAPS.md` with discovery context, root cause (when known), and a proposed fix
3. Link to the GitHub issue or commit that introduces the root cause if it's identifiable

## Pairs with

- `genesis-workbench-deploy-wizard` — kicks deploys off
- `genesis-workbench-troubleshooting` — static recipes for well-known individual failures
- `genesis-workbench-installation` — reference for what deploy.sh does and what settings are needed
- `docs/deployments/<workspace>/UX-GAPS.md` — running log of gaps + proposed fixes (file-per-deployment)

## Instructions (for Claude)

1. When user says "check my jobs" / "watch deploy" / "why did X fail?" — invoke this skill
2. Run the pre-flight (`databricks current-user me` + count GWB jobs)
3. Offer the user `--once` vs continuous polling; default to `--once` unless they're in the middle of an active deploy
4. On failures found, match error messages against the common patterns above and propose the named fix
5. If the fix involves modifying local files (e.g., YAML timeout bump), show the exact edit + offer to apply it
6. If the fix involves re-triggering a job, show the `databricks jobs run-now --json` command + offer to run it
7. After proposing/applying fixes, re-run the monitor with `--once` to confirm resolution
8. Log any NEW failure pattern not in this skill's recipes as a UX-GAPS entry in the workspace-specific doc
