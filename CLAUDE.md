# Genesis Workbench — Claude Code Entry Point

When a user in this repo asks for help with deployment, installation, development, troubleshooting, or workflow usage, load the relevant skill below **before** answering. These files contain authoritative step-by-step guidance, error auto-handlers, and validation commands that supersede general knowledge.

## Skill index

| User intent | Skill file |
|---|---|
| "deploy Genesis Workbench", "install GWB", "set up on new workspace", "run deploy.sh" | [`claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md`](claude_skills/SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md) |
| "destroy GWB", "tear down genesis workbench", "uninstall GWB", "remove all modules" | [`claude_skills/SKILL_GENESIS_WORKBENCH_DESTROY_WIZARD.md`](claude_skills/SKILL_GENESIS_WORKBENCH_DESTROY_WIZARD.md) |
| Reference docs for installation / config / deployment mechanics | [`claude_skills/SKILL_GENESIS_WORKBENCH_INSTALLATION.md`](claude_skills/SKILL_GENESIS_WORKBENCH_INSTALLATION.md) |
| Overview of GWB — modules, capabilities, architecture | [`claude_skills/SKILL_GENESIS_WORKBENCH.md`](claude_skills/SKILL_GENESIS_WORKBENCH.md) |
| Deploy failures, registration errors, endpoint issues, runtime errors | [`claude_skills/SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md`](claude_skills/SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md) |
| Adding new models / workflows / UI tabs | [`claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md`](claude_skills/SKILL_GENESIS_WORKBENCH_DEVELOPMENT.md) |
| Adding a long-running batch workflow (form → job → MLflow → search past runs → result dialog) | [`claude_skills/SKILL_GENESIS_WORKBENCH_BATCH_WORKFLOW_PATTERN.md`](claude_skills/SKILL_GENESIS_WORKBENCH_BATCH_WORKFLOW_PATTERN.md) |
| Using the GWB UI — how each tab works, inputs/outputs | [`claude_skills/SKILL_GENESIS_WORKBENCH_WORKFLOWS.md`](claude_skills/SKILL_GENESIS_WORKBENCH_WORKFLOWS.md) |

## Quick start — deploy

From this repo root:
```bash
./deploy.sh core <aws|azure|gcp>     # must be first; creates modules/core/.deployed
./deploy.sh <module> <cloud>          # each additional module, one at a time
```

Deploy modules **one at a time** and wait for each module's first post-deploy job to reach `RUNNING` before launching the next — this serializes GPU cluster-create and surfaces quota issues one module at a time. Full rationale + polling commands in the deploy wizard skill above.

## Post-deploy: monitor jobs for failures (do NOT trust the "✅ SUCCESS" banner)

`./deploy.sh`'s `✅ SUCCESS! Deployment complete` only means the bundle deployed and the script's poll returned — it is **not** proof the module works. Register/deploy/data-prep jobs run **asynchronously** and can FAIL (or hit an internal 60-min wait timeout) *after* `deploy.sh` exits. **After every module deploy, monitor ALL job runs and check for non-SUCCESS results, and verify serving endpoints reach READY:**

```bash
# any non-SUCCESS job runs (recent)
databricks jobs list-runs --limit 50 --output json | jq -r '.runs[] | select(.state.result_state and .state.result_state!="SUCCESS") | "FAILED  \(.run_name)  run=\(.run_id)"'
# endpoint readiness
databricks serving-endpoints list --output json | jq -r '.endpoints[] | "\(.name)\t\(.state.ready)"'
```

**Known false-positive:** a slow GPU `deploy_model_job` (e.g. scGPT ~95 min) exceeds the **3600s internal wait** → the calling `update_model_catalog_*` task (or `deploy.sh`) reports `TimeoutError: Job run … did not complete within 3600 seconds`, but the deploy usually **SUCCEEDED** — confirm via the **endpoint READY state**, not the task result. Full failure catalog (genmol SP defaults, scimilarity volume-delete, unwired `gene_sequences`, cellxgene Volume-write errno 95, the wait-timeout false-positive) is in the deploy-wizard skill's **Post-deploy verification** section and `CHANGELOG.md` → Unreleased.
