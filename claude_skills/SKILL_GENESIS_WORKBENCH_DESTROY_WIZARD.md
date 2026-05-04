---
name: genesis-workbench-destroy-wizard
description: Interactive guided tear-down of Genesis Workbench from a Databricks workspace. Walks the user through pre-flight checks, confirms each module destroy, runs ./destroy.sh in the required order (core LAST), verifies completion, and points out resources that survive destroy (Delta tables, VS endpoints when cleanup is disabled).
---

# Genesis Workbench Destroy Wizard

Drive an interactive, validated destroy of [Genesis Workbench](https://github.com/databricks-industry-solutions/genesis-workbench) from a Databricks workspace. Counterpart to `SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md`. This skill enforces the **core-must-be-destroyed-last** ordering rule, confirms every destructive action with the user, and surfaces resources that the destroy scripts do not clean up automatically.

Use this skill whenever the user says things like "destroy Genesis Workbench", "tear down GWB", "uninstall genesis workbench", "remove all modules", "rip out everything", or is sitting in a `genesis-workbench` repo and asks to clean up.

## Pre-flight checks (run first, in parallel)

```bash
databricks --version
databricks current-user me              # confirm auth and which workspace
databricks auth describe | grep Host    # show host explicitly
ls modules/*/.deployed 2>/dev/null      # which modules have deployment markers
databricks apps list | grep -i genesis  # is the GWB app currently deployed?
```

If `databricks current-user me` 401s, instruct the user to re-auth: `databricks auth login --host <workspace_url>` in their own terminal. Don't auto-run interactive auth.

If no `.deployed` markers exist, there's likely nothing to destroy via this flow — ask the user to confirm what they actually want torn down before proceeding (workspace-only resources may need manual cleanup via UI/CLI).

## The hard ordering rule — core LAST

The repo's root `destroy.sh:23-37` enforces this with a dependency check:

```bash
if [[ "$CWD" == "core" ]]; then
    echo "Checking for dependencies"
    find modules -type d | while read -r dir; do
        if [[ "$(basename "$dir")" == "core" ]]; then continue; fi
        if [[ -e "$dir/.deployed" ]]; then
            echo "🚫 Deployment exist in $dir. Cannot remove core module"
            exit 1
        fi
    done
fi
```

So if any non-core module still has a `.deployed` marker, attempting `./destroy.sh core <cloud>` will hard-fail. Always destroy non-core modules first. Recommended order (mirrors deploy in reverse for predictability — but any non-core order works):

1. parabricks
2. bionemo
3. small_molecule
4. single_cell
5. protein_studies
6. disease_biology
7. **core** (last)

## Per-module destroy command

Single canonical entry point:

```bash
./destroy.sh <module> <cloud>
```

`<cloud>` ∈ `aws | azure | gcp`. The root script:
1. Prompts `Do you wish to continue? (y/n)` (`destroy.sh:19`). When invoking from a Claude bash session, pipe `echo y |` to auto-consent — but **always confirm with the user with AskUserQuestion before piping**.
2. For non-core modules: enforces the dependency check above.
3. `cd modules/<module>` and calls the module-level `destroy.sh`, which runs `databricks bundle destroy --target prod_<cloud> --auto-approve`. This removes bundle-declared resources only.
4. For non-core modules, then runs `destroy_module_job` in core via `databricks bundle run`, which executes `modules/core/notebooks/destroy_module.py`. That notebook deletes MLflow serving endpoints registered in the GWB `models` table, optionally deletes per-module Vector Search resources (see "Survivors" below), and soft-deletes module rows.

Aggregator modules (`single_cell`, `protein_studies`, `small_molecule`, `disease_biology`) loop through their submodules in their own `destroy.sh` and call each leaf submodule's `destroy.sh` in turn. Atomic modules (`bionemo`, `parabricks`, `core`) destroy directly.

## Conversational flow

Before destroying anything, present the inventory and confirm scope using `AskUserQuestion`. Phrase questions in singular-future ("Destroy parabricks now?") so each step is its own decision.

For each module in the recommended order:

1. State what's about to be torn down (jobs, volumes, app reference, etc.) and the workspace Jobs URL the user can watch.
2. Confirm with the user via `AskUserQuestion` (single Yes/No).
3. Run:
   ```bash
   cd <repo-root>
   echo y | ./destroy.sh <module> <cloud>
   ```
4. After it returns, verify:
   - `modules/<module>/.deployed` no longer exists.
   - For non-core: the `destroy_module_job` run finished SUCCESS — fetch via `databricks jobs list-runs --job-id <id> --limit 1`.
5. Move to the next module.

After the last non-core module is gone, kick off `./destroy.sh core <cloud>`. The dependency check passes (no `.deployed` markers remaining), and core's bundle destroy removes the app and the rest of core's resources.

## What gets deleted vs. what survives

| Resource | Created by | Destroy behavior |
|---|---|---|
| Bundle-declared jobs (`register_*`, `dbx_gwb_*`, etc.) | bundle YAML | Deleted by `bundle destroy` |
| Bundle-declared UC volumes (model caches) | bundle YAML | Deleted by `bundle destroy` |
| Notebooks under `.bundle/.../files/...` | bundle sync | Deleted by `bundle destroy` |
| Bundle-declared dashboards / experiments | bundle YAML | Deleted by `bundle destroy` |
| `databricks_app.genesis_workbench_app` | core bundle YAML | Deleted by core's `bundle destroy` |
| MLflow serving endpoints registered in GWB `models` table | runtime, by `register_*` notebooks | Deleted by `destroy_module.py` calling `delete_endpoint(...)` |
| GWB `models` rows | catalog | Soft-only: `is_active='false'`, registry rows kept |
| GWB `settings` rows for the module | catalog | Hard-deleted by `destroy_module.py` |
| **Vector Search endpoints** (`gwb_scimilarity_vs_endpoint`, `gwb_sequence_search_vs_endpoint`) | runtime, 06c / 04 notebooks | **Conditional** — deleted only when the cleanup block in `destroy_module.py:80-124` is uncommented. **Currently commented out** — they linger and continue to bill. |
| **Vector Search indexes** (`<catalog>.<schema>.scimilarity_cell_index`, `sequence_embedding_index`) | runtime, 06c / 04 | Same conditional behavior |
| Source Delta tables holding embeddings (`scimilarity_cells`, `sequence_db`, `sequence_embeddings`) | runtime, 06b / 02-03 | **Never deleted** by destroy (intentional — storage is cheap and rebuilds are expensive) |
| Registered MLflow model versions in UC (`models:/SCimilarity_*`, etc.) | runtime, `register_*` | **Never deleted** (only soft-flagged inactive in GWB `models`) |
| Secret scope `genesis_workbench_secret_scope` | `deploy.sh` / `update.sh` (not the bundle) | **Never deleted** — survives destroy. Drop manually with `databricks secrets delete-scope <name>` if you want a clean slate. |
| UC catalog `<core_catalog>` | external (created during deploy wizard or pre-existing) | **Never deleted** |

Before triggering destroy, surface the conditional row to the user: tell them whether VS resources will or won't be cleaned up by the run, based on whether the cleanup block in `destroy_module.py` is currently active.

## Manual cleanup of survivors (after destroy completes)

If VS cleanup was disabled and the user wants a fully clean slate:

```bash
# Indexes first, then endpoints — endpoint deletion fails while it has live indexes
databricks vector-search-indexes delete-index <catalog>.<schema>.scimilarity_cell_index
databricks vector-search-indexes delete-index <catalog>.<schema>.sequence_embedding_index
databricks vector-search-endpoints delete-endpoint gwb_scimilarity_vs_endpoint
databricks vector-search-endpoints delete-endpoint gwb_sequence_search_vs_endpoint
```

Other optional clean-slate steps:

```bash
# Drop the secret scope (only if no other GWB redeploy is planned soon)
databricks secrets delete-scope genesis_workbench_secret_scope

# Hard-delete the embedding Delta tables (the user has previously preferred to keep these)
# Pass --warehouse-id from application.env
databricks api post /api/2.0/sql/statements --json '{"warehouse_id":"<id>","statement":"DROP TABLE IF EXISTS <catalog>.<schema>.scimilarity_cells","wait_timeout":"30s"}'
databricks api post /api/2.0/sql/statements --json '{"warehouse_id":"<id>","statement":"DROP TABLE IF EXISTS <catalog>.<schema>.sequence_db","wait_timeout":"30s"}'
databricks api post /api/2.0/sql/statements --json '{"warehouse_id":"<id>","statement":"DROP TABLE IF EXISTS <catalog>.<schema>.sequence_embeddings","wait_timeout":"30s"}'

# Hard-delete soft-flagged MLflow registered models (use with caution — destroys model artifacts)
databricks api post /api/2.0/mlflow/registered-models/delete --json '{"name":"<full_model_name>"}'
```

Always confirm with the user before any of these — they're not part of the standard destroy flow.

## Error auto-handlers

| Failure signal | Action |
|---|---|
| `🚫 Deployment exist in <dir>. Cannot remove core module` | Read which `<dir>` triggered it; tell user that module still needs `./destroy.sh <module> <cloud>` before core can be torn down. |
| `bundle destroy ... no value assigned to required variable <name>` | Same `--var=` assembly pattern as the deploy wizard. Build `EXTRA_PARAMS` from `application.env` + `<cloud>.env` + `modules/<module>/module_<cloud>.env` (if present) and pass `--var="$EXTRA_PARAMS"`. |
| `Catalog '<name>' is not accessible in current workspace` during `destroy_module_job` | Catalog binding lost. Re-bind with: `databricks api patch /api/2.1/unity-catalog/bindings/catalog/<name> --json '{"add":[{"workspace_id":<id>,"binding_type":"BINDING_TYPE_READ_WRITE"}]}'`. Alternatively `databricks catalogs update <name> --isolation-mode OPEN`. |
| `databricks current-user me` 401 | Tell user to re-auth: `databricks auth login --host <workspace_url>` in their own terminal. |
| `bundle destroy` complains about an in-flight job run on a resource it's deleting | Wait for the run to finish, or cancel via `databricks jobs cancel-all-runs --job-id <id>`. Confirm with user before cancelling. |
| `delete_endpoint` errors with "endpoint already deleted" | Safe to ignore — `destroy_module.py` already wraps the call in try/except. |
| `delete_index` / `delete_endpoint` (VS) errors with `BadRequest: index has live syncs` | Wait for the in-flight pipeline to finish, or call `databricks pipelines stop <pipeline_id>` first. |

## Post-destroy

Print:
- Which modules are now gone (`.deployed` markers absent).
- Which resources survived intentionally (Delta tables, MLflow registry rows, secret scope, catalog).
- Which resources need manual cleanup (VS endpoints/indexes if the cleanup block is disabled).
- Pointer to the **deploy wizard** (`SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md`) for redeploying.

## When to use this skill

- User says "destroy Genesis Workbench", "tear down GWB", "uninstall", "remove all modules", "clean up the workspace".
- User has hit a destroy failure and wants guided recovery — resume at the appropriate step.
- User wants a partial tear-down (e.g., only `single_cell`) — apply only the relevant phase, do **not** auto-cascade to core unless explicitly asked.

## When NOT to use this skill

- User wants to redeploy / install / update — use `SKILL_GENESIS_WORKBENCH_DEPLOY_WIZARD.md`.
- User wants to fix a single failed task in a running registration job — use `SKILL_GENESIS_WORKBENCH_TROUBLESHOOTING.md`.
- User wants to drop only Delta tables or only VS resources without removing modules — handle ad-hoc, do not invoke this end-to-end flow.

## Related skills

- `genesis-workbench-deploy-wizard` — install / redeploy.
- `genesis-workbench-installation` — reference docs for deploy/destroy mechanics.
- `genesis-workbench-troubleshooting` — recovery recipes for failed registration runs.
- `databricks-authentication` — fix DEFAULT profile / re-auth.
