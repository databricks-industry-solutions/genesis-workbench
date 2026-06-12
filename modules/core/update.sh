#!/bin/bash
# update.sh — redeploy the genesis-workbench app without touching the
# settings/models/model_deployments/batch_models tables.
#
# Use this instead of `deploy.sh` for every redeploy on a populated
# install. `deploy.sh` is gated on `.deployed` not existing for the
# destructive initialize_core_job, but the safer thing to do on an
# already-deployed install is to take that path out of reach entirely —
# which is exactly what this script does. The bundle deploy +
# genesis_workbench_app run still re-upload code, refresh the app
# container, regrant permissions, and update wheels in the UC Volume.

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: update <cloud> [--ui-only]"
    echo 'Example: update aws'
    echo '         update aws --ui-only   # skip wheel rebuild, secret refresh, grants, UC volume copy'
    exit 1
fi

CLOUD=$1
UI_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --ui-only) UI_ONLY=true ;;
  esac
done

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp> [--ui-only]"; exit 1 ;;
esac

source module.env
source ../../application.env

# ─── Toolchain check ──────────────────────────────────────────────────────
if ! command -v node >/dev/null 2>&1; then
    echo "🚫 node is required to build the React frontend. Install Node.js 18+ before running update.sh."
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "🚫 npm is required to build the React frontend. Install Node.js (which ships npm) 18+ before running update.sh."
    exit 1
fi

if [ "$UI_ONLY" = "false" ]; then
    echo ""
    echo "▶️ Refreshing secret scope values"
    echo ""

    if databricks secrets list-scopes | grep -qw "$secret_scope_name"; then
        echo "Scope $secret_scope_name already exists."
    else
        databricks secrets create-scope "$secret_scope_name"
        echo "Scope $secret_scope_name created."
    fi

    databricks secrets put-secret $secret_scope_name core_catalog_name --string-value $core_catalog_name
    databricks secrets put-secret $secret_scope_name core_schema_name --string-value $core_schema_name
    databricks secrets put-secret $secret_scope_name dev_user_prefix --string-value "${dev_user_prefix:-}"
else
    echo ""
    echo "▶️ --ui-only: skipping secret scope refresh"
    echo ""
fi

# Wheel build runs every time — the bundle's initialize_core_job.yml references
# library/genesis_workbench/dist/*.whl, so `databricks bundle deploy` fails if
# that file isn't present. poetry build is fast enough that there's no value in
# trying to cache it for --ui-only.
echo ""
echo "▶️ Building genesis_workbench library wheel"
echo ""

cd library/genesis_workbench
poetry build
cd ../../

WHEEL=$(ls library/genesis_workbench/dist/*.whl | head -1)
if [ -z "$WHEEL" ]; then
    echo "🚫 No wheel built — check 'poetry build' output above."
    exit 1
fi
WHEEL_NAME=$(basename "$WHEEL")
mkdir -p app/backend/lib
rm -f app/backend/lib/genesis_workbench-*.whl
cp "$WHEEL" app/backend/lib/
echo "Staged $WHEEL_NAME → app/backend/lib/"

# Same staging for the sibling MCP app (mcp-genesis-workbench). The wheel is
# NOT gitignored (gitignored files are excluded from the DAB sync) — it is
# force-uploaded via databricks.yml sync.include and deleted in cleanup below.
mkdir -p mcp_app/backend/lib
rm -f mcp_app/backend/lib/genesis_workbench-*.whl
cp "$WHEEL" mcp_app/backend/lib/
echo "Staged $WHEEL_NAME → mcp_app/backend/lib/"

# Guarantee the staged wheels + dist are removed on ANY exit — success, a failed
# `bundle deploy/run`, or Ctrl-C. The wheel is intentionally NOT gitignored (that
# would exclude it from the DAB sync), so it must never be left behind to be
# accidentally committed. A trap (vs. a trailing cleanup step) ensures this runs
# even when an earlier step errors out.
_cleanup_artifacts() {
    rm -rf library/genesis_workbench/dist
    rm -f app/backend/lib/genesis_workbench-*.whl
    rm -f mcp_app/backend/lib/genesis_workbench-*.whl
}
trap _cleanup_artifacts EXIT

if ! grep -wq "$WHEEL_NAME" app/requirements.txt; then
    echo "⚠️  app/requirements.txt does not reference $WHEEL_NAME — update it to match the pyproject version."
    exit 1
fi
if ! grep -wq "$WHEEL_NAME" mcp_app/requirements.txt; then
    echo "⚠️  mcp_app/requirements.txt does not reference $WHEEL_NAME — update it to match the pyproject version."
    exit 1
fi

echo ""
echo "▶️ Building React frontend"
echo ""

cd app/frontend
if [ ! -d node_modules ]; then
    echo "node_modules missing — running npm install"
    npm install
fi
npm run build
cd ../../

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
fi

echo "Extra Params: $EXTRA_PARAMS"

echo ""
echo "▶️ Validating bundle"
echo ""

databricks bundle validate --target $TARGET --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Deploying bundle (target=$TARGET)"
echo ""

# IMPORTANT: no initialize_core_job here. That job drops + recreates
# settings/models/model_deployments/batch_models. Use `deploy.sh` only
# for a first-time install of an empty workspace; never on a populated one.
databricks bundle deploy --target $TARGET --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Copying libraries to UC Volume"
echo ""

# Copied right after `bundle deploy` (which ensures the `libraries` Volume exists)
# and BEFORE grant_app_permissions_job, which %pip-installs the genesis_workbench
# wheel from this Volume. Copying it afterwards would install a stale wheel, or
# fail with "ModuleNotFoundError: No module named 'genesis_workbench'" on a fresh
# Volume. Each module's notebooks also %pip install from here.
# Copy the current wheel FIRST, then prune older versions — so the Volume always
# holds at least the current wheel. (Removing-then-copying left a window where the
# Volume had NO genesis_workbench wheel; an ai_canvas orchestrator run launched in
# that window failed at import with "ModuleNotFoundError: No module named
# 'genesis_workbench'" — its scan globs "genesis_workbench*" here and found nothing.)
for file in library/genesis_workbench/dist/*.whl; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Copying $filename to dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename"
    databricks fs cp library/genesis_workbench/dist/$filename dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename --overwrite
  fi
done

# Prune any OTHER genesis_workbench wheels (stale versions), keeping the one just
# copied — so the orchestrator's glob stays unambiguous and the Volume is never empty.
for vfile in $(databricks fs ls "dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries" 2>/dev/null | grep -E '^genesis_workbench-.*\.whl$'); do
  if [ "$vfile" != "$WHEEL_NAME" ]; then
    echo "Removing stale wheel $vfile from UC Volume"
    databricks fs rm "dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$vfile" || true
  fi
done

for file in library/glow/*; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Copying $filename to dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename"
    databricks fs cp "$file" dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename --overwrite
  fi
done

echo ""
echo "▶️ Publishing the node catalog (Vortex/MCP single source of truth)"
echo "    (writes the node_catalog table from the wheel's built-in nodes)"
echo ""

# Must run after the wheel is on the UC Volume above — the notebook %pip-installs it.
databricks bundle run --target $TARGET publish_node_catalog_job --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Deploying UI Application (genesis-workbench)"
echo ""

databricks bundle run --target $TARGET genesis_workbench_app --var="$EXTRA_PARAMS"

mcp_app_name=mcp-genesis-workbench

if [ "$UI_ONLY" = "false" ]; then
    echo ""
    echo "▶️ Deploying MCP Application ($mcp_app_name)"
    echo ""

    databricks bundle run --target $TARGET mcp_genesis_workbench_app --var="$EXTRA_PARAMS"

    echo ""
    echo "▶️ Granting app service principals access to catalog"
    echo ""

    app_sp_id=$(databricks apps get $app_name --output json | jq -r '.service_principal_client_id')
    mcp_app_sp_id=$(databricks apps get $mcp_app_name --output json | jq -r '.service_principal_client_id')
    echo "App service principals: $app_sp_id (UI), $mcp_app_sp_id (MCP)"

    for sp in "$app_sp_id" "$mcp_app_sp_id"; do
        databricks grants update catalog $core_catalog_name --json "{\"changes\": [{\"principal\": \"$sp\", \"add\": [\"USE_CATALOG\"]}]}"
        databricks grants update schema $core_catalog_name.$core_schema_name --json "{\"changes\": [{\"principal\": \"$sp\", \"add\": [\"USE_SCHEMA\", \"SELECT\", \"MODIFY\"]}]}"
    done

    echo "Catalog and schema permissions granted (both apps)."

    echo ""
    echo "▶️ Granting app permissions for endpoints, jobs, volumes, models (both apps)"
    echo ""
    # NOTE: the genesis_workbench wheel is copied to the UC Volume earlier (right
    # after `bundle deploy`) because this serverless job %pip-installs it from there.
    # app_names (colon-separated) makes the grant notebook cover both app SPs.
    databricks bundle run --target $TARGET grant_app_permissions_job --var="$EXTRA_PARAMS,app_names=$app_name:$mcp_app_name"
else
    echo ""
    echo "▶️ --ui-only: skipping catalog grants, app-permissions job, and UC Volume library copy"
    echo ""
fi

# Clean up local build artifacts — ALWAYS, in both the full and --ui-only paths.
# The wheel is staged on every run (the staging block above is unconditional
# because the DAB sync force-uploads it from app/backend/lib/), so it must be
# removed afterwards in BOTH paths or it lingers in the working tree / risks
# being committed. It is intentionally NOT gitignored: gitignored files are
# excluded from the DAB sync, so the wheel must be present at deploy time and
# deleted here instead.
echo ""
echo "▶️ Cleaning up local build artifacts (also guaranteed via EXIT trap)"
echo ""
_cleanup_artifacts

# Note: NOT writing .deployed here — update.sh is for redeploys, the
# .deployed marker is owned by deploy.sh.
echo ""
if [ "$UI_ONLY" = "true" ]; then
    echo "✅ UI-only update complete. Frontend redeployed; settings/models tables, wheels, grants, and UC volume untouched."
else
    echo "✅ Update complete. App redeployed; settings/models tables untouched."
fi
