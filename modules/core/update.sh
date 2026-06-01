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
    echo "Usage: update <cloud>"
    echo 'Example: update aws'
    exit 1
fi

CLOUD=$1

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp>"; exit 1 ;;
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

echo ""
echo "▶️ Building genesis_workbench library wheel"
echo ""

cd library/genesis_workbench
poetry build
cd ../../

# Copy the freshly-built wheel into the React backend's lib/. The wheel name
# (genesis_workbench-X.Y.Z-py3-none-any.whl) is pinned in app/requirements.txt,
# so a version bump in pyproject.toml requires updating requirements.txt too.
WHEEL=$(ls library/genesis_workbench/dist/*.whl | head -1)
if [ -z "$WHEEL" ]; then
    echo "🚫 No wheel built — check 'poetry build' output above."
    exit 1
fi
WHEEL_NAME=$(basename "$WHEEL")
mkdir -p app/backend/lib
# Remove stale wheel(s) so the bundle sync doesn't upload obsolete versions.
rm -f app/backend/lib/genesis_workbench-*.whl
cp "$WHEEL" app/backend/lib/
echo "Staged $WHEEL_NAME → app/backend/lib/"

if ! grep -wq "$WHEEL_NAME" app/requirements.txt; then
    echo "⚠️  app/requirements.txt does not reference $WHEEL_NAME — update it to match the pyproject version."
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
echo "▶️ Deploying UI Application (genesis-workbench)"
echo ""

databricks bundle run --target $TARGET genesis_workbench_app --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Granting app service principal access to catalog"
echo ""

app_sp_id=$(databricks apps get $app_name --output json | jq -r '.service_principal_client_id')
echo "App service principal: $app_sp_id"

databricks grants update catalog $core_catalog_name --json "{\"changes\": [{\"principal\": \"$app_sp_id\", \"add\": [\"USE_CATALOG\"]}]}"
databricks grants update schema $core_catalog_name.$core_schema_name --json "{\"changes\": [{\"principal\": \"$app_sp_id\", \"add\": [\"USE_SCHEMA\", \"SELECT\", \"MODIFY\"]}]}"

echo "Catalog and schema permissions granted."

echo ""
echo "▶️ Granting app permissions for endpoints, jobs, volumes, models"
echo ""
databricks bundle run --target $TARGET grant_app_permissions_job --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Copying libraries to UC Volume"
echo ""

for file in library/genesis_workbench/dist/*.whl; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Copying $filename to dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename"
    databricks fs cp library/genesis_workbench/dist/$filename dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename --overwrite
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
echo "▶️ Cleaning up local build artifacts"
echo ""
rm -rf library/genesis_workbench/dist

# Note: NOT writing .deployed here — update.sh is for redeploys, the
# .deployed marker is owned by deploy.sh.
echo ""
echo "✅ Update complete. App redeployed; settings/models tables untouched."
