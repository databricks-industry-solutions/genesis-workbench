#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: deploy <cloud>"
    echo 'Example: deploy aws'
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
# The React frontend needs Node + npm. Fail early with a clear message if the
# operator runs deploy.sh on a machine without them rather than letting npm
# silently fall over later.
if ! command -v node >/dev/null 2>&1; then
    echo "🚫 node is required to build the React frontend. Install Node.js 18+ before running deploy.sh."
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "🚫 npm is required to build the React frontend. Install Node.js (which ships npm) 18+ before running deploy.sh."
    exit 1
fi

echo ""
echo "▶️ Creating a secret scope"
echo ""

echo "Scope name: $secret_scope_name"

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

echo ""
echo "▶️ Creating schema if not exists"
echo ""

set +e
databricks schemas get $core_catalog_name.$core_schema_name
if [ "$?" -eq "0" ]
then
  echo "Schema $core_catalog_name.$core_schema_name already exists"
else
  echo "Schema $core_catalog_name.$core_schema_name does not exist. Creating.."
  databricks schemas create $core_schema_name $core_catalog_name
fi
set -e

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

databricks bundle deploy --target $TARGET --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Copying libraries to UC Volume"
echo ""

# IMPORTANT: this copy must run right after `bundle deploy` (which creates the
# `libraries` Volume) and BEFORE any job runs. Serverless jobs such as
# grant_app_permissions_job %pip-install the genesis_workbench wheel from this
# Volume at runtime — on a fresh deploy the Volume is empty until this step, so
# copying it later fails with "ModuleNotFoundError: No module named
# 'genesis_workbench'". Each module's notebooks also %pip install from here.
for file in library/genesis_workbench/dist/*.whl; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Copying $filename to dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename"
    databricks fs cp library/genesis_workbench/dist/$filename dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename --overwrite
  fi
done

# Glow library (JAR + wheel) is consumed by the GWAS submodule's job-cluster
# init scripts. Lives outside the python wheel so it can be downloaded by
# Spark drivers via spark.jars.packages without a venv install round-trip.
for file in library/glow/*; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Copying $filename to dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename"
    databricks fs cp "$file" dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename --overwrite
  fi
done

#Run init job only if not deployed before
if [[ ! -e ".deployed" ]]; then

  echo ""
  echo "▶️ Running initialization job"
  echo ""

  databricks bundle run --target $TARGET initialize_core_job --var="$EXTRA_PARAMS"
fi

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
# Idempotent — iterates DATABRICKS_APP_NAMES so the genesis-workbench SP (and
# any sibling app if app_names is set to a colon-separated list) ends up with
# CAN_QUERY / CAN_MANAGE_RUN / READ+WRITE VOLUME / EXECUTE on every existing
# resource. Safe to re-run on every deploy; new resources get reconciled then.
# NOTE: the genesis_workbench wheel is copied to the UC Volume earlier (right
# after `bundle deploy`) because this serverless job %pip-installs it from there.
databricks bundle run --target $TARGET grant_app_permissions_job --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Cleaning up local build artifacts"
echo ""
# poetry's dist/ regenerates on every deploy; the staged wheel under
# app/backend/lib/ stays because it's referenced by requirements.txt and
# served at runtime.
rm -rf library/genesis_workbench/dist

date +"%Y-%m-%d %H:%M:%S" > .deployed
