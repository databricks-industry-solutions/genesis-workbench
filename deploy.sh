#!/bin/bash

set -e

# Use locally-installed Terraform to avoid the expired HashiCorp PGP key
# that breaks the Databricks CLI's embedded Terraform download.
# Dynamic path + version detection (was hardcoded to /opt/homebrew/bin/terraform + 1.3.9 —
# see docs/deployments/fevm-mmt-aws-usw2/UX-GAPS.md entry #1).
export DATABRICKS_TF_EXEC_PATH="$(command -v terraform)"
export DATABRICKS_TF_VERSION="$(terraform version -json | jq -r .terraform_version)"

# Source application.env early so databricks_profile is available for CLI calls.
# Falls back to DEFAULT if not set — preserves existing behavior.
if [ -f "application.env" ]; then
    set -a
    source application.env
    set +a
fi
export DATABRICKS_CONFIG_PROFILE="${databricks_profile:-DEFAULT}"
echo "Using databricks profile: $DATABRICKS_CONFIG_PROFILE"

if [ "$#" -lt 2 ]; then
    echo "Usage: deploy <module> <cloud> "
    echo 'Example: deploy core aws'
    exit 1
fi

CWD=$1
CLOUD=$2

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: deploy <module> <aws|azure|gcp>"; exit 1 ;;
esac

if [[ "$CWD" != "core" && ! -f "modules/core/.deployed" ]]; then
    echo "🚫 Deploy core module first before installing sub-modules"
    exit 1
fi

echo "Installing Poetry"

# Prefer the official curl-based installer — bypasses PEP 668 (Homebrew's
# externally-managed-environment) and works without --user or venv gymnastics.
# Falls back gracefully if poetry is already on PATH.
if ! command -v poetry >/dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org | python3 -
fi
export PATH="$HOME/.local/bin:$PATH"
command -v poetry || { echo "❌ poetry install failed"; exit 1; }

echo "================================"
echo "⚙️ Preparing to deploy module $CWD"
echo "================================"

source application.env

#export BUNDLE_VAR_databricks_host=$databricks_host

cd modules/$CWD
chmod +x deploy.sh
./deploy.sh $CLOUD

cd ../..
echo "======================================="
echo "⚙️ Running initialization job for $CWD"
echo "======================================="

cd modules/core

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(grep -v '^databricks_profile=' ../../application.env | tr '\n' ',' | sed 's/,$//')

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
else
    EXTRA_PARAMS_MODULE=''
fi

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD,$EXTRA_PARAMS_MODULE"
user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
        
databricks bundle run --target $TARGET --params "module=$CWD" initialize_module_job --var="$EXTRA_PARAMS"

cd ../..

if [ $? -eq 0 ]; then
    echo "================================"
    echo "✅ SUCCESS! Deployment complete."
    echo "================================"
else
    echo "================================"
    echo "❗️ ERROR! Deployment failed."
    echo "================================"
fi




