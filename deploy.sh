#!/bin/bash

set -e

# Use locally-installed Terraform to avoid the expired HashiCorp PGP key
# that breaks the Databricks CLI's embedded Terraform download.
export DATABRICKS_TF_EXEC_PATH=/opt/homebrew/bin/terraform
export DATABRICKS_TF_VERSION=1.3.9

if [ "$#" -lt 2 ]; then
    echo "Usage: deploy <module> <cloud> [--only-submodule <path>]"
    echo 'Example: deploy core aws'
    echo 'Example: deploy single_cell aws --only-submodule scimilarity/scimilarity_v0.4.0_weights_v1.1'
    exit 1
fi

CWD=$1
CLOUD=$2
shift 2

ONLY_SUBMODULE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-submodule)
            ONLY_SUBMODULE="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1"; exit 1
            ;;
    esac
done

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: deploy <module> <aws|azure|gcp> [--only-submodule <path>]"; exit 1 ;;
esac

if [[ -n "$ONLY_SUBMODULE" && ! -d "modules/$CWD/$ONLY_SUBMODULE" ]]; then
    echo "🚫 Submodule '$ONLY_SUBMODULE' not found in modules/$CWD/"
    echo "    (atomic modules like core, bionemo, parabricks have no submodules)"
    exit 1
fi

if [[ "$CWD" != "core" && ! -f "modules/core/.deployed" ]]; then
    echo "🚫 Deploy core module first before installing sub-modules"
    exit 1
fi

echo "Installing Poetry"

#curl -sSL https://install.python-poetry.org | python3 -
#export PATH="/root/.local/bin:$PATH"
pip install poetry

echo "================================"
echo "⚙️ Preparing to deploy module $CWD"
echo "================================"

source application.env

#export BUNDLE_VAR_databricks_host=$databricks_host

cd modules/$CWD
chmod +x deploy.sh
if [[ -n "$ONLY_SUBMODULE" ]]; then
    ./deploy.sh $CLOUD --only-submodule "$ONLY_SUBMODULE"
else
    ./deploy.sh $CLOUD
fi

cd ../..
echo "======================================="
echo "⚙️ Running initialization job for $CWD"
echo "======================================="

cd modules/core

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

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




