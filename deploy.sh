#!/bin/bash

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: deploy <module> <cloud> "
    echo 'Example: deploy core aws'
    exit 1
fi

CWD=$1
CLOUD=$2

if [[ "$CWD" != "core" && ! -f "modules/core/.deployed" ]]; then
    echo "🚫 Deploy core module first before installing sub-modules"
    exit 1
fi

echo "Installing Poetry"

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

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
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
else
    EXTRA_PARAMS_MODULE=''
fi

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD,$EXTRA_PARAMS_MODULE"
user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
        
databricks bundle run --params "module=$CWD" initialize_module_job --var="$EXTRA_PARAMS"

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




