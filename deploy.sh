#!/bin/bash

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: deploy <module> <env> <cloud> "
    echo 'Example: deploy core dev aws'
    exit 1
fi


CWD=$1
ENV=$2
CLOUD=$3


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

cd modules/$CWD
chmod +x deploy.sh
./deploy.sh $ENV $CLOUD

cd ../..
echo "======================================="
echo "⚙️ Running initialization job for $CWD"
echo "======================================="

cd modules/core

source env.env

EXTRA_PARAMS_CLOUD=$(paste -sd, "$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "env.env")
EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"
user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
        
databricks bundle run -t $ENV --params "module=$CWD" initialize_module_job --var="$EXTRA_PARAMS"

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




