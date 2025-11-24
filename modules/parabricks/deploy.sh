#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cloud>"
    echo "Example: deploy aws"
    exit 1
fi

CLOUD=$1

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
else
    EXTRA_PARAMS_MODULE=''
fi

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD,$EXTRA_PARAMS_MODULE"

echo "Extra Params: $EXTRA_PARAMS"

echo ""
echo "▶️ [Parabricks] Validating bundle"
echo ""

databricks bundle validate --var="$EXTRA_PARAMS" 

echo ""
echo "▶️ [Parabricks] Deploying bundle"
echo ""

databricks bundle deploy --var="$EXTRA_PARAMS" 

if [[ ! -e ".deployed" ]]; then
    echo ""
    echo "▶️ [Parabricks] Running module initialization job as a backend task"
    echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
    echo ""

    user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
    databricks bundle run --params "user_email=$user_email" initial_setup_job --var="$EXTRA_PARAMS"  --no-wait
fi

date +"%Y-%m-%d %H:%M:%S" > .deployed