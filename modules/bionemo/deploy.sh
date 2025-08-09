#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <env> <cloud>"
    echo "Example: deploy dev aws"
    exit 1
fi

ENV=$1
CLOUD=$2

source env.env

EXTRA_PARAMS_CLOUD=$(paste -sd, "$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "env.env")
EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

echo "Extra Params: $EXTRA_PARAMS"

echo ""
echo "▶️ [BioNeMo] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [BioNeMo] Deploying bundle"
echo ""

databricks bundle deploy -t $ENV $EXTRA_PARAMS

echo ""
echo "▶️ [BioNeMo] Running model registration job as a backend task"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run -t $ENV initial_setup_job $EXTRA_PARAMS --no-wait