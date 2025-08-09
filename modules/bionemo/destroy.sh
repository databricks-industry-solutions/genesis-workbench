#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: destroy <env> <cloud>"
    echo "Example: destroy dev aws"
    exit 1
fi

ENV=$1
CLOUD=$2

source env.env

EXTRA_PARAMS_CLOUD=$(paste -sd, "$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "env.env")
EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

echo "Extra Params: $EXTRA_PARAMS"

echo "=========================================================="
echo "⚙️ Preparing to destroy module scgpt_v0.2.4 from $ENV"
echo "=========================================================="

databricks bundle destroy -t $ENV $EXTRA_PARAMS --auto-approve

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi

