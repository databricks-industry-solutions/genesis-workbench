#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: destroy <cloud>"
    echo "Example: destroy aws"
    exit 1
fi

CLOUD=$1

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
fi

echo "Extra Params: $EXTRA_PARAMS"

echo "=========================================================="
echo "⚙️ Preparing to destroy module bionemo "
echo "=========================================================="

databricks bundle destroy --var="$EXTRA_PARAMS" --auto-approve

rm .deployed

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi


rm .deployed