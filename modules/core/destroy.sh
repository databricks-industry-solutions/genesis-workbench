#!/bin/bash

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> "
    echo 'Example: destroy dev'
    exit 1
fi

ENV=$1

source env.env

EXTRA_PARAMS=$(paste -sd, "env.env")

echo "Extra Params: $EXTRA_PARAMS"

echo "=========================================================="
echo "⚙️ Preparing to destroy module core from $ENV"
echo "=========================================================="

databricks bundle destroy -t $ENV --var="$EXTRA_PARAMS" --auto-approve

rm .deployed

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi

