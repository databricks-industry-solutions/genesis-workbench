#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <env> <cloud>"
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

echo "⚙️ Starting destroy of module Single Cell"

for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
    do
        cd $module
        ./destroy.sh $ENV --var="$EXTRA_PARAMS"
        cd ../..
    done

rm .deployed