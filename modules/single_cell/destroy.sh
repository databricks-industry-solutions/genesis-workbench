#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> "
    echo 'Example: destroy dev'
    exit 1
fi

ENV=$1

source env.env

EXTRA_PARAMS=$(paste -sd, "env.env")

echo "Extra Params: $EXTRA_PARAMS"

echo "⚙️ Starting destroy of module Single Cell"

for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
    do
        cd $module
        ./destroy.sh $ENV --var="$EXTRA_PARAMS"
        cd ../..
    done

