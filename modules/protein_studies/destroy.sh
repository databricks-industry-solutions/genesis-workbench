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

echo "⚙️ Starting destroy of module Protein Studies"

for module in alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
    do
        cd $module
        echo "Running command destroy.sh $ENV --var=\"$EXTRA_PARAMS\" " 
        chmod +x destroy.sh
        
        ./destroy.sh $ENV --var="$EXTRA_PARAMS" 
        cd ../..
    done


rm .deployed

