#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_working_directory> <env> "
    echo 'Example: destroy dev '
    exit 1
fi

ENV=$1

source env.env

EXTRA_PARAMS=$(paste -sd, "env.env")

echo "Extra Params: $EXTRA_PARAMS"

echo "⚙️ Starting destroy of module Protein Studies"

#for module in alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
for module in boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
    do
        cd $module
        ./destroy.sh $ENV --var="$EXTRA_PARAMS" 
        cd ../..
    done




