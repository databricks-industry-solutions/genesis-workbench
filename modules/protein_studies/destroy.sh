#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>  <additional build variables>"
    echo 'Example: destroy core dev --var="dev_user_prefix=scn"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo "⚙️ Starting destroy of module Protein Studies"


for module in alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
    do
        cd $module
        ./destroy.sh $ENV $EXTRA_PARAMS
        cd ../..
    done




