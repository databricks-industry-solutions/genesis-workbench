#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> "
    echo 'Example: deploy dev '
    exit 1
fi

ENV=$1

source env.env

EXTRA_PARAMS=$(paste -sd, "env.env")

echo "Extra Params: $EXTRA_PARAMS"

echo "##############################################"
echo "⏩️ Starting deploy of Protein Studies module #"

#for module in alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
# TODO
for module in alphafold/alphafold_v2.3.2
    do
        echo "###########################################"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh
        echo "Running command deploy.sh $ENV --var=\"$EXTRA_PARAMS\" " 
        ./deploy.sh $ENV --var="$EXTRA_PARAMS" 
        cd ../..        
    done

echo "##############################################"


