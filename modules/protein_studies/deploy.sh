#!/bin/bash
set -e

CLOUD=$1

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
fi

echo "Extra Params: $EXTRA_PARAMS"

echo "##############################################"
echo "⏩️ Starting deploy of Protein Studies module #"

#for module in alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
#for module in boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
#for module in esm2_embeddings/esm2_embeddings_v1 sequence_search/sequence_search_v1
for module in sequence_search/sequence_search_v1 
    do
        echo "###########################################"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh
        
        echo "Running command deploy.sh --var=\"$EXTRA_PARAMS\" " 
        ./deploy.sh --var="$EXTRA_PARAMS" 
        cd ../..        
    done

echo "##############################################"

date +"%Y-%m-%d %H:%M:%S" > .deployed
