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

echo "###########################################"
echo "⏩️ Starting deploy of Single Cell module  #"

# for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
for module in scanpy/scanpy_v0.0.1
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