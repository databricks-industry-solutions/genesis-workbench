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
if [[ -f "module_${CLOUD}.env" ]]; then
    EXTRA_PARAMS_MODULE_CLOUD=$(paste -sd, "module_${CLOUD}.env")
else
    EXTRA_PARAMS_MODULE_CLOUD=""
fi
EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE_CLOUD"

echo "Extra Params: $EXTRA_PARAMS"

echo "⚙️ Starting destroy of module Single Cell"

# for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
# for module in scanpy/scanpy_v0.0.1 rapidssinglecell/rapidssinglecell_v0.0.1
for module in scanpy/scanpy_v0.0.1 rapidssinglecell/rapidssinglecell_v0.0.1 scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
    do
        cd $module
        echo "Running command destroy.sh --var=\"$EXTRA_PARAMS\" " 
        chmod +x destroy.sh

        ./destroy.sh --var="$EXTRA_PARAMS"
        cd ../..
    done

rm .deployed