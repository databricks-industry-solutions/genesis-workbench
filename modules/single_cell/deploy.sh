#!/bin/bash
set -e

CLOUD=$1
shift

ONLY_SUBMODULE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-submodule) ONLY_SUBMODULE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

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
    EXTRA_PARAMS_MODULE_CLOUD=''
fi
EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE_CLOUD"

echo "Extra Params: $EXTRA_PARAMS"

echo "###########################################"
echo "⏩️ Starting deploy of Single Cell module  #"

ALL_SUBMODULES=(scanpy/scanpy_v0.0.1 rapidssinglecell/rapidssinglecell_v0.0.1 scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1)

if [[ -n "$ONLY_SUBMODULE" ]]; then
    found=false
    for s in "${ALL_SUBMODULES[@]}"; do
        if [[ "$s" == "$ONLY_SUBMODULE" ]]; then found=true; break; fi
    done
    if [[ "$found" != "true" ]]; then
        echo "Error: --only-submodule must be one of: ${ALL_SUBMODULES[*]}"
        exit 1
    fi
    SUBMODULES=("$ONLY_SUBMODULE")
else
    SUBMODULES=("${ALL_SUBMODULES[@]}")
fi

for module in "${SUBMODULES[@]}"
    do
        echo "###########################################"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh

        echo "Running command deploy.sh $CLOUD --var=\"$EXTRA_PARAMS\" "
        ./deploy.sh $CLOUD --var="$EXTRA_PARAMS"
        cd ../..
    done
echo "##############################################"

# Only mark parent as fully deployed for full deploys
if [[ -z "$ONLY_SUBMODULE" ]]; then
    date +"%Y-%m-%d %H:%M:%S" > .deployed
fi