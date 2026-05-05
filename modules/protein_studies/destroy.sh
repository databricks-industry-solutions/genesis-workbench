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

echo "Extra Params: $EXTRA_PARAMS"

echo "⚙️ Starting destroy of module Protein Studies"

ALL_SUBMODULES=(alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0 esm2_embeddings/esm2_embeddings_v1 sequence_search/sequence_search_v1)

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
        cd $module
        echo "Running command destroy.sh $CLOUD --var=\"$EXTRA_PARAMS\" "
        chmod +x destroy.sh

        ./destroy.sh $CLOUD --var="$EXTRA_PARAMS"
        cd ../..
    done


if [[ -z "$ONLY_SUBMODULE" ]]; then
    rm -f .deployed
fi

