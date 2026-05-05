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

echo "⚙️ Starting destroy of module Disease Biology"

ALL_SUBMODULES=(gwas/gwas_v1 vcf_ingestion/vcf_ingestion_v1 variant_annotation/variant_annotation_v1)

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
