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

echo "⚙️ Starting destroy of module Disease Biology"

for module in gwas/gwas_v1
    do
        cd $module
        echo "Running command destroy.sh --var=\"$EXTRA_PARAMS\" "
        chmod +x destroy.sh

        ./destroy.sh --var="$EXTRA_PARAMS"
        cd ../..
    done

rm .deployed
