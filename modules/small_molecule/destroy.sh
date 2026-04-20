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
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE_CLOUD"
else
    EXTRA_PARAMS_MODULE_CLOUD=''
fi


echo "Extra Params: $EXTRA_PARAMS"

echo "⚙️ Starting destroy of module Small Molecule"

for module in chemprop/chemprop_v2 diffdock/diffdock_v1 proteina_complexa/proteina_complexa_v1
    do
        cd $module
        echo "Running command destroy.sh $CLOUD --var=\"$EXTRA_PARAMS\" "
        chmod +x destroy.sh

        ./destroy.sh $CLOUD --var="$EXTRA_PARAMS"
        cd ../..
    done

rm -f .deployed
