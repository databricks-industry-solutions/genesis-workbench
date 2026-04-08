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
fi

echo "Extra Params: $EXTRA_PARAMS"

echo "##############################################"
echo "⏩️ Starting deploy of Disease Biology module  #"

#for module in gwas/gwas_v1 vcf_ingestion/vcf_ingestion_v1 variant_annotation/variant_annotation_v1
for module in vcf_ingestion/vcf_ingestion_v1 variant_annotation/variant_annotation_v1

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
