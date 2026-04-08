#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [VCF Ingestion] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [VCF Ingestion] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

if [[ ! -e ".deployed" ]]; then
    echo ""
    echo "▶️ [VCF Ingestion] Running initial setup job"
    echo ""

    user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
    databricks bundle run --params "user_email=$user_email" vcf_ingestion_initial_setup_job $EXTRA_PARAMS --no-wait
fi

date +"%Y-%m-%d %H:%M:%S" > .deployed
