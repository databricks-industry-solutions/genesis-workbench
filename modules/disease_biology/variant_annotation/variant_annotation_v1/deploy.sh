#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [Variant Annotation] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [Variant Annotation] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

if [[ ! -e ".deployed" ]]; then
    echo ""
    echo "▶️ [Variant Annotation] Running initial setup job"
    echo "🚨 This job will download ClinVar and set up reference data. See Jobs & Pipeline tab for status"
    echo ""

    user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
    databricks bundle run --params "user_email=$user_email" variant_annotation_initial_setup_job $EXTRA_PARAMS --no-wait
fi

date +"%Y-%m-%d %H:%M:%S" > .deployed
