#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [GWAS] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [GWAS] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

echo ""
echo "▶️ [GWAS] Running initial setup job"
echo "🚨 This job will install Glow and download reference genomes. See Jobs & Pipeline tab for status"
echo ""

user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
databricks bundle run --params "user_email=$user_email" gwas_initial_setup_job $EXTRA_PARAMS --no-wait

