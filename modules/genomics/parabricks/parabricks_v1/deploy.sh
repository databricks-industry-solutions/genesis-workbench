#!/bin/bash
set -e

CLOUD=$1
EXTRA_PARAMS=${@:2}

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp> --var=..."; exit 1 ;;
esac

echo ""
echo "▶️ [Parabricks] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [Parabricks] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [Parabricks] Running initial setup job"
echo "🚨 Creates Parabricks GPU cluster + registers batch model. See Jobs & Pipeline tab for status"
echo ""

user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
databricks bundle run --target $TARGET --params "user_email=$user_email" initial_setup_job $EXTRA_PARAMS --no-wait
