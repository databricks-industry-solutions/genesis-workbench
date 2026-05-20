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
echo "▶️ [TEDDY] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [TEDDY] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [TEDDY] Running model registration job as a backend task"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET register_teddy $EXTRA_PARAMS --no-wait
