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
echo "▶️ [KERMT] Validating bundle (target=$TARGET)"
echo ""
databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [KERMT] Deploying bundle (target=$TARGET)"
echo ""
databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [KERMT] Running stage/register job (downloads GROVERbase, creates kermt_weights,"
echo "           bundles the TDC sample, persists job ids + app permissions)"
echo "🚨 First run downloads the ~185MB GROVERbase checkpoint. See Jobs tab for status."
echo ""
databricks bundle run --target $TARGET register_kermt $EXTRA_PARAMS
