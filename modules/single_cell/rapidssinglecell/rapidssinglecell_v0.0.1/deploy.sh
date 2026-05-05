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
echo "▶️ [rapidssinglecell] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [rapidssinglecell] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [rapidssinglecell] Running registration job to grant app permissions"
echo ""

databricks bundle run --target $TARGET register_rapidssinglecell_job $EXTRA_PARAMS

echo ""
echo "▶️ [rapidssinglecell] Downloading gene reference tables"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET download_gene_references_gwb $EXTRA_PARAMS --no-wait

echo ""
echo "✅ [rapidssinglecell] Deployment complete"
echo ""
