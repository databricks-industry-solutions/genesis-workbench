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
echo ">>> [ESM2 Embeddings] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo ">>> [ESM2 Embeddings] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo ">>> [ESM2 Embeddings] Running model registration job"
echo "This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET register_esm2_embeddings $EXTRA_PARAMS --no-wait
