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
echo "▶️ [HLAIIPred] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [HLAIIPred] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [HLAIIPred] Running model registration job"
echo "🚨 Clones the public pfizer-opensource/HLAIIPred repo (Apache-2.0, ~9 MB weights), registers the"
echo "    PyFunc, and deploys a CPU serving endpoint. See the Jobs tab for status."
echo ""

databricks bundle run --target $TARGET register_hlaiipred $EXTRA_PARAMS --no-wait
