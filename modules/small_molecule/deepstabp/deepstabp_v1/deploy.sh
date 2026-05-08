
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
echo "▶️ [DeepSTABp] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [DeepSTABp] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [DeepSTABp] Running model registration job"
echo "🚨 First run downloads ProtT5-XL backbone (~3 GB) from HuggingFace + the upstream MLP head (~80 MB). Allow several minutes."
echo ""

databricks bundle run --target $TARGET register_deepstabp $EXTRA_PARAMS --no-wait
