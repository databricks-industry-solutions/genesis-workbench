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

echo ""
echo "▶️ [KERMT] Fine-tuning the default model (TDC ClinTox sample) on a GPU cluster."
echo "🚨 This runs a full fine-tune and can take a while (GPU cluster-create + training)."
echo "           It writes a kermt_weights row (ft_id) that the deploy step below picks up."
echo ""
databricks bundle run --target $TARGET kermt_finetune $EXTRA_PARAMS

echo ""
echo "▶️ [KERMT] Deploying the serving endpoint from the fine-tuned checkpoint."
echo "           No ft_id is passed, so the deploy notebook uses the latest active ft_id"
echo "           (the one just produced by the finetune step above)."
echo ""
databricks bundle run --target $TARGET kermt_deploy $EXTRA_PARAMS
