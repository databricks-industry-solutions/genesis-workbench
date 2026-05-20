#!/bin/bash

CLOUD=$1
EXTRA_PARAMS=${@:2}

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp> --var=..."; exit 1 ;;
esac

echo "=========================================================="
echo "⚙️ Preparing to destroy module teddy_g_v1 (target=$TARGET)"
echo "=========================================================="
echo ""
echo "ℹ️  PRESERVED across destroy (NOT bundle resources):"
echo "     • Delta table: {catalog}.{schema}.teddy_cells   ← rebuild takes hours"
echo "     • VS endpoint: gwb_teddy_vs_endpoint"
echo "     • VS index:    {catalog}.{schema}.teddy_cell_index"
echo "   These are created procedurally by notebooks 03/04 and live outside the"
echo "   bundle, so 'databricks bundle destroy' leaves them intact. Re-deploying"
echo "   later is a no-op for them (notebooks 03/04 idempotency-check first)."
echo ""
echo "🗑️  Removed by destroy (bundle resources):"
echo "     • Job:     register_teddy"
echo "     • Volume:  teddy (HF model snapshot — re-downloaded on next deploy)"
echo ""

databricks bundle destroy --target $TARGET $EXTRA_PARAMS --auto-approve

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi
