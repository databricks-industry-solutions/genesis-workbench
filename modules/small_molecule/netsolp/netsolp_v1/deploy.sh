
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
echo "▶️ [NetSolP] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [NetSolP] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [NetSolP] Running model registration job"
echo "🚨 NetSolP weights must be uploaded to the UC volume before this job will succeed."
echo "   See modules/small_molecule/netsolp/netsolp_v1/notebooks/01_register_netsolp.py header for instructions."
echo ""

databricks bundle run --target $TARGET register_netsolp $EXTRA_PARAMS --no-wait
