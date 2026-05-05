#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: destroy <cloud>"
    echo 'Example: destroy aws'
    exit 1
fi

CLOUD=$1

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp>"; exit 1 ;;
esac

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
fi
  
echo "Extra Params: $EXTRA_PARAMS"

echo "=========================================================="
echo "⚙️ Preparing to destroy module core "
echo "=========================================================="

databricks bundle destroy --target $TARGET --var="$EXTRA_PARAMS" --auto-approve

rm -f .deployed

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi

