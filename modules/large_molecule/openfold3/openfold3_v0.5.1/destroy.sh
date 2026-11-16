#!/bin/bash
set -e

CLOUD=$1
EXTRA_PARAMS=${@:2}

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp>"; exit 1 ;;
esac

# NOTE: destroying the bundle removes the MANAGED model Volume and with it the cached checkpoint and
# all prediction results.
databricks bundle destroy --target $TARGET $EXTRA_PARAMS --auto-approve
