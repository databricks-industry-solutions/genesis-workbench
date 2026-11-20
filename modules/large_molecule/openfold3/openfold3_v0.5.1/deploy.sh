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

# OpenFold3 (openfold3==0.5.1, Apache-2.0) + the default OpenBind-0 checkpoint (Apache-2.0) are PUBLIC:
# the fold job pip-installs the package into an isolated venv and downloads the ~2 GB checkpoint to the
# model Volume at runtime. Nothing is vendored and no token/secret is needed (unlike AlphaFold2's
# ~2 TB database download). MSA comes from the remote ColabFold server at fold time (use_msa toggle).

echo ""
echo "▶️ [OpenFold3] Validating bundle (target=$TARGET)"
echo ""
databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [OpenFold3] Deploying bundle (target=$TARGET) — creates the model Volume + the run/register jobs"
echo ""
databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [OpenFold3] Running registration job (persist the fold job id, grant app permissions, register"
echo "    the 'openfold3' batch model). The ~2 GB checkpoint downloads lazily on the FIRST fold run."
echo ""
databricks bundle run --target $TARGET openfold3_register $EXTRA_PARAMS --no-wait
