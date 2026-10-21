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

# The orchestrator loads RFD4-Proteina in-process on an H100, pip-installing the
# model code from the PUBLIC GitHub repo (same as the rfd4_proteina submodule — nothing
# vendored), and downloads released weights from Hugging Face. The HF weights repo may
# be gated, so it reads an HF token from a Databricks secret. Seed it here from
# $HF_TOKEN if provided; otherwise warn with the exact command (a run will fail without
# it if the HF repo is gated).
RFD4_HF_TOKEN_SCOPE="${RFD4_HF_TOKEN_SCOPE:-dbx_genesis_workbench}"
RFD4_HF_TOKEN_KEY="${RFD4_HF_TOKEN_KEY:-hf_token}"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "▶️ [Antibody Design] Storing Hugging Face token in secret ${RFD4_HF_TOKEN_SCOPE}/${RFD4_HF_TOKEN_KEY}"
  databricks secrets put-secret "$RFD4_HF_TOKEN_SCOPE" "$RFD4_HF_TOKEN_KEY" --string-value "$HF_TOKEN"
elif ! databricks secrets list-secrets "$RFD4_HF_TOKEN_SCOPE" 2>/dev/null | grep -qw "$RFD4_HF_TOKEN_KEY"; then
  echo "⚠️  [Antibody Design] Secret ${RFD4_HF_TOKEN_SCOPE}/${RFD4_HF_TOKEN_KEY} is not set."
  echo "    The orchestrator downloads RFD4-Proteina weights from Hugging Face (may be gated). If the HF repo is gated, set:"
  echo "      databricks secrets put-secret ${RFD4_HF_TOKEN_SCOPE} ${RFD4_HF_TOKEN_KEY} --string-value <HF_TOKEN>"
  echo "    or re-run: HF_TOKEN=<HF_TOKEN> $0 $CLOUD  (continuing; a run will fail if the repo is gated and token is missing)"
fi

echo ""
echo "▶️ [Antibody Design] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [Antibody Design] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [Antibody Design] Running the registration job (persist the orchestrator job id +"
echo "    grant the app SP CAN_MANAGE_RUN + WRITE on the antigen-upload volume)."
echo "    Foreground (NOT --no-wait) — registration must finish before the app can launch runs."
echo ""

databricks bundle run --target $TARGET register_antibody_design_job $EXTRA_PARAMS
