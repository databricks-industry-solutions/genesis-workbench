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

# The orchestrator loads RFD4-Proteina (BSD-3-Clause public repo) in-process on an H100,
# pip-installing the model code from github.com/RosettaCommons/RFD4-Proteina.git at runtime
# (nothing vendored). The HF weights repo may be gated; seed it here from $HF_TOKEN if provided;
# otherwise warn with the exact command (the job fails fast with the same instruction).
HF_TOKEN_SCOPE="${HF_TOKEN_SCOPE:-dbx_genesis_workbench}"
HF_TOKEN_KEY="${HF_TOKEN_KEY:-hf_token}"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "▶️ [Vaccine Immunogen] Storing Hugging Face token in secret ${HF_TOKEN_SCOPE}/${HF_TOKEN_KEY}"
  databricks secrets put-secret "$HF_TOKEN_SCOPE" "$HF_TOKEN_KEY" --string-value "$HF_TOKEN"
elif ! databricks secrets list-secrets "$HF_TOKEN_SCOPE" 2>/dev/null | grep -qw "$HF_TOKEN_KEY"; then
  echo "⚠️  [Vaccine Immunogen] Secret ${HF_TOKEN_SCOPE}/${HF_TOKEN_KEY} is not set."
  echo "    If the nvidia/RFD4-Proteina HF repo is gated, set a token with:"
  echo "      databricks secrets put-secret ${HF_TOKEN_SCOPE} ${HF_TOKEN_KEY} --string-value <HF_TOKEN>"
  echo "    or re-run: HF_TOKEN=<HF_TOKEN> $0 $CLOUD  (continuing; a run will fail if the repo is gated)"
fi

echo ""
echo "▶️ [Vaccine Immunogen] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [Vaccine Immunogen] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [Vaccine Immunogen] Running the registration job (persist the orchestrator job id +"
echo "    grant the app SP CAN_MANAGE_RUN + WRITE on the motif-upload volume)."
echo "    Foreground (NOT --no-wait) — registration must finish before the app can launch runs."
echo ""

databricks bundle run --target $TARGET register_vaccine_immunogen_job $EXTRA_PARAMS
