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

# RFD4-Proteina's model code is now a PUBLIC GitHub repo (RosettaCommons, BSD-3) — pip-installed by the
# register/finetune/deploy jobs at runtime, nothing vendored into GWB, and NO GitHub PAT needed. The
# released WEIGHTS come from a Hugging Face repo that may be gated, so the jobs authenticate with a Hugging
# Face token read from a Databricks secret. Seed it here from $HF_TOKEN if provided; otherwise warn with
# the exact command (the register job fails fast with the same instruction if the secret is missing).
RFD4_HF_TOKEN_SCOPE="${RFD4_HF_TOKEN_SCOPE:-dbx_genesis_workbench}"
RFD4_HF_TOKEN_KEY="${RFD4_HF_TOKEN_KEY:-hf_token}"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "▶️ [RFD4-Proteina] Storing Hugging Face token in secret ${RFD4_HF_TOKEN_SCOPE}/${RFD4_HF_TOKEN_KEY}"
  databricks secrets put-secret "$RFD4_HF_TOKEN_SCOPE" "$RFD4_HF_TOKEN_KEY" --string-value "$HF_TOKEN"
elif ! databricks secrets list-secrets "$RFD4_HF_TOKEN_SCOPE" 2>/dev/null | grep -qw "$RFD4_HF_TOKEN_KEY"; then
  echo "⚠️  [RFD4-Proteina] Secret ${RFD4_HF_TOKEN_SCOPE}/${RFD4_HF_TOKEN_KEY} is not set."
  echo "    The jobs need a Hugging Face token with access to the gated RFD4-Proteina weights. Set it with:"
  echo "      databricks secrets put-secret ${RFD4_HF_TOKEN_SCOPE} ${RFD4_HF_TOKEN_KEY} --string-value <HF_TOKEN>"
  echo "    or re-run: HF_TOKEN=<token> $0 $CLOUD  (continuing; the register job will fail without it)"
fi

echo ""
echo "▶️ [RFD4-Proteina] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [RFD4-Proteina] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [RFD4-Proteina] Running model registration job (pull H100/CUDA-13 ckpts, register PyFunc,"
echo "    deploy the GPU_XLARGE/H100 serving endpoint, and register the finetune/deploy jobs)."
echo "🚨 This builds a CUDA-13 stack on an H100 and deploys an H100 (GPU_XLARGE) endpoint — it can"
echo "    take a long time. See the Jobs & Pipelines tab for status."
echo ""

databricks bundle run --target $TARGET register_rfd4_proteina $EXTRA_PARAMS --no-wait
