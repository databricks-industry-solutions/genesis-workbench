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

# RFD4-Proteina's model code lives in a PRIVATE GitHub repo (no public HuggingFace mirror) and is
# pip-installed by the register/finetune/deploy jobs at runtime — nothing is vendored into GWB, so a
# customer needs nothing cloned locally. The jobs authenticate with a GitHub PAT (read access to the
# repo) read from a Databricks secret. Seed it here from $RFD4_GITHUB_TOKEN if provided; otherwise
# warn with the exact command (the job fails fast with the same instruction if the secret is missing).
RFD4_GIT_TOKEN_SCOPE="${RFD4_GIT_TOKEN_SCOPE:-dbx_genesis_workbench}"
RFD4_GIT_TOKEN_KEY="${RFD4_GIT_TOKEN_KEY:-rfd4_github_token}"
if [ -n "${RFD4_GITHUB_TOKEN:-}" ]; then
  echo "▶️ [RFD4-Proteina] Storing GitHub PAT in secret ${RFD4_GIT_TOKEN_SCOPE}/${RFD4_GIT_TOKEN_KEY}"
  databricks secrets put-secret "$RFD4_GIT_TOKEN_SCOPE" "$RFD4_GIT_TOKEN_KEY" --string-value "$RFD4_GITHUB_TOKEN"
elif ! databricks secrets list-secrets "$RFD4_GIT_TOKEN_SCOPE" 2>/dev/null | grep -qw "$RFD4_GIT_TOKEN_KEY"; then
  echo "⚠️  [RFD4-Proteina] Secret ${RFD4_GIT_TOKEN_SCOPE}/${RFD4_GIT_TOKEN_KEY} is not set."
  echo "    The jobs need a GitHub PAT with read access to the private RFD4-Proteina repo. Set it with:"
  echo "      databricks secrets put-secret ${RFD4_GIT_TOKEN_SCOPE} ${RFD4_GIT_TOKEN_KEY} --string-value <PAT>"
  echo "    or re-run: RFD4_GITHUB_TOKEN=<PAT> $0 $CLOUD  (continuing; the register job will fail without it)"
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
