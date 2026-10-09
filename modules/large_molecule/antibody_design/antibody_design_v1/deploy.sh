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
# model code from the PRIVATE GitHub repo at runtime (same as the rfd4_proteina
# submodule — nothing vendored). It authenticates with a GitHub PAT read from a
# Databricks secret. Seed it here from $RFD4_GITHUB_TOKEN if provided; otherwise
# warn with the exact command (the job fails fast with the same instruction).
RFD4_GIT_TOKEN_SCOPE="${RFD4_GIT_TOKEN_SCOPE:-dbx_genesis_workbench}"
RFD4_GIT_TOKEN_KEY="${RFD4_GIT_TOKEN_KEY:-rfd4_github_token}"
if [ -n "${RFD4_GITHUB_TOKEN:-}" ]; then
  echo "▶️ [Antibody Design] Storing GitHub PAT in secret ${RFD4_GIT_TOKEN_SCOPE}/${RFD4_GIT_TOKEN_KEY}"
  databricks secrets put-secret "$RFD4_GIT_TOKEN_SCOPE" "$RFD4_GIT_TOKEN_KEY" --string-value "$RFD4_GITHUB_TOKEN"
elif ! databricks secrets list-secrets "$RFD4_GIT_TOKEN_SCOPE" 2>/dev/null | grep -qw "$RFD4_GIT_TOKEN_KEY"; then
  echo "⚠️  [Antibody Design] Secret ${RFD4_GIT_TOKEN_SCOPE}/${RFD4_GIT_TOKEN_KEY} is not set."
  echo "    The orchestrator needs a GitHub PAT with read access to the private RFD4-Proteina repo. Set it with:"
  echo "      databricks secrets put-secret ${RFD4_GIT_TOKEN_SCOPE} ${RFD4_GIT_TOKEN_KEY} --string-value <PAT>"
  echo "    or re-run: RFD4_GITHUB_TOKEN=<PAT> $0 $CLOUD  (continuing; a run will fail without it)"
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
