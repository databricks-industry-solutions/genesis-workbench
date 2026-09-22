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

# The parabricks_alignment job's run_germline task runs on serverless GPU via an
# ai_runtime_task (stock NGC Parabricks image) — it's launched ON DEMAND from the UI, not at
# deploy. This script registers the NGC image with AI Compute and stages the ai_runtime_task
# entrypoint, then deploys the bundle (which packages the code tarball via artifacts:) and
# runs the initial setup job (installs Glow, downloads reference genomes, and registers both
# gwas_analysis and parabricks_alignment as batch models).

IMAGE=nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1
WS_COMMAND=/Workspace/Shared/gwas_parabricks    # literal path for the ai_runtime_task command_path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE_ARG=""
[ -n "$DATABRICKS_CONFIG_PROFILE" ] && PROFILE_ARG="-p $DATABRICKS_CONFIG_PROFILE"

# --- Install the AI Runtime (air) CLI if missing (only used to register the image) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v air >/dev/null 2>&1; then
  echo ""
  echo "▶️ [GWAS] Installing the databricks-air (AI Runtime) CLI"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv tool install --force databricks-air --python 3.12
fi

# --- Register the public NGC Parabricks image with AI Compute (no credentials) ---
echo ""
echo "▶️ [GWAS] Registering public NGC image with AI Compute: $IMAGE"
set +e
air register image "$IMAGE" $PROFILE_ARG
set -e

# --- Stage the ai_runtime_task entrypoint to a literal /Workspace path ---
echo ""
echo "▶️ [GWAS] Staging alignment command.sh to $WS_COMMAND"
databricks workspace mkdirs "$WS_COMMAND"
databricks workspace import "$WS_COMMAND/command.sh" --file "$SCRIPT_DIR/airt/command.sh" --format RAW --overwrite

echo ""
echo "▶️ [GWAS] Validating bundle (target=$TARGET)"
databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [GWAS] Deploying bundle (target=$TARGET)"
databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [GWAS] Running initial setup job"
echo "🚨 This job will install Glow, download reference genomes, and register batch models. See Jobs & Pipeline tab."
echo ""
user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
databricks bundle run --target $TARGET --params "user_email=$user_email" gwas_initial_setup_job $EXTRA_PARAMS --no-wait
