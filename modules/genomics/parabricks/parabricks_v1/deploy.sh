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

# Parabricks is a GWB batch workflow: deploy REGISTERS a persistent serverless-GPU job
# (ai_runtime_task, run_parabricks) that the UI launches ON DEMAND (jobs run-now). This
# script does NOT run pbrun — it registers the image, deploys the job (bundle packages the
# code tarball via artifacts:), and runs the registration job (register_batch_model +
# app-SP grant). The pbrun run happens when a user clicks Launch in the UI.
#
# Image = NVIDIA's stock NGC Parabricks container, used directly (AI Runtime accepts nvcr.io
# and pulls the public image — no Docker Hub mirror, custom Dockerfile, or credentials).

IMAGE=nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1
WS_COMMAND=/Workspace/Shared/parabricks           # literal path for the job's command_path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# `air` doesn't read DATABRICKS_CONFIG_PROFILE like the databricks CLI — pass it through.
PROFILE_ARG=""
[ -n "$DATABRICKS_CONFIG_PROFILE" ] && PROFILE_ARG="-p $DATABRICKS_CONFIG_PROFILE"

# --- 1) Install the AI Runtime (air) CLI if missing (used only to register the image) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v air >/dev/null 2>&1; then
  echo ""
  echo "▶️ [Parabricks] Installing the databricks-air (AI Runtime) CLI"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv tool install --force databricks-air --python 3.12
fi

# --- 2) Register the public NGC image with AI Compute (no credentials; pulls + caches) ---
echo ""
echo "▶️ [Parabricks] Registering public NGC image with AI Compute: $IMAGE"
set +e
air register image "$IMAGE" $PROFILE_ARG
_rc=$?
set -e
[ "$_rc" -eq 0 ] || echo "⚠️  air register returned $_rc (already registered?) — continuing"

# --- 3) Stage command.sh to a literal /Workspace path (the ai_runtime_task command_path) ---
# The launcher rejects a /Users/... command_path, so it can't be a bundle file; upload it here.
echo ""
echo "▶️ [Parabricks] Staging command.sh to $WS_COMMAND"
databricks workspace mkdirs "$WS_COMMAND"
databricks workspace import "$WS_COMMAND/command.sh" --file "$SCRIPT_DIR/airt/command.sh" --format RAW --overwrite

# --- 4) Validate + deploy (creates run_parabricks job; bundle packages ./dist/code.tgz) ---
echo ""
echo "▶️ [Parabricks] Validating + deploying bundle (target=$TARGET)"
databricks bundle validate --target $TARGET $EXTRA_PARAMS
databricks bundle deploy --target $TARGET $EXTRA_PARAMS

# --- 5) Register the batch model so the UI can launch it on demand (NOT a pbrun run) ---
# Foreground: registration must finish before a user can launch from the UI.
echo ""
echo "▶️ [Parabricks] Running registration job (register_batch_model + app-SP grant)"
databricks bundle run --target $TARGET parabricks_initial_setup_job $EXTRA_PARAMS
