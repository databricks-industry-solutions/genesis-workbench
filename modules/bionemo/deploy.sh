#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cloud>"
    echo "Example: deploy aws"
    exit 1
fi

CLOUD=$1

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
else
    EXTRA_PARAMS_MODULE=''
fi

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD,$EXTRA_PARAMS_MODULE"

# BioNeMo ESM2 fine-tune + inference run on serverless GPU via ai_runtime_task, using the STOCK
# NGC BioNeMo image DIRECTLY (public; AI Runtime pulls it — no Docker Hub mirror or credentials).
# This registers the image, stages the ai_runtime_task entrypoints to a literal /Workspace path
# (the launcher rejects /Users/... command_path), deploys the bundle (which packages the code
# tarball via artifacts:), and runs the registration job. The finetune/inference jobs are launched
# ON DEMAND from the UI (jobs run-now), not at deploy.

IMAGE=nvcr.io/nvidia/clara/bionemo-framework:2.6.1
WS=/Workspace/Shared/bionemo
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
PROFILE_ARG=""
[ -n "$DATABRICKS_CONFIG_PROFILE" ] && PROFILE_ARG="-p $DATABRICKS_CONFIG_PROFILE"

# --- Install the AI Runtime (air) CLI if missing (only to register the image) ---
if ! command -v air >/dev/null 2>&1; then
  echo ""
  echo "▶️ [BioNeMo] Installing the databricks-air (AI Runtime) CLI"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv tool install --force databricks-air --python 3.12
fi

# --- Register the public NGC BioNeMo image with AI Compute (no credentials) ---
echo ""
echo "▶️ [BioNeMo] Registering public NGC image with AI Compute: $IMAGE"
echo "🚨 First registration replicates a large image (several minutes)"
set +e
air register image "$IMAGE" $PROFILE_ARG
set -e

# --- Stage the ai_runtime_task entrypoints to a literal /Workspace path ---
echo ""
echo "▶️ [BioNeMo] Staging ai_runtime_task entrypoints to $WS"
databricks workspace mkdirs "$WS"
for f in command_common.sh command_finetune.sh command_inference.sh; do
  databricks workspace import "$WS/$f" --file "$SCRIPT_DIR/airt/$f" --format RAW --overwrite
done

echo ""
echo "▶️ [BioNeMo] Validating bundle"
databricks bundle validate --var="$EXTRA_PARAMS"

echo ""
echo "▶️ [BioNeMo] Deploying bundle"
databricks bundle deploy --var="$EXTRA_PARAMS"

if [[ ! -e ".deployed" ]]; then
    echo ""
    echo "▶️ [BioNeMo] Running model registration job as a backend task"
    echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
    echo ""

    user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
    databricks bundle run --params "user_email=$user_email" initial_setup_job --var="$EXTRA_PARAMS"  --no-wait
fi

date +"%Y-%m-%d %H:%M:%S" > .deployed
