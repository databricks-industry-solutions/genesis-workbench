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

# Parabricks is a GWB batch workflow: deploy pushes the container image into the Databricks
# Artifact Registry (DAR, Unity Catalog), creates the parabricks data-cache volume (via the
# bundle), REGISTERS a persistent serverless-GPU job (ai_runtime_task, run_parabricks) that the
# UI launches ON DEMAND (jobs run-now), and runs the registration job (register_batch_model +
# app-SP grant). This script does NOT run pbrun — that happens when a user clicks Launch.
#
# Image = NVIDIA's stock NGC Parabricks container (public; no NGC login). It is pushed into DAR
# with `databricks air images push` and referenced by the ai_runtime_task via its Unity Catalog
# coordinates (unity_catalog_image_path), per the Databricks AI Runtime cookbook.

IMAGE=nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1
IMAGE_TAG=4.5.1-1                                  # DAR artifact tag (matches the NGC tag)
DAR_ARTIFACT=parabricks                            # DAR artifact -> <catalog>.<schema>.parabricks
WS_COMMAND=/Workspace/Shared/parabricks           # literal path for the job's command_path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GWB convention: catalog + schema come from application.env, passed through as --var here.
# Parse them out so we can target DAR (they also drive the bundle's parabricks volume + job).
CATALOG=$(printf '%s' "$EXTRA_PARAMS" | grep -oE 'core_catalog_name=[^,"[:space:]]+' | head -1 | cut -d= -f2)
SCHEMA=$(printf '%s' "$EXTRA_PARAMS" | grep -oE 'core_schema_name=[^,"[:space:]]+' | head -1 | cut -d= -f2)
if [ -z "$CATALOG" ] || [ -z "$SCHEMA" ]; then
  echo "🚫 [Parabricks] could not parse core_catalog_name/core_schema_name from --var"; exit 1
fi

# --- 1) Push the NGC image into the Databricks Artifact Registry (DAR / Unity Catalog) ---
# `databricks air images push` pulls a linux/amd64 image via local Docker and pushes it to
# <catalog>.<schema>.<artifact>:<tag>. Requires Docker running locally. The image is several GB,
# so the first push takes a while; later deploys reuse the cached local image (omit --pull).
echo ""
echo "▶️ [Parabricks] Pushing $IMAGE to DAR: ${CATALOG}.${SCHEMA}.${DAR_ARTIFACT}:${IMAGE_TAG}"
databricks air images push -s "$IMAGE" --catalog "$CATALOG" --schema "$SCHEMA" --artifact "${DAR_ARTIFACT}:${IMAGE_TAG}"

# --- 2) Stage command.sh to a literal /Workspace path (the ai_runtime_task command_path) ---
# The launcher rejects a /Users/... command_path, so it can't be a bundle file; upload it here.
echo ""
echo "▶️ [Parabricks] Staging command.sh to $WS_COMMAND"
databricks workspace mkdirs "$WS_COMMAND"
databricks workspace import "$WS_COMMAND/command.sh" --file "$SCRIPT_DIR/airt/command.sh" --format RAW --overwrite

# --- 3) Validate + deploy (creates run_parabricks job + parabricks volume; packages ./dist/code.tgz) ---
echo ""
echo "▶️ [Parabricks] Validating + deploying bundle (target=$TARGET)"
databricks bundle validate --target $TARGET $EXTRA_PARAMS
databricks bundle deploy --target $TARGET $EXTRA_PARAMS

# --- 4) Register the batch model so the UI can launch it on demand (NOT a pbrun run) ---
# Foreground: registration must finish before a user can launch from the UI.
echo ""
echo "▶️ [Parabricks] Running registration job (register_batch_model + app-SP grant)"
databricks bundle run --target $TARGET parabricks_initial_setup_job $EXTRA_PARAMS
