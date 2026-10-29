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
# deploy. This script pushes the NGC image into the Databricks Artifact Registry (DAR) and
# stages the ai_runtime_task entrypoint, then deploys the bundle (which packages the code
# tarball via artifacts:) and runs the initial setup job (installs Glow, downloads reference
# genomes, and registers both gwas_analysis and parabricks_alignment as batch models).

IMAGE=nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1
IMAGE_TAG=4.5.1-1                               # DAR artifact tag (matches the NGC tag)
DAR_ARTIFACT=parabricks                         # DAR artifact -> <catalog>.<schema>.parabricks
WS_COMMAND=/Workspace/Shared/gwas_parabricks    # literal path for the ai_runtime_task command_path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GWB convention: catalog + schema come from application.env, passed through as --var here.
# Parse them out so we can target DAR (same artifact the parabricks submodule pushes; the
# gwas parabricks_alignment job references it via unity_catalog_image_path).
CATALOG=$(printf '%s' "$EXTRA_PARAMS" | grep -oE 'core_catalog_name=[^,"[:space:]]+' | head -1 | cut -d= -f2)
SCHEMA=$(printf '%s' "$EXTRA_PARAMS" | grep -oE 'core_schema_name=[^,"[:space:]]+' | head -1 | cut -d= -f2)
if [ -z "$CATALOG" ] || [ -z "$SCHEMA" ]; then
  echo "🚫 [GWAS] could not parse core_catalog_name/core_schema_name from --var"; exit 1
fi

# --- Push the public NGC Parabricks image into DAR (Unity Catalog; no NGC credentials) ---
# `databricks air images push` pulls the linux/amd64 image via local Docker and pushes it to
# <catalog>.<schema>.<artifact>:<tag>. Requires Docker running locally. The image is several GB;
# idempotent — a later deploy (or the parabricks submodule) reuses the cached local image.
echo ""
echo "▶️ [GWAS] Pushing $IMAGE to DAR: ${CATALOG}.${SCHEMA}.${DAR_ARTIFACT}:${IMAGE_TAG}"
databricks air images push -s "$IMAGE" --catalog "$CATALOG" --schema "$SCHEMA" --artifact "${DAR_ARTIFACT}:${IMAGE_TAG}"

# --- Stage the ai_runtime_task entrypoint to a literal /Workspace path ---
# An ai_runtime_task can't receive run-now job_parameters, so per-run inputs reach the container
# via a shared UC-Volume inbox (the app writes a JSON per run; command.sh consumes the oldest).
# The container can't build the inbox path (catalog/schema are job_parameters it never sees), so
# bake the catalog/schema-specific dir into command.sh here.
PB_INBOX_DIR="/Volumes/${CATALOG}/${SCHEMA}/gwas_data/_pb_inbox"
STAGED_CMD="$(mktemp -d)/command.sh"
sed "s#__PB_INBOX_DIR__#${PB_INBOX_DIR}#g" "$SCRIPT_DIR/airt/command.sh" > "$STAGED_CMD"
echo ""
echo "▶️ [GWAS] Staging alignment command.sh to $WS_COMMAND (inbox=$PB_INBOX_DIR)"
databricks workspace mkdirs "$WS_COMMAND"
databricks workspace import "$WS_COMMAND/command.sh" --file "$STAGED_CMD" --format RAW --overwrite

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
