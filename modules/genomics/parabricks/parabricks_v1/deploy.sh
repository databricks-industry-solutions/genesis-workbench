#!/bin/bash
set -e

CLOUD=$1
EXTRA_PARAMS=${@:2}   # accepted (orchestrator passes --var=...) but unused: no creds needed

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp> [--var=...]"; exit 1 ;;
esac

# Parabricks runs on serverless GPU via the AI Runtime CLI (`air run`) — the workshop
# account has EC2 GPU vCPU quota=0 (classic a10 can't provision) and classic
# custom-container clusters are gated, so serverless GPU is the only path here. A DABs
# `ai_runtime_task` can't ship the code (its launcher wants a prebuilt tarball ->
# "Tarball not found"); `air run` auto-packages airt/ (code_source snapshot -> its own
# tarball) and generates its own launcher.
#
# The image is NVIDIA's STOCK NGC Parabricks container used DIRECTLY — AI Runtime accepts
# nvcr.io and pulls the public image (verified end-to-end), so there is no Docker Hub
# mirror, custom Dockerfile, or registry credential to manage. run_parabricks.py self-stages
# the NVIDIA sample data (downloads to node-local scratch via the python stdlib) on first run.

# Must match airt/parabricks_workload.yaml's environment.docker_image.url.
IMAGE=nvcr.io/nvidia/clara/clara-parabricks:4.5.1-1
AIRT_DIR="$(cd "$(dirname "$0")" && pwd)/airt"

# `air` doesn't read DATABRICKS_CONFIG_PROFILE like the databricks CLI does — pass it
# through explicitly when set (the deploy runs with DATABRICKS_CONFIG_PROFILE exported).
PROFILE_ARG=""
[ -n "$DATABRICKS_CONFIG_PROFILE" ] && PROFILE_ARG="-p $DATABRICKS_CONFIG_PROFILE"

# --- 1) Install the AI Runtime (air) CLI if missing ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v air >/dev/null 2>&1; then
  echo ""
  echo "▶️ [Parabricks] Installing the databricks-air (AI Runtime) CLI"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv tool install --force databricks-air --python 3.12
fi

# --- 2) Register the public NGC image with AI Compute (no credentials; pulls + caches) ---
# Non-fatal: a re-run of an already-registered image is fine; don't abort the deploy.
echo ""
echo "▶️ [Parabricks] Registering public NGC image with AI Compute: $IMAGE"
echo "🚨 First registration replicates the image (a few minutes)"
set +e
air register image "$IMAGE" $PROFILE_ARG
_rc=$?
set -e
[ "$_rc" -eq 0 ] || echo "⚠️  air register returned $_rc (already registered?) — continuing"

# --- 3) Submit the serverless-GPU Parabricks workload (air run auto-packages airt/) ---
# air run returns after submit (no --watch); the run downloads the ~11.8GB NVIDIA sample
# on a cold node, runs pbrun fq2bam -> deepvariant, and persists outputs to the Volume.
echo ""
echo "▶️ [Parabricks] Submitting run_parabricks (serverless GPU_1xA10) via air run"
echo "🚨 First run downloads the ~11.8GB NVIDIA sample; monitor via: air logs <run_id> $PROFILE_ARG"
( cd "$AIRT_DIR" && air run --file parabricks_workload.yaml $PROFILE_ARG )
