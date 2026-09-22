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

# Parabricks runs on serverless GPU via the AI Runtime CLI (`air run`) — the workshop
# account has EC2 GPU vCPU quota=0 (classic a10 can't provision) and classic
# custom-container clusters are gated, so serverless GPU is the only path here. A DABs
# `ai_runtime_task` can't ship the code (its launcher wants a prebuilt tarball ->
# "Tarball not found"); `air run` auto-packages airt/ (see airt/parabricks_workload.yaml
# code_source) and generates its own launcher, so it's the mechanism. This script does the
# one-time setup — the `air` CLI, the Docker Hub auth secret, and registering the custom
# image with AI Compute — then submits the workload. run_parabricks.py self-stages the
# NVIDIA sample data (downloads to node-local scratch) on first run.

# --- Parse the incoming --var="k=v,..." into shell vars (docker creds + catalog/schema) ---
pairs=${EXTRA_PARAMS#--var=}
pairs=${pairs%\"}
pairs=${pairs#\"}
IFS=',' read -ra _items <<< "$pairs"
for _item in "${_items[@]}"; do
  _k=${_item%%=*}; _v=${_item#*=}
  printf -v "$_k" '%s' "$_v"
done

SCOPE=genesis_workbench_secret_scope
AIRT_DIR="$(cd "$(dirname "$0")" && pwd)/airt"

# `air` doesn't read DATABRICKS_CONFIG_PROFILE like the databricks CLI does — pass it
# through explicitly when it's set (the deploy is run with DATABRICKS_CONFIG_PROFILE
# exported), otherwise fall back to air's default profile.
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

# --- 2) Store the Docker Hub registry auth (base64 user:token) in the secret scope ---
# AI Compute image registration expects the --key secret to be base64("username:password").
echo ""
echo "▶️ [Parabricks] Storing Docker Hub registry auth in secret scope $SCOPE"
_auth=$(printf '%s:%s' "$parabricks_docker_userid" "$parabricks_docker_token" | base64 | tr -d '\n')
databricks secrets put-secret "$SCOPE" parabricks_docker_auth --string-value "$_auth"

# --- 3) Register the custom image with AI Compute (replicates it; can take minutes) ---
# Non-fatal: a re-run of an already-registered image is fine; don't abort the deploy.
echo ""
echo "▶️ [Parabricks] Registering image with AI Compute: $parabricks_docker_image"
echo "🚨 First registration replicates the image (~minutes for large images)"
set +e
air register image "$parabricks_docker_image" --scope "$SCOPE" --key parabricks_docker_auth $PROFILE_ARG
_rc=$?
set -e
[ "$_rc" -eq 0 ] || echo "⚠️  air register returned $_rc (already registered, or check with: air get run ...) — continuing"

# --- 4) Submit the serverless-GPU Parabricks workload (air run auto-packages airt/) ---
# air run returns after submit (no --watch); the run downloads the ~11.8GB NVIDIA sample
# on a cold node, then runs pbrun fq2bam -> deepvariant and persists outputs to the Volume.
echo ""
echo "▶️ [Parabricks] Submitting run_parabricks (serverless GPU_1xA10) via air run"
echo "🚨 First run downloads the ~11.8GB NVIDIA sample; monitor via: air logs <run_id> $PROFILE_ARG"
( cd "$AIRT_DIR" && air run --file parabricks_workload.yaml $PROFILE_ARG )
