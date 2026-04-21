#!/bin/bash
#
# setup_secret_scope.sh — create the GWB Docker secret scope on a Databricks workspace.
#
# Creates a secret scope and populates it with the four Docker credentials
# needed by BioNeMo + Parabricks + (Disease Biology's GWAS alignment) modules.
#
# Naming convention documented in docs/deployments/docker-secrets-convention.md.
#
# Usage:
#   ./scripts/setup_secret_scope.sh <profile> [scope-name]
#
# Example:
#   ./scripts/setup_secret_scope.sh fevm-mmt-aws-usw2 mmt
#
# Credentials can be passed via env vars or entered interactively:
#   GWB_BIONEMO_DOCKER_USER
#   GWB_BIONEMO_DOCKER_TOKEN
#   GWB_PARABRICKS_DOCKER_USER
#   GWB_PARABRICKS_DOCKER_TOKEN
#
# Tokens never appear in the command history — read -s masks interactive input;
# env vars (if used) stay in the shell scope only.
#
set -euo pipefail

PROFILE="${1:-}"
SCOPE="${2:-gwb_docker}"

if [[ -z "$PROFILE" ]]; then
  echo "Usage: $0 <profile> [scope-name]" >&2
  echo "" >&2
  echo "Creates a Databricks secret scope and populates it with Docker credentials for" >&2
  echo "the BioNeMo + Parabricks modules. Name defaults to 'gwb_docker' if not given." >&2
  exit 1
fi

command -v databricks >/dev/null 2>&1 || { echo "❌ databricks CLI not on PATH" >&2; exit 2; }

# Verify auth on the target profile before doing anything
if ! databricks current-user me --profile "$PROFILE" >/dev/null 2>&1; then
  echo "❌ Cannot authenticate with profile '$PROFILE'. Run: databricks auth login --profile $PROFILE" >&2
  exit 3
fi

read_if_unset () {
  # $1 = env var name; $2 = prompt; $3 = 's' to mask input (for tokens), anything else for plain
  local var="$1" prompt="$2" mask="${3:-}"
  if [[ -n "${!var:-}" ]]; then
    return 0
  fi
  local val
  if [[ "$mask" == "s" ]]; then
    read -r -s -p "$prompt: " val
    echo ""
  else
    read -r -p "$prompt: " val
  fi
  eval "$var=\"\$val\""
}

read_if_unset GWB_BIONEMO_DOCKER_USER    "BioNeMo Docker user (e.g., Docker Hub username)"
read_if_unset GWB_BIONEMO_DOCKER_TOKEN   "BioNeMo Docker PAT (hidden)" s
read_if_unset GWB_PARABRICKS_DOCKER_USER "Parabricks Docker user"
read_if_unset GWB_PARABRICKS_DOCKER_TOKEN "Parabricks Docker PAT (hidden)" s

echo ""
echo "▶︎ Creating secret scope '$SCOPE' on profile '$PROFILE'..."
if databricks secrets create-scope "$SCOPE" --profile "$PROFILE" 2>/dev/null; then
  echo "  ✓ scope created"
else
  echo "  ℹ scope may already exist — continuing"
fi

echo ""
echo "▶︎ Putting 4 secrets (gwb_<module>_docker_{user,token}) into '$SCOPE'..."
databricks secrets put-secret "$SCOPE" gwb_bionemo_docker_user    --string-value "$GWB_BIONEMO_DOCKER_USER"    --profile "$PROFILE"
databricks secrets put-secret "$SCOPE" gwb_bionemo_docker_token   --string-value "$GWB_BIONEMO_DOCKER_TOKEN"   --profile "$PROFILE"
databricks secrets put-secret "$SCOPE" gwb_parabricks_docker_user --string-value "$GWB_PARABRICKS_DOCKER_USER" --profile "$PROFILE"
databricks secrets put-secret "$SCOPE" gwb_parabricks_docker_token --string-value "$GWB_PARABRICKS_DOCKER_TOKEN" --profile "$PROFILE"
echo "  ✓ 4 secrets put"

echo ""
echo "▶︎ Verify:"
databricks secrets list-secrets "$SCOPE" --profile "$PROFILE"

echo ""
echo "✅ Done."
echo ""
echo "Next: update modules/bionemo/module.env + modules/parabricks/module.env + modules/disease_biology/module.env"
echo "     to reference the secret-scope keys instead of plaintext PATs:"
echo ""
echo "     bionemo_docker_userid={{secrets/$SCOPE/gwb_bionemo_docker_user}}"
echo "     bionemo_docker_token={{secrets/$SCOPE/gwb_bionemo_docker_token}}"
echo "     parabricks_docker_userid={{secrets/$SCOPE/gwb_parabricks_docker_user}}"
echo "     parabricks_docker_token={{secrets/$SCOPE/gwb_parabricks_docker_token}}"
echo ""
echo "     Full pattern in docs/deployments/docker-secrets-convention.md."
