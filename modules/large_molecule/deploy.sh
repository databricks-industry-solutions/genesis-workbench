#!/bin/bash
set -e

CLOUD=$1
shift

ONLY_SUBMODULE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-submodule) ONLY_SUBMODULE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Hugging Face token (RFD4-Proteina weight pulls) ─────────────────────────────────
# The token lives in the gitignored module.env (never committed) and is seeded into the
# Databricks secret scope here at deploy time — the RFD4 register / finetune / antibody /
# vaccine jobs read the weights from that secret. Mirrors the core module's env->secret
# pattern. The weights repo is currently PUBLIC so this is optional; wiring it keeps gated
# revisions / private forks working with no code change. (hf_token is a non-exported shell
# var, so it is NOT inherited by the bundle/submodule child processes.)
HF_TOKEN_SCOPE="${rfd4_hf_token_scope:-dbx_genesis_workbench}"
HF_TOKEN_KEY="${rfd4_hf_token_key:-hf_token}"
if [[ -f "module.env" ]]; then
    # shellcheck disable=SC1091
    source module.env
fi
if [[ -n "${hf_token:-}" ]]; then
    echo "▶️ Seeding Hugging Face token into secret ${HF_TOKEN_SCOPE}/${HF_TOKEN_KEY} (from module.env)"
    databricks secrets put-secret "$HF_TOKEN_SCOPE" "$HF_TOKEN_KEY" --string-value "$hf_token"
fi

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

if [[ -f "module.env" ]]; then
    # Pass module.env entries as bundle --vars, but NEVER the raw hf_token secret value
    # (it is seeded as a secret above, not a bundle variable). Also drop comments/blank lines.
    EXTRA_PARAMS_MODULE=$(grep -vE '^[[:space:]]*(hf_token[[:space:]]*=|#|$)' module.env | paste -sd, -)
    [[ -n "$EXTRA_PARAMS_MODULE" ]] && EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
fi

echo "Extra Params: $EXTRA_PARAMS"

echo "##############################################"
echo "⏩️ Starting deploy of Large Molecule module #"

ALL_SUBMODULES=(alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0 esm2_embeddings/esm2_embeddings_v1 sequence_search/sequence_search_v1 enzyme_optimization/enzyme_optimization_v1 rfd4_proteina/rfd4_proteina_v1 antibody_design/antibody_design_v1 vaccine_immunogen/vaccine_immunogen_v1)

if [[ -n "$ONLY_SUBMODULE" ]]; then
    found=false
    for s in "${ALL_SUBMODULES[@]}"; do
        if [[ "$s" == "$ONLY_SUBMODULE" ]]; then found=true; break; fi
    done
    if [[ "$found" != "true" ]]; then
        echo "Error: --only-submodule must be one of: ${ALL_SUBMODULES[*]}"
        exit 1
    fi
    SUBMODULES=("$ONLY_SUBMODULE")
else
    SUBMODULES=("${ALL_SUBMODULES[@]}")
fi

for module in "${SUBMODULES[@]}"
    do
        echo "###########################################"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh

        echo "Running command deploy.sh $CLOUD --var=\"$EXTRA_PARAMS\" "
        ./deploy.sh $CLOUD --var="$EXTRA_PARAMS"
        cd ../..
    done

echo "##############################################"

if [[ -z "$ONLY_SUBMODULE" ]]; then
    date +"%Y-%m-%d %H:%M:%S" > .deployed
fi
