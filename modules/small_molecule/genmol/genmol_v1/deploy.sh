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

echo ""
echo "▶️ [GenMol] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [GenMol] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [GenMol] Running model registration job"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET register_genmol $EXTRA_PARAMS --no-wait

echo ""
echo "▶️ [GenMol] Building lookup tables (target_binders + repurposing_hub)"
echo ""

# Self-contained small-molecule lookup data (ChEMBL binders + Broad Repurposing Hub).
# Independent of the model deploy; safe to re-run on its own to refresh the tables.
databricks bundle run --target $TARGET init_genmol_data $EXTRA_PARAMS --no-wait

echo ""
echo "▶️ [GenMol] Registering the Guided Molecule Design orchestrator job"
echo "    (persists its job id + grants the app SP CAN_MANAGE_RUN so the app can dispatch it)"
echo ""

databricks bundle run --target $TARGET register_molecule_optimization_job $EXTRA_PARAMS
