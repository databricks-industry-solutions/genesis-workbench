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
echo "▶️ [scanpy] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [scanpy] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [scanpy] Running registration job to grant app permissions"
echo ""

databricks bundle run --target $TARGET register_scanpy_job $EXTRA_PARAMS

echo ""
echo "▶️ [scanpy] Downloading gene reference tables"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET download_gene_references_gwb $EXTRA_PARAMS --no-wait

echo ""
echo "▶️ [scanpy] Downloading gene set (GMT) files for pathway enrichment"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET download_genesets_gwb $EXTRA_PARAMS --no-wait

echo ""
echo "▶️ [scanpy] Downloading CellxGene reference datasets"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET download_cellxgene_gwb $EXTRA_PARAMS --no-wait

echo ""
echo "✅ [scanpy] Deployment complete"
echo ""
