
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [rapidssinglecell] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [rapidssinglecell] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

echo ""
echo "▶️ [rapidssinglecell] Running registration job to grant app permissions"
echo ""

databricks bundle run register_rapidssinglecell_job $EXTRA_PARAMS

echo ""
echo "▶️ [rapidssinglecell] Downloading gene reference tables"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run download_gene_references_gwb $EXTRA_PARAMS --no-wait

echo ""
echo "✅ [rapidssinglecell] Deployment complete"
echo ""
