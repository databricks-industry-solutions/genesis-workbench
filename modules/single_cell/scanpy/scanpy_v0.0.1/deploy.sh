
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [scanpy] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [scanpy] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

# Run registration job to grant app permissions
if [[ ! -e ".deployed" ]]; then
    echo ""
    echo "▶️ [scanpy] Running registration job to grant app permissions"
    echo ""
    
    databricks bundle run register_scanpy_job $EXTRA_PARAMS
    
    echo ""
    echo "▶️ [scanpy] Downloading gene reference tables"
    echo ""
    
    databricks bundle run download_gene_references_gwb $EXTRA_PARAMS
    
    echo ""
    echo "✅ [scanpy] Deployment complete"
    echo ""
    
    date +"%Y-%m-%d %H:%M:%S" > .deployed
fi