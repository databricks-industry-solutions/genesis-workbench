
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

# Run registration job to grant app permissions
if [[ ! -e ".deployed" ]]; then
    echo ""
    echo "▶️ [rapidssinglecell] Running registration job to grant app permissions"
    echo ""
    
    databricks bundle run register_rapidssinglecell_job $EXTRA_PARAMS
    
    echo ""
    echo "✅ [rapidssinglecell] Deployment complete"
    echo ""
    
    date +"%Y-%m-%d %H:%M:%S" > .deployed
fi