
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> <additional build variables>"
    echo 'Example: deploy dev --var="dev_user_prefix=scn,core_catalog_name=genesis_workbench,core_schema_name=dev_srijit_nair_dbx_genesis_workbench_core"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo ""
echo "▶️ [RFdiffusion] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [RFdiffusion] Deploying bundle"
echo ""

databricks bundle deploy -t $ENV $EXTRA_PARAMS

echo ""
echo "▶️ [RFdiffusion] Running model registration job"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""
databricks bundle run -t $ENV register_rfdiffusion $EXTRA_PARAMS --no-wait

