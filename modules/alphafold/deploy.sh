
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
echo "▶️ Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ Deploying bundle"
echo ""

databricks bundle deploy -t $ENV $EXTRA_PARAMS

echo ""
echo "▶️ Running model file downloads"
echo ""

databricks bundle run -t $ENV alphafold_downloads $EXTRA_PARAMS

