
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> <additional build variables>"
    echo 'Example: deploy dev --var="dev_user_prefix=scn,core_catalog_name=genesis_workbench,core_schema_name=dev_srijit_nair_dbx_genesis_workbench_core"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo "⏩️ Starting deploy of Single Cell module"

for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
    do
        echo "----------------------------------"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh
        ./deploy.sh $ENV $EXTRA_PARAMS
        cd ../..
    done



