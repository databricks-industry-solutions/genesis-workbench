
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> <additional build variables>"
    echo 'Example: deploy dev --var="dev_user_prefix=scn,core_catalog_name=genesis_workbench,core_schema_name=dev_srijit_nair_dbx_genesis_workbench_core"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo "⏩️ Starting deploy of Protein Studies module"

for module in alphafold/alphafold_v2.3.2 boltz/boltz_1 esmfold/esmfold_v1 protein_mpnn/protein_mpnn_v0.1.0 rfdiffusion/rfdiffusion_v1.1.0
    do
        echo "----------------------------------"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh
        ./deploy.sh $ENV $EXTRA_PARAMS
        cd ../..
    done



