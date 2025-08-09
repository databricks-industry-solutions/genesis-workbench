#!/bin/bash

set -e 

if [ "$#" -lt 3 ]; then
    echo "Usage: destroy <module> <env> <cloud>"
    echo 'Example: destroy core dev aws'
    exit 1
fi

CWD=$1
ENV=$2
CLOUD=$3

echo "#################################################################################"
echo "ALL RESOURCES deployed as part of $CWD module will deleted from $ENV environment."
echo "This operation CANNOT be undone. "
echo "##################################################################################"
read -p "Do you wish to continue? (y/n): " answer

if [[ "$answer" =~ ^([yY][eE][sS]|[yY])$ ]]; then

    if [[ "$CWD" == "core" ]]; then
        echo "Checking for dependencies"
        find modules -type d | while read -r dir; do
            if [[ "$(basename "$dir")" == "core" ]]; then
                continue
            fi

            if [[ -e "$dir/.deployed" ]]; then
                echo "🚫 Deployment exist in $dir. Cannot remove core module"
                exit 1
            fi
        done

    fi

    cd modules/$CWD
    chmod +x destroy.sh
    ./destroy.sh $ENV $CLOUD
    echo " " 

    if [[ "$CWD" != "core" ]]; then
        cd ../core
        source env.env
        EXTRA_PARAMS_CLOUD=$(paste -sd, "$CLOUD.env")
        EXTRA_PARAMS_GENERAL=$(paste -sd, "env.env")
        EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"
        user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
        
        echo "⏩️ Running job to delete all endpoints and archive the inference tables"

        databricks bundle run -t $ENV --params "module=$CWD,destroy_user_email=$user_email" destroy_module_job --var="$EXTRA_PARAMS"
        cd ../..
    fi
    

else
    echo "Aborted."
    exit 1
fi



