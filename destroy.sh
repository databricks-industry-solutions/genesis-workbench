#!/bin/bash

set -e 
 
CWD=$1
CLOUD=$2

echo "#################################################################################"
echo "ALL RESOURCES deployed as part of $CWD module will deleted        "
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
        echo "Dependency check complete"

    fi

    cd modules/$CWD
    chmod +x destroy.sh
    ./destroy.sh $CLOUD
    echo " " 

    if [[ "$CWD" != "core" ]]; then
        cd ../core
        EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
        EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

        EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

        if [[ -f "module.env" ]]; then
            EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
            EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
        fi
        user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
        
        echo "⏩️ Running job to delete all endpoints and archive the inference tables"

        databricks bundle run --params "module=$CWD,destroy_user_email=$user_email" destroy_module_job --var="$EXTRA_PARAMS"
        cd ../..
    fi
    

else
    echo "Aborted."
    exit 1
fi



