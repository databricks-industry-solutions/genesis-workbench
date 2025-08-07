#!/bin/bash

set -e 

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>"
    echo 'Example: destroy core dev '
    exit 1
fi

CWD=$1
ENV=$2

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
    ./destroy.sh $ENV
    echo " " 

    if [[ "$CWD" != "core" ]]; then
        cd ../core
        source env.env
        EXTRA_PARAMS=$(paste -sd, "env.env")
        user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
        
        echo "⏩️ Running job to delete all endpoints and archive the inference tables"

        databricks bundle run -t $ENV --params "model_category=$CWD,destroy_user_email=$user_email" destroy_endpoints_job --var="$EXTRA_PARAMS"
        cd ..
    fi
    

else
    echo "Aborted."
    exit 1
fi



