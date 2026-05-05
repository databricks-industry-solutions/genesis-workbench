#!/bin/bash

set -e 
 
CWD=$1
CLOUD=$2
shift 2

ONLY_SUBMODULE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-submodule)
            ONLY_SUBMODULE="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1"; exit 1
            ;;
    esac
done

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: destroy <module> <aws|azure|gcp> [--only-submodule <path>]"; exit 1 ;;
esac

if [[ -n "$ONLY_SUBMODULE" && ! -d "modules/$CWD/$ONLY_SUBMODULE" ]]; then
    echo "🚫 Submodule '$ONLY_SUBMODULE' not found in modules/$CWD/"
    echo "    (atomic modules like core, bionemo, parabricks have no submodules)"
    exit 1
fi

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
    if [[ -n "$ONLY_SUBMODULE" ]]; then
        ./destroy.sh $CLOUD --only-submodule "$ONLY_SUBMODULE"
    else
        ./destroy.sh $CLOUD
    fi
    echo " "

    # Skip the parent-level destroy_module_job for partial (--only-submodule) destroys
    # because it deletes ALL serving endpoints registered under module_category=$CWD,
    # which would clobber endpoints from submodules the user didn't ask to destroy.
    if [[ "$CWD" != "core" && -z "$ONLY_SUBMODULE" ]]; then
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

        databricks bundle run --target $TARGET --params "module=$CWD,destroy_user_email=$user_email" destroy_module_job --var="$EXTRA_PARAMS"
        cd ../..
    fi
    

else
    echo "Aborted."
    exit 1
fi



