#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>  <additional build variables>"
    echo 'Example: destroy core dev --var="dev_user_prefix=scn"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo "⚙️ Starting destroy of module $CWD"

for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
    do
        cd $module
        ./destroy.sh $ENV $EXTRA_PARAMS
        cd ../..
    done




