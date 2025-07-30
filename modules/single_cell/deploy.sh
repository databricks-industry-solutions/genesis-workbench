
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> "
    echo 'Example: deploy dev '
    exit 1
fi

ENV=$1

source env.env

EXTRA_PARAMS=$(paste -sd, "env.env")

echo "Extra Params: $EXTRA_PARAMS"

echo "⏩️ Starting deploy of Single Cell module"

for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
    do
        echo "----------------------------------"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh
        ./deploy.sh $ENV --var="$EXTRA_PARAMS" 
        cd ../..
    done



