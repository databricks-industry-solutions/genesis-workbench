
#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <env> <cloud>"
    echo "Example: deploy dev aws"
    exit 1
fi

ENV=$1
CLOUD=$2

source env.env

EXTRA_PARAMS_CLOUD=$(paste -sd, "$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "env.env")
EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

echo "Extra Params: $EXTRA_PARAMS"

echo "###########################################"
echo "⏩️ Starting deploy of Single Cell module  #"

for module in scgpt/scgpt_v0.2.4 scimilarity/scimilarity_v0.4.0_weights_v1.1
    do
        echo "###########################################"
        echo "Deploying $module"
        cd $module
        chmod +x deploy.sh

        echo "Running command deploy.sh $ENV --var=\"$EXTRA_PARAMS\" " 
        ./deploy.sh $ENV --var="$EXTRA_PARAMS" 
        cd ../..
    done
echo "##############################################"

date +"%Y-%m-%d %H:%M:%S" > .deployed