#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <cloud>"
    echo "Example: deploy aws"
    exit 1
fi

CLOUD=$1

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
else
    EXTRA_PARAMS_MODULE=''
fi

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD,$EXTRA_PARAMS_MODULE"

# BioNeMo ESM2 fine-tune + inference are containerless: they run on serverless GPU as
# notebook_task jobs (hardware_accelerator: GPU_1xA10) that read ESM-2 from Hugging Face and
# run it on Transformer Engine. No NGC container, no classic A10 cluster, no air CLI / image
# registration. Deploy just registers the jobs; the UI launches them ON DEMAND (jobs run-now).

echo ""
echo "▶️ [BioNeMo] Validating bundle"
databricks bundle validate --var="$EXTRA_PARAMS"

echo ""
echo "▶️ [BioNeMo] Deploying bundle"
databricks bundle deploy --var="$EXTRA_PARAMS"

if [[ ! -e ".deployed" ]]; then
    echo ""
    echo "▶️ [BioNeMo] Running model registration job as a backend task"
    echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
    echo ""

    user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
    databricks bundle run --params "user_email=$user_email" initial_setup_job --var="$EXTRA_PARAMS"  --no-wait
fi

date +"%Y-%m-%d %H:%M:%S" > .deployed
