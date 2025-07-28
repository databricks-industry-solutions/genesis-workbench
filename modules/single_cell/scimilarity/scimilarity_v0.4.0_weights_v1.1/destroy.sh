#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>  <additional build variables>"
    echo 'Example: destroy dev --var="dev_user_prefix=scn"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo "========================================================================"
echo "⚙️ Preparing to destroy module scimilarity_v0.4.0_weights_v1.1 from $ENV"
echo "========================================================================"

databricks bundle destroy -t $ENV $EXTRA_PARAMS --auto-approve

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi

