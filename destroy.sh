#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>  <additional build variables>"
    echo 'Example: destroy core dev --var="dev_user_prefix=scn"'
    exit 1
fi

CWD=$1
ENV=$2
EXTRA_PARAMS=${@: 3}

echo "================================"
echo "⚙️ Preparing to destroy module $CWD"
echo "================================"

cd modules/$CWD
databricks bundle destroy -t $ENV $EXTRA_PARAMS
if [ $? -eq 0 ]; then
    echo "================================"
    echo "✅ SUCCESS! Destroy complete."
    echo "================================"
else
    echo "================================"
    echo "❗️ ERROR! Destroy failed."
    echo "================================"
fi


