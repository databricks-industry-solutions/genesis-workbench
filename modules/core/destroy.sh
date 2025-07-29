#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> "
    echo 'Example: destroy dev'
    exit 1
fi

ENV=$1

source env.env

echo "=========================================================="
echo "⚙️ Preparing to destroy module core from $ENV"
echo "=========================================================="

databricks bundle destroy -t $ENV \
    --var="dev_user_prefix=$dev_user_prefix,core_catalog_name=$core_catalog_name,core_schema_name=$core_schema_name,bionemo_docker_token=$bionemo_docker_token" \
    --auto-approve

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi

