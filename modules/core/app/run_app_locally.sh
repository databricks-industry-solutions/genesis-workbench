#!/bin/bash
set -e

cd ../library/genesis_workbench
poetry build

cd ../../app

mkdir -p lib
yes | cp -rf ../library/genesis_workbench/dist/*.whl lib/

pip uninstall -y -r requirements.txt

pip install -r requirements.txt

source env.env

rm lib/*.whl

cd ..

echo "Extra params: $EXTRA_PARAMS"

echo $EXTRA_PARAMS > app/extra_params.txt

databricks bundle deploy -t dev \
    --var="dev_user_prefix=$DEV_USER_PREFIX,core_catalog_name=$CORE_CATALOG_NAME,core_schema_name=$CORE_SCHEMA_NAME,bionemo_docker_token=$BIONEMO_DOCKER_TOKEN"

cd app

streamlit run home.py


