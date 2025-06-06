#!/bin/bash
set -e

cd ../library/genesis_workbench
poetry build

cd ../../app

mkdir -p lib
yes | cp -rf ../library/genesis_workbench/dist/*.whl lib/

pip uninstall -y -r requirements.txt

pip install -r requirements.txt

#set the following variables
# export IS_TOKEN_AUTH="Y"
# export SQL_WAREHOUSE="8f210e00850a2c16"
# export DATABRICKS_HOSTNAME="https://adb-830292400663869.9.azuredatabricks.net"
# export DATABRICKS_TOKEN="xxx"

source env.env

rm lib/*.whl

cd ..

echo "Extra params: $EXTRA_PARAMS"

echo $EXTRA_PARAMS > app/extra_params.txt

databricks bundle deploy -t dev --var=$EXTRA_PARAMS

cd app

streamlit run home.py


