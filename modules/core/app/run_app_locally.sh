#!/bin/bash
set -e

cd ../library/genesis_workbench
poetry build

cd ../../app

yes | cp -rf ../library/genesis_workbench/dist/*.whl lib/

pip uninstall -y -r requirements.txt

pip install -r requirements.txt

#set the following variables
export IS_LOCAL_TEST="Y"
#export SQL_WAREHOUSE="8f210e00850a2c16"
#export DATABRICKS_HOSTNAME="https://adb-830292400663869.9.azuredatabricks.net"
#export DATABRICKS_TOKEN="aaaa"
source env.env

rm lib/*.whl

streamlit run home.py


