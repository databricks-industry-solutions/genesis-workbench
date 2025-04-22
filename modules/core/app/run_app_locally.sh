#!/bin/bash
set -e

cd ../library/genesis_workbench
poetry build

cd ../../app

yes | cp -rf ../library/genesis_workbench/dist/*.whl lib/

pip uninstall -y -r requirements.txt

pip install -r requirements.txt
pip install mlflow

export SQL_WAREHOUSE="8f210e00850a2c16"
export DATABRICKS_HOST="https://adb-830292400663869.9.azuredatabricks.net"
export DATABRICKS_HOSTNAME="https://adb-830292400663869.9.azuredatabricks.net"
export IS_LOCAL_TEST="Y"
#SET DATABRICKS_TOKEN as below before you run
#export DATABRICKS_TOKEN="abcd1234"

streamlit run home.py
