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

cd app

streamlit run home.py


