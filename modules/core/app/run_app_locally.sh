#!/bin/bash
set -e

echo "#############################################################"
echo "MAKE SURE core MODULE IS DEPLOYED BEFORE RUNNING APP LOCALLY"
echo "#############################################################"

read -p "Do you wish to continue? (y/n): " answer
if [[ "$answer" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Continuing..."
    # put your continuing code here

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

else
    echo "Aborted."
    exit 1
fi


