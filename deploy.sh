#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_working_directory>"
    exit 1
fi

CWD=$1
cd $CWD

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

databricks bundle validate
if [ $? -eq 0 ]; then
    databricks bundle deploy
else
    echo "Validation failed. Deployment aborted."
fi