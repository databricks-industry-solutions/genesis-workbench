#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_working_directory> <env>  <additional build variables>"
    echo 'Example: deploy core dev --var="dev_user_prefix=scn"'
    exit 1
fi

CWD=$1
ENV=$2
EXTRA_PARAMS=${@: 3}

echo "Installing Poetry"

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

echo "================================"
echo "⚙️ Preparing to deploy module $CWD"
echo "================================"

cd modules/$CWD
chmod +x deploy.sh
./deploy.sh $ENV $EXTRA_PARAMS

if [ $? -eq 0 ]; then
    echo "================================"
    echo "✅ SUCCESS! Deployment complete."
    echo "================================"
else
    echo "================================"
    echo "❗️ ERROR! Deployment failed."
    echo "================================"
fi


