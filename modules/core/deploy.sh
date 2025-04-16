
#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> <additional build variables>"
    echo 'Example: deploy dev --var="dev_user_prefix=scn"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo "▶️ Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

if [ $? -ne 0 ]; then
    echo "❗️ Error. Aborting deploy."
    exit 1
fi

echo "▶️ Deploying bundle"
echo ""

databricks bundle deploy -t $ENV $EXTRA_PARAMS
if [ $? -ne 0 ]; then
    echo "❗️ Error. Aborting deploy."
    exit 1
fi

echo "▶️ Running initialization job"
echo ""

databricks bundle run -t $ENV genesis_workbench_job $EXTRA_PARAMS
if [ $? -ne 0 ]; then
    echo "❗️ Error. Aborting deploy."
    exit 1
fi

echo "▶️ Deploying UI Application"
echo ""

