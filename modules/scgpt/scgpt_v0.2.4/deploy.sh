
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> <additional build variables>"
    echo 'Example: deploy dev --var="dev_user_prefix=yyang,core_catalog_name=genesis_workbench,core_schema_name=dev_srijit_nair_dbx_genesis_workbench_core"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo ""
echo "▶️ Extracting variables"
echo ""

var_strs="${EXTRA_PARAMS//--var=}"

extracted_content=$(sed 's/.*"\([^"]*\)".*/\1/' <<< "$var_strs")
rm -f env.env
while read -d, -r pair; do
  IFS='=' read -r key val <<<"$pair"
  echo "export $key=$val" >> env.env
done <<<"$extracted_content,"

source env.env
rm -f env.env

echo ""
echo "▶️ Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ Deploying bundle"
echo ""

databricks bundle deploy -t $ENV $EXTRA_PARAMS

echo ""
echo "▶️ Running model registration job"
echo ""

databricks bundle run -t $ENV register_scgpt $EXTRA_PARAMS