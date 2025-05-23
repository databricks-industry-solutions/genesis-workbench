
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> <additional build variables>"
    echo 'Example: deploy dev --var="dev_user_prefix=scn,core_catalog_name=genesis_workbench,core_schema_name=dev_srijit_nair_dbx_genesis_workbench_core"'
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
echo "▶️ Building libraries"
echo ""

cd library/genesis_workbench
poetry build

cd ../../

echo ""
echo "▶️ Adding libraries and context information to app"
echo ""

mkdir -p app/lib

# Loop through all .whl files in the directory
for file in library/genesis_workbench/dist/*.whl; do
  echo "Checking $file"
  # Check if the file exists (in case there are no .whl files)
  if [ -f "$file" ]; then
    # Extract just the filename (not the full path)
    filename=$(basename "$file")
    cp -rf library/genesis_workbench/dist/$filename app/lib/

    echo "Checking if $filename exists as dependency"

    if ! grep -wq "$filename" app/requirements.txt; then        
        echo "Adding $filename to the app dependency"
        # Append the filename to the output file 
        echo -e "\nlib/$filename" >> app/requirements.txt
    else
        echo "Dependency already exists"
    fi

  fi
done

echo $EXTRA_PARAMS > app/extra_params.txt

echo ""
echo "▶️ Creating schema if not exists"
echo ""

set +e
databricks schemas get $core_catalog_name.$core_schema_name
if [ "$?" -eq "0" ]
then
  echo "Schema $core_catalog_name.$core_schema_name already exists"
else
  echo "Schema $core_catalog_name.$core_schema_name does not exist.Creating.."
  databricks schemas create $core_schema_name $core_catalog_name
fi
set -e

echo ""
echo "▶️ Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ Deploying bundle"
echo ""

databricks bundle deploy -t $ENV $EXTRA_PARAMS

echo ""
echo "▶️ Running initialization job"
echo ""

databricks bundle run -t $ENV initial_setup_job $EXTRA_PARAMS

echo ""
echo "▶️ Deploying UI Application"
echo ""

databricks bundle run -t $ENV genesis_workbench_app $EXTRA_PARAMS


echo ""
echo "▶️ Copying libraries to UC Volume"
echo ""

# Loop through all .whl files in the directory
for file in library/genesis_workbench/dist/*.whl; do
  echo "Checking $file"
  # Check if the file exists (in case there are no .whl files)
  if [ -f "$file" ]; then
    # Extract just the filename (not the full path)
    filename=$(basename "$file")
    echo "Copying $filename to dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename"
    databricks fs cp library/genesis_workbench/dist/$filename dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename --overwrite
  fi
done

#unfortunately databricks sync uses gitignore to synce files
#so we need to manualy delete the wheel files we created so that it does not
#get checked into git
echo ""
echo "▶️ Cleaning up wheel files"
echo ""
rm app/lib/*.whl
rm -rf library/genesis_workbench/dist
