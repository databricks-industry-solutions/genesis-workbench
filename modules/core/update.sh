
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: update <cloud>"
    echo 'Example: update aws'
    exit 1
fi

CLOUD=$1

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp>"; exit 1 ;;
esac

source module.env
source ../../application.env

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

EXTRA_PARAMS_CLOUD=$(paste -sd, "../../$CLOUD.env")
EXTRA_PARAMS_GENERAL=$(paste -sd, "../../application.env")

EXTRA_PARAMS="$EXTRA_PARAMS_GENERAL,$EXTRA_PARAMS_CLOUD"

if [[ -f "module.env" ]]; then
    EXTRA_PARAMS_MODULE=$(paste -sd, "module.env")
    EXTRA_PARAMS="$EXTRA_PARAMS,$EXTRA_PARAMS_MODULE"
fi

echo "Extra Params: $EXTRA_PARAMS"

echo ""
echo "▶️ Validating bundle"
echo ""

databricks bundle validate --target $TARGET --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Deploying UI Application"
echo ""

databricks bundle run --target $TARGET genesis_workbench_app --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Granting app service principal access to catalog"
echo ""

app_sp_id=$(databricks apps get $app_name --output json | jq -r '.service_principal_client_id')
echo "App service principal: $app_sp_id"

databricks grants update catalog $core_catalog_name --json "{\"changes\": [{\"principal\": \"$app_sp_id\", \"add\": [\"USE_CATALOG\"]}]}"
databricks grants update schema $core_catalog_name.$core_schema_name --json "{\"changes\": [{\"principal\": \"$app_sp_id\", \"add\": [\"USE_SCHEMA\", \"SELECT\", \"MODIFY\"]}]}"

echo "Catalog and schema permissions granted."

echo ""
echo "▶️ Granting app permissions for endpoints and jobs"
echo ""

databricks bundle run --target $TARGET grant_app_permissions_job --var="$EXTRA_PARAMS"

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

# Copy Glow library files (JAR + wheel) to UC Volume
for file in library/glow/*; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Copying $filename to dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename"
    databricks fs cp "$file" dbfs:/Volumes/$core_catalog_name/$core_schema_name/libraries/$filename --overwrite
  fi
done

#unfortunately databricks sync uses gitignore to sync files
#so we need to manualy delete the wheel files we created so that it does not
#get checked into git
echo ""
echo "▶️ Cleaning up wheel files"
echo ""
rm app/lib/*.whl
rm -rf library/genesis_workbench/dist

date +"%Y-%m-%d %H:%M:%S" > .deployed
