
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: update <cloud>"
    echo 'Example: update aws'
    exit 1
fi

CLOUD=$1

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

databricks bundle validate --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Deploying bundle"
echo ""

databricks bundle deploy --var="$EXTRA_PARAMS"

echo ""
echo "▶️ Deploying UI Application"
echo ""

databricks bundle run genesis_workbench_app --var="$EXTRA_PARAMS"

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

#unfortunately databricks sync uses gitignore to sync files
#so we need to manualy delete the wheel files we created so that it does not
#get checked into git
echo ""
echo "▶️ Cleaning up wheel files"
echo ""
rm app/lib/*.whl
rm -rf library/genesis_workbench/dist

date +"%Y-%m-%d %H:%M:%S" > .deployed
