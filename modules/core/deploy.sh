
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
echo "▶️ Building libraries"
echo ""

cd library/genesis_workbench
poetry build

cd ../../

echo ""
echo "▶️ Adding libraries and context information to app"
echo ""

yes | cp -rf library/genesis_workbench/dist/*.whl app/lib/
#databricks fs cp library/genesis_workbench/dist/*.whl dbfs:/Volumes/genesis_workbench/dev_srijit_nair_dbx_genesis_workbench_core/libraries

ls -al app/lib/

# Loop through all .whl files in the directory
for file in library/genesis_workbench/dist/*.whl; do
  echo "Checking $file"
  # Check if the file exists (in case there are no .whl files)
  if [ -f "$file" ]; then
    # Extract just the filename (not the full path)
    filename=$(basename "$file")
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

databricks bundle run -t $ENV genesis_workbench_job $EXTRA_PARAMS

echo ""
echo "▶️ Deploying UI Application"
echo ""

databricks bundle run -t $ENV genesis_workbench_app $EXTRA_PARAMS

#unfortunately databricks sync uses gitignore to synce files
#so we need to manualy delete the wheel files we created so that it does not
#get checked into git
echo ""
echo "▶️ Cleaning up wheel files"
echo ""
rm app/lib/*.whl
rm -rf library/genesis_workbench/dist
