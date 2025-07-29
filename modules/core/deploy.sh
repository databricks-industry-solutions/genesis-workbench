
#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> "
    echo 'Example: deploy dev'
    exit 1
fi

ENV=$1
#EXTRA_PARAMS=${@: 2}

# echo ""
# echo "▶️ Extracting variables"
# echo ""

# var_strs="${EXTRA_PARAMS//--var=}"

# extracted_content=$(sed 's/.*"\([^"]*\)".*/\1/' <<< "$var_strs")
# rm -f env.env
# while read -d, -r pair; do
#   IFS='=' read -r key val <<<"$pair"
#   echo "export $key=$val" >> env.env
# done <<<"$extracted_content,"

source env.env

echo ""
echo "▶️ Creating a secret scope"
echo ""

SCOPE_NAME="${ENV}_${dev_user_prefix}_genesis_workbench_application_settings_scope"

echo "Scope name: $SCOPE_NAME"

if databricks secrets list-scopes | grep -qw "$SCOPE_NAME"; then
    echo "Scope $SCOPE_NAME already exists."
else
    databricks secrets create-scope "$SCOPE_NAME"
    echo "Scope $SCOPE_NAME created."
fi

databricks secrets put-secret $SCOPE_NAME core_catalog_name --string-value $core_catalog_name
databricks secrets put-secret $SCOPE_NAME core_schema_name --string-value $core_schema_name

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

#echo $EXTRA_PARAMS > app/extra_params.txt
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

databricks bundle validate -t $ENV \
  --var="dev_user_prefix=$dev_user_prefix,core_catalog_name=$core_catalog_name,core_schema_name=$core_schema_name,bionemo_docker_token=$bionemo_docker_token"

echo ""
echo "▶️ Deploying bundle"
echo ""

databricks bundle deploy -t $ENV \
  --var="dev_user_prefix=$dev_user_prefix,core_catalog_name=$core_catalog_name,core_schema_name=$core_schema_name,bionemo_docker_token=$bionemo_docker_token"

echo ""
echo "▶️ Running initialization job"
echo ""

databricks bundle run -t $ENV initial_setup_job \
  --var="dev_user_prefix=$dev_user_prefix,core_catalog_name=$core_catalog_name,core_schema_name=$core_schema_name,bionemo_docker_token=$bionemo_docker_token"

echo ""
echo "▶️ Deploying UI Application"
echo ""

databricks bundle run -t $ENV genesis_workbench_app \
  --var="dev_user_prefix=$dev_user_prefix,core_catalog_name=$core_catalog_name,core_schema_name=$core_schema_name,bionemo_docker_token=$bionemo_docker_token"


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
