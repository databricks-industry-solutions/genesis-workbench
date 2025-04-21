
#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <env> <additional build variables>"
    echo 'Example: deploy dev --var="dev_user_prefix=scn"'
    exit 1
fi

ENV=$1
EXTRA_PARAMS=${@: 2}

echo "▶️ Building libraries"
echo ""

cd library/genesis_workbench
poetry build

if [ $? -ne 0 ]; then
    echo "❗️ Error building wheel. Aborting deploy."
    exit 1
fi
cd ../../

echo "▶️ Copying library wheel file"
echo ""

cp library/genesis_workbench/dist/*.whl app/ 
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
        echo -e "\n$filename" >> app/requirements.txt
    else
        echo "Dependency already exists"
    fi

  fi
done

if [ $? -ne 0 ]; then
    echo "❗️ Error copying wheel. Aborting deploy."
    exit 1
fi

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

databricks bundle run -t $ENV genesis_workbench_app $EXTRA_PARAMS
if [ $? -ne 0 ]; then
    echo "❗️ Error. Aborting deploy."
    exit 1
fi