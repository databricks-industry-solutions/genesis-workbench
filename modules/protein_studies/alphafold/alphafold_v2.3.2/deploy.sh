
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [AlphaFold] Creating a Volume if not exists"
echo ""

pairs=${EXTRA_PARAMS#--var=}
pairs=${pairs%\"}
pairs=${pairs#\"}

# Now: pairs="var1=val1,var2=val2"

IFS=',' read -ra items <<< "$pairs"
for item in "${items[@]}"; do
    key=${item%%=*}
    val=${item#*=}
    printf -v "$key" '%s' "$val"   # creates variable named $key with value $val
done

echo "Catalog: $core_catalog_name"
echo "Schema: $core_schema_name"

set +e
echo "databricks volumes create $core_catalog_name $core_schema_name alphafold MANAGED"
databricks volumes create $core_catalog_name $core_schema_name alphafold MANAGED
if [ "$?" -eq "0" ]
then
  echo "Created volume for Alphafold: $core_catalog_name.$core_schema_name.alphafold"
else
  echo "Volume $core_catalog_name.$core_schema_name.alphafold exists"
fi
set -e

echo ""
echo "▶️ [AlphaFold] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [AlphaFold] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

echo ""
echo "▶️ [AlphaFold] Running model file downloads"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run alphafold_register_and_downloads $EXTRA_PARAMS --no-wait