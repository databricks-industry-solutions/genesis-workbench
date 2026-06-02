#!/bin/bash
set -e

CLOUD=$1
EXTRA_PARAMS=${@:2}

case "$CLOUD" in
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *) echo "Usage: $0 <aws|azure|gcp> --var=..."; exit 1 ;;
esac

echo ""
echo "▶️ [AlphaFold] Creating a Volume if not exists (target=$TARGET)"
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
databricks volumes create $core_catalog_name $core_schema_name alphafold MANAGED
if [ "$?" -eq "0" ]
then
  echo "Created volume for Alphafold: $core_catalog_name.$core_schema_name.alphafold"
else
  echo "Volume $core_catalog_name.$core_schema_name.alphafold exists"
fi
set -e

echo ""
echo "▶️ [AlphaFold] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [AlphaFold] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [AlphaFold] Running model file downloads"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET alphafold_register_and_downloads $EXTRA_PARAMS --no-wait
