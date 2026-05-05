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
echo "▶️ [SCimilarity] Creating cache Volume if not exists (target=$TARGET)"
echo ""

pairs=${EXTRA_PARAMS#--var=}
pairs=${pairs%\"}
pairs=${pairs#\"}

IFS=',' read -ra items <<< "$pairs"
for item in "${items[@]}"; do
    key=${item%%=*}
    val=${item#*=}
    printf -v "$key" '%s' "$val"
done

: "${cache_dir:=scimilarity}"

echo "Catalog: $core_catalog_name"
echo "Schema:  $core_schema_name"
echo "Volume:  $cache_dir"

set +e
databricks volumes create "$core_catalog_name" "$core_schema_name" "$cache_dir" MANAGED
if [ "$?" -eq "0" ]; then
  echo "Created volume $core_catalog_name.$core_schema_name.$cache_dir"
else
  echo "Volume $core_catalog_name.$core_schema_name.$cache_dir already exists"
fi
set -e

echo ""
echo "▶️ [SCimilarity] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [SCimilarity] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [SCimilarity] Running model registration job"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run --target $TARGET register_scimilarity $EXTRA_PARAMS --no-wait


