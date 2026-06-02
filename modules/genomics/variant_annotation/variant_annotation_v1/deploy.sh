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
echo "▶️ [Variant Annotation] Validating bundle (target=$TARGET)"
echo ""

databricks bundle validate --target $TARGET $EXTRA_PARAMS

echo ""
echo "▶️ [Variant Annotation] Deploying bundle (target=$TARGET)"
echo ""

databricks bundle deploy --target $TARGET $EXTRA_PARAMS --force

echo ""
echo "▶️ [Variant Annotation] Uploading ACMG gene panel BED file to volume"
echo ""

# Parse catalog/schema from the --var params for the fs cp command
eval $(echo "$EXTRA_PARAMS" | tr ',' '\n' | grep -E '^core_catalog_name=|^core_schema_name=' | sed 's/^/export /')

databricks fs mkdirs "dbfs:/Volumes/$core_catalog_name/$core_schema_name/variant_annotation_reference/acmg"
databricks fs cp data/ACMG_SFv3.2_GRCh38.bed "dbfs:/Volumes/$core_catalog_name/$core_schema_name/variant_annotation_reference/acmg/ACMG_SFv3.2_GRCh38.bed" --overwrite

echo ""
echo "▶️ [Variant Annotation] Uploading sample pathogenic VCF to data volume"
echo ""

databricks fs mkdirs "dbfs:/Volumes/$core_catalog_name/$core_schema_name/variant_annotation_data/sample"
databricks fs cp data/brca_pathogenic_corrected.vcf "dbfs:/Volumes/$core_catalog_name/$core_schema_name/variant_annotation_data/sample/brca_pathogenic_corrected.vcf" --overwrite

echo ""
echo "▶️ [Variant Annotation] Running initial setup job"
echo "🚨 This job will download ClinVar and set up reference data. See Jobs & Pipeline tab for status"
echo ""

user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
databricks bundle run --target $TARGET --params "user_email=$user_email" variant_annotation_initial_setup_job $EXTRA_PARAMS --no-wait

