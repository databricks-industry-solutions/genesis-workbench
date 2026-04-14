#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [Variant Annotation] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [Variant Annotation] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS --force

echo ""
echo "▶️ [Variant Annotation] Uploading ACMG gene panel BED file to volume"
echo ""

# Parse catalog/schema from the --var params for the fs cp command
eval $(echo "$EXTRA_PARAMS" | tr ',' '\n' | grep -E '^core_catalog_name=|^core_schema_name=' | sed 's/^/export /')

databricks fs mkdirs "dbfs:/Volumes/$core_catalog_name/$core_schema_name/variant_annotation_reference/acmg"
databricks fs cp data/ACMG_SFv3.2_GRCh38.bed "dbfs:/Volumes/$core_catalog_name/$core_schema_name/variant_annotation_reference/acmg/ACMG_SFv3.2_GRCh38.bed" --overwrite

echo ""
echo "▶️ [Variant Annotation] Running initial setup job"
echo "🚨 This job will download ClinVar and set up reference data. See Jobs & Pipeline tab for status"
echo ""

user_email=$(databricks current-user me | jq '.emails[0].value' | tr -d '"')
databricks bundle run --params "user_email=$user_email" variant_annotation_initial_setup_job $EXTRA_PARAMS --no-wait

