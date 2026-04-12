
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [Proteina-Complexa] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [Proteina-Complexa] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

echo ""
echo "▶️ [Proteina-Complexa] Running model registration job"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run register_proteina_complexa $EXTRA_PARAMS --no-wait
