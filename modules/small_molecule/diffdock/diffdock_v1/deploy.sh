
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [DiffDock] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [DiffDock] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

echo ""
echo "▶️ [DiffDock] Running model registration job"
echo "🚨 This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run register_diffdock $EXTRA_PARAMS --no-wait
