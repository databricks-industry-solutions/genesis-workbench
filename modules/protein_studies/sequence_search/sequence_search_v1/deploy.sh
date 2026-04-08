
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo ">>> [Sequence Search] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo ">>> [Sequence Search] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

echo ""
echo ">>> [Sequence Search] Running sequence search workflow"
echo "This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run sequence_search_workflow $EXTRA_PARAMS --no-wait

