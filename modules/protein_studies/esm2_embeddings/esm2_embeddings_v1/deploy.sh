
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo ">>> [ESM2 Embeddings] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo ">>> [ESM2 Embeddings] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

echo ""
echo ">>> [ESM2 Embeddings] Running model registration job"
echo "This job might take a long time to finish. See Jobs & Pipeline tab for status"
echo ""

databricks bundle run register_esm2_embeddings $EXTRA_PARAMS --no-wait

