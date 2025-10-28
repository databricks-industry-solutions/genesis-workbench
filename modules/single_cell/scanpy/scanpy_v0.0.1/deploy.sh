
#!/bin/bash
set -e

EXTRA_PARAMS=${@: 1}

echo ""
echo "▶️ [scanpy] Validating bundle"
echo ""

databricks bundle validate $EXTRA_PARAMS

echo ""
echo "▶️ [scanpy] Deploying bundle"
echo ""

databricks bundle deploy $EXTRA_PARAMS

# no need to run as we only wish to create the job for scanpy

# echo ""
# echo "▶️ [scanpy] creating the job definition for scanpy"
# echo ""

# databricks bundle {} $EXTRA_PARAMS --no-wait