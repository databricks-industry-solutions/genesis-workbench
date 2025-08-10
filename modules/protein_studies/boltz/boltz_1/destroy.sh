#!/bin/bash

EXTRA_PARAMS=${@: 1}

echo "==============================================="
echo "⚙️ Preparing to destroy module boltz_1 "
echo "==============================================="

databricks bundle destroy $EXTRA_PARAMS --auto-approve

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Destroy complete."
else
    echo "❗️ ERROR! Destroy failed."
fi

