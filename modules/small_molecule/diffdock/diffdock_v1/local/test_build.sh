#!/bin/bash
# Quick test: validates that all DiffDock dependencies install correctly
# in an environment matching the model serving container.
#
# Usage: cd modules/small_molecule/diffdock/diffdock_v1/local && ./test_build.sh

set -e

echo "=== Testing DiffDock Scoring Container ==="
docker build -f Dockerfile.scoring -t diffdock-scoring-test .
docker run --rm diffdock-scoring-test

echo ""
echo "=== Testing DiffDock ESM Container ==="
docker build -f Dockerfile.esm -t diffdock-esm-test .
docker run --rm diffdock-esm-test

echo ""
echo "=== Both containers built successfully! ==="
echo "The pip install will work on model serving."
