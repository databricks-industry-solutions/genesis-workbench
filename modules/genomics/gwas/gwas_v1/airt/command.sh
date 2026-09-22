#!/bin/bash
# ai_runtime_task entrypoint (command_path). deploy.sh stages this to a literal
# /Workspace/Shared/gwas_parabricks/command.sh (the launcher rejects /Users/... paths); the
# DAB delivers the code tarball (artifacts: type tgz) to $CODE_SOURCE_PATH.
set -uo pipefail

# Map the UI's run-now job_parameters (if they surface as env vars) to the PB_* names
# run_alignment.py reads. If they don't reach the container, run_alignment.py falls back to a
# shared-Volume config ($PB_CONFIG) and errors clearly if required inputs are still missing.
[ -n "${fastq_r1:-}" ]              && export PB_FASTQ_R1="$fastq_r1"
[ -n "${fastq_r2:-}" ]              && export PB_FASTQ_R2="$fastq_r2"
[ -n "${reference_genome_path:-}" ] && export PB_REF="$reference_genome_path"
[ -n "${output_volume_path:-}" ]    && export PB_OUTPUT_VOLUME="$output_volume_path"
[ -n "${mlflow_run_id:-}" ]         && export PB_MLFLOW_RUN_ID="$mlflow_run_id"

echo "=== gwas alignment command.sh (CODE_SOURCE_PATH=$CODE_SOURCE_PATH) ==="
echo "params seen: FASTQ_R1=[${PB_FASTQ_R1:-}] REF=[${PB_REF:-}] mlflow_run_id=[${PB_MLFLOW_RUN_ID:-}]"

PY=""
for base in "${CODE_SOURCE_PATH:-}" /databricks/code_source "$(dirname "${BASH_SOURCE[0]}")"; do
  [ -n "$base" ] || continue
  PY=$(find "$base" -name run_alignment.py 2>/dev/null | head -1)
  [ -n "$PY" ] && break
done
if [ -z "$PY" ]; then
  echo "ERROR: run_alignment.py not found under CODE_SOURCE_PATH"; exit 3
fi
cd "$(dirname "$PY")"
exec python3 run_alignment.py
