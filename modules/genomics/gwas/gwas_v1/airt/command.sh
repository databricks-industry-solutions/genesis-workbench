#!/bin/bash
# ai_runtime_task entrypoint (command_path). deploy.sh stages this to a literal
# /Workspace/Shared/gwas_parabricks/command.sh (the launcher rejects /Users/... paths); the
# DAB delivers the code tarball (artifacts: type tgz) to $CODE_SOURCE_PATH.
#
# An ai_runtime_task cannot receive run-now job_parameters as env vars — the DAB task has no
# env/param field (confirmed against `databricks bundle schema`). So per-run inputs (FASTQ /
# ref / output / mlflow_run_id) reach the container through a shared UC-Volume "inbox": the
# job's write_pb_config notebook task writes one JSON per run, keyed by the Databricks job run
# id, into $PB_INBOX_DIR. This container recovers its OWN run id from the AI runtime's
# SGC_DEBUG_INFO env var (SGC_DEBUG_INFO.run_id == {{job.run_id}}) and reads exactly that file —
# so concurrent runs can never cross-wire their inputs.
set -uo pipefail

# deploy.sh bakes the catalog/schema-specific inbox dir here — the container can't build it
# itself (catalog/schema are also job_parameters it never receives).
PB_INBOX_DIR="__PB_INBOX_DIR__"

echo "=== gwas alignment command.sh (CODE_SOURCE_PATH=$CODE_SOURCE_PATH) ==="

# This run's job run id, from the AI runtime's SGC_DEBUG_INFO (the only place the container can
# learn it — job_parameters don't reach here). write_pb_config keyed the config file by it.
RUN_ID=$(python3 -c 'import os,json; print(json.loads(os.environ.get("SGC_DEBUG_INFO","{}")).get("run_id",""))' 2>/dev/null)
echo "job run_id (from SGC_DEBUG_INFO) = [${RUN_ID}]"

CFG=""
if [ -n "$RUN_ID" ] && [ -f "$PB_INBOX_DIR/$RUN_ID.json" ]; then
  CFG="$PB_INBOX_DIR/$RUN_ID.json"                       # this run's own config — the normal path
elif [ -d "$PB_INBOX_DIR" ]; then
  # Fallback only if SGC_DEBUG_INFO ever stops carrying run_id: take the oldest pending config
  # and claim it by renaming, so a later run can't re-pick it.
  CAND=$(ls -1tr "$PB_INBOX_DIR"/*.json 2>/dev/null | head -1)
  if [ -n "${CAND:-}" ]; then
    if mv "$CAND" "$CAND.consumed" 2>/dev/null; then CAND="$CAND.consumed"; fi
    CFG="$CAND"
    echo "WARN: SGC_DEBUG_INFO.run_id unavailable — fell back to oldest pending config"
  fi
fi
if [ -n "$CFG" ]; then
  export PB_CONFIG="$CFG"                                # run_alignment.py reads $PB_CONFIG
  echo "using run config: $PB_CONFIG"
else
  echo "WARN: no run config found in $PB_INBOX_DIR — run_alignment.py will report missing inputs"
fi

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
