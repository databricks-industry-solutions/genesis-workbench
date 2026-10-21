#!/bin/bash
# ai_runtime_task entrypoint (command_path). deploy.sh stages this to a literal
# /Workspace/Shared/parabricks/command.sh (the launcher rejects /Users/... paths); the DAB
# also delivers the code tarball (artifacts: type tgz) to $CODE_SOURCE_PATH. WORKDIR isn't
# honored on AIR, so locate run_parabricks.py under the extracted code.
set -uo pipefail

# If the UI's jobs run-now job_parameters surface as env vars, map them to the PB_* names
# run_parabricks.py reads. (If they don't reach the container, these are unset and
# run_parabricks.py falls back to its shared-Volume config / sample defaults.)
[ -n "${pb_fq1:-}" ]         && export PB_FQ1="$pb_fq1"
[ -n "${pb_fq2:-}" ]         && export PB_FQ2="$pb_fq2"
[ -n "${pb_ref:-}" ]         && export PB_REF="$pb_ref"
[ -n "${pb_known_sites:-}" ] && export PB_KNOWN_SITES="$pb_known_sites"
[ -n "${mlflow_run_id:-}" ]  && export PB_MLFLOW_RUN_ID="$mlflow_run_id"

echo "=== parabricks command.sh (CODE_SOURCE_PATH=$CODE_SOURCE_PATH) ==="
echo "params seen: PB_FQ1=[${PB_FQ1:-}] PB_REF=[${PB_REF:-}] mlflow_run_id=[${mlflow_run_id:-}]"

PY=""
for base in "${CODE_SOURCE_PATH:-}" /databricks/code_source "$(dirname "${BASH_SOURCE[0]}")"; do
  [ -n "$base" ] || continue
  PY=$(find "$base" -name run_parabricks.py 2>/dev/null | head -1)
  [ -n "$PY" ] && break
done
if [ -z "$PY" ]; then
  echo "ERROR: run_parabricks.py not found under CODE_SOURCE_PATH"; exit 3
fi
cd "$(dirname "$PY")"
exec python3 run_parabricks.py
