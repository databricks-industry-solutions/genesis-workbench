#!/bin/bash
# Shared ai_runtime_task setup for the BioNeMo ESM2 jobs. Wrappers (command_finetune.sh /
# command_inference.sh) call this with the entry script name. deploy.sh stages all three to
# /Workspace/Shared/bionemo/ (command_path must be a literal /Workspace path); the code tarball
# (artifacts: type tgz) delivers the run_*.py + the patched finetune_esm2.py to $CODE_SOURCE_PATH.
set -uo pipefail

ENTRY="${1:?entry script name required}"
CS="${CODE_SOURCE_PATH:-/databricks/code_source}"

# Deps not in the base NGC BioNeMo image (the notebooks pip-installed these).
pip install --no-cache-dir databricks-sql-connector==4.0.3 mlflow==2.22.0 >/dev/null 2>&1 \
  || echo "[deps] pip install warning (continuing)"

# Apply NVIDIA's finetune_esm2.py bugfix onto the STOCK image at runtime (the custom Dockerfile
# baked this in via COPY; we use the stock NGC image + overwrite the same path from code_source).
PATCH_DST=/usr/local/lib/python3.12/dist-packages/bionemo/esm2/scripts/finetune_esm2.py
SRC=$(find "$CS" -name finetune_esm2.py 2>/dev/null | head -1)
if [ -n "$SRC" ] && [ -f "$PATCH_DST" ]; then
  cp "$SRC" "$PATCH_DST" && echo "[patch] applied finetune_esm2.py -> $PATCH_DST"
fi

# Map run-now job_parameters (if they surface as env) to the PB_* names the run_*.py read.
mapv(){ local v="${!1:-}"; [ -n "$v" ] && export "$2=$v"; }
mapv core_catalog PB_CATALOG;       mapv core_schema PB_SCHEMA
mapv sql_warehouse_id PB_SQL_WAREHOUSE_ID
mapv esm_variant PB_ESM_VARIANT;    mapv task_type PB_TASK_TYPE
mapv user_email PB_USER_EMAIL;      mapv mlflow_run_id PB_MLFLOW_RUN_ID
mapv experiment_name PB_EXPERIMENT_NAME
# finetune params
mapv train_data_location PB_TRAIN_DATA;   mapv validation_data_location PB_VAL_DATA
mapv finetune_label PB_FINETUNE_LABEL;    mapv model_volume PB_MODEL_VOLUME
mapv mlp_ft_dropout PB_MLP_DROPOUT;       mapv mlp_hidden_size PB_MLP_HIDDEN
mapv mlp_target_size PB_MLP_TARGET;       mapv num_steps PB_NUM_STEPS
mapv lr PB_LR;                            mapv lr_multiplier PB_LR_MULTIPLIER
mapv micro_batch_size PB_MICRO_BATCH_SIZE; mapv precision PB_PRECISION
# inference params
mapv is_base_model PB_IS_BASE_MODEL;      mapv finetune_run_id PB_FINETUNE_RUN_ID
mapv data_location PB_DATA_LOCATION;      mapv sequence_column_name PB_SEQUENCE_COLUMN
mapv result_location PB_RESULT_LOCATION;  mapv run_name PB_RUN_NAME

echo "=== bionemo command_common: entry=$ENTRY CODE_SOURCE_PATH=$CS ==="
PY=$(find "$CS" -name "$ENTRY" 2>/dev/null | head -1)
[ -z "$PY" ] && { echo "ERROR: $ENTRY not found under $CS"; exit 3; }
cd "$(dirname "$PY")"
exec python3 "$ENTRY"
