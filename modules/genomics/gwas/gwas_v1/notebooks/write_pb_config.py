# Databricks notebook source
# Bridges this run's Variant Calling inputs into the shared UC-Volume inbox the run_germline
# ai_runtime_task container reads. An ai_runtime_task can't receive run-now job_parameters as
# env vars (the DAB task has no env/param field), but a notebook task can — so this runs first
# (run_germline depends_on it) and writes the inputs to a JSON keyed by the Databricks job run
# id. The container recovers that same id from its SGC_DEBUG_INFO env var and reads exactly this
# file, so concurrent runs never cross-wire. Runs as the job owner, who owns the gwas_data volume.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("fastq_r1", "")
dbutils.widgets.text("fastq_r2", "")
dbutils.widgets.text("reference_genome_path", "")
dbutils.widgets.text("output_volume_path", "")
dbutils.widgets.text("mlflow_run_id", "")
# Databricks substitutes {{job.run_id}} here — the same value the container sees as
# SGC_DEBUG_INFO.run_id, so both sides agree on the file name.
dbutils.widgets.text("job_run_id", "")

# COMMAND ----------

import json
import os

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
job_run_id = dbutils.widgets.get("job_run_id")

config = {
    "PB_FASTQ_R1": dbutils.widgets.get("fastq_r1"),
    "PB_FASTQ_R2": dbutils.widgets.get("fastq_r2"),
    "PB_REF": dbutils.widgets.get("reference_genome_path"),
    "PB_OUTPUT_VOLUME": dbutils.widgets.get("output_volume_path"),
    "PB_MLFLOW_RUN_ID": dbutils.widgets.get("mlflow_run_id"),
}

missing = [k for k, v in config.items() if not v]
assert not missing, f"missing required inputs for the alignment container: {missing}"
assert job_run_id, "job_run_id is empty — the task must pass job_run_id={{job.run_id}}"

# Must match the inbox dir deploy.sh bakes into command.sh (PB_INBOX_DIR).
inbox_dir = f"/Volumes/{catalog}/{schema}/gwas_data/_pb_inbox"
os.makedirs(inbox_dir, exist_ok=True)
config_path = f"{inbox_dir}/{job_run_id}.json"
with open(config_path, "w") as f:
    json.dump(config, f)

print(f"wrote {config_path}: {config}")
