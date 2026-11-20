# Databricks notebook source
# MAGIC %md
# MAGIC # Update MLflow run status
# MAGIC Terminal status setter for the OpenFold3 fold job's mark_success / mark_failure tasks, so the
# MAGIC app's "Search Past Runs" always reflects a final `job_status` even if the fold task dies before
# MAGIC it can tag itself.

# COMMAND ----------

dbutils.widgets.text("mlflow_run_id", "run", "MLflow Run Id")
dbutils.widgets.text("job_status", "fold_complete", "Job Status")

mlflow_run_id = dbutils.widgets.get("mlflow_run_id")
job_status = dbutils.widgets.get("job_status")

# COMMAND ----------

import mlflow

with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.set_tag("job_status", job_status)

print(f"Set job_status={job_status} on run {mlflow_run_id}")
