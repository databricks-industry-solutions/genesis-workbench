# Databricks notebook source
# Failure-finalize task. Triggered by the scanpy job's `tag_failed` task with
# run_if=AT_LEAST_ONE_FAILED so it only fires when one of the four analyze
# routes (run_lt_small/medium/large/run_gt_large) crashes.
#
# Why this exists: the dispatcher pre-creates an MLflow run with
# tags.job_status='started'. The analyze notebook flips it to 'complete' at
# the end of its run. If the analyze notebook crashes mid-flight, the final
# cell never runs and the run stays 'started' indefinitely, which the
# Search Past Runs UI then shows as 🟩⬜⬜ forever. This task fills that gap.

import mlflow

mlflow.set_tracking_uri("databricks")

dbutils.widgets.text("mlflow_run_id", "", "MLflow Run Id (pre-created by dispatcher)")

mlflow_run_id = dbutils.widgets.get("mlflow_run_id") or ""
if not mlflow_run_id:
    print("[tag_run_failed] No mlflow_run_id provided — nothing to update.")
else:
    try:
        client = mlflow.tracking.MlflowClient()
        client.set_tag(mlflow_run_id, "job_status", "failed")
        print(f"[tag_run_failed] Tagged run {mlflow_run_id} as failed.")
    except Exception as e:
        print(f"[tag_run_failed] Failed to tag run {mlflow_run_id}: {e}")
        raise
