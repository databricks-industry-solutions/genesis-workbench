# Databricks notebook source
# Failure-finalize task. Triggered by the rapids-singlecell job's `tag_failed`
# task with run_if=AT_LEAST_ONE_FAILED so it only fires when the analyze task
# crashes. Mirrors the scanpy job's tag_run_failed notebook.

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
