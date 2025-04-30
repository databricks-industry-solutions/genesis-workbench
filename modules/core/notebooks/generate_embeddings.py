# Databricks notebook source
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("model_deployment_id", "11", "Model Deployment Id")
dbutils.widgets.text("source_data_location", "11", "Model Deployment Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
gwb_model_id = dbutils.widgets.get("gwb_model_id")
workload_type = dbutils.widgets.get("workload_type")
workload_size = dbutils.widgets.get("workload_size")
deploy_user = dbutils.widgets.get("deploy_user")

