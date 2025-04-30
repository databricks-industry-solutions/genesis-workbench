# Databricks notebook source
#parameters to the notebook

dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("gwb_model_id", "11", "Model Id")
dbutils.widgets.text("deployment_name", "my_finetuned_deployment", "Deployment Name")
dbutils.widgets.text("deployment_description", "description", "Deployment Description")

dbutils.widgets.text("workload_type", "CPU", "Endpoint Workload Type")
dbutils.widgets.text("workload_size", "Medium", "Endpoint Workload Size")
dbutils.widgets.text("deploy_user", "a@b.com", "User Id")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
gwb_model_id = dbutils.widgets.get("gwb_model_id")
deployment_name = dbutils.widgets.get("deployment_name")
deployment_description = dbutils.widgets.get("deployment_description")
workload_type = dbutils.widgets.get("workload_type")
workload_size = dbutils.widgets.get("workload_size")
deploy_user = dbutils.widgets.get("deploy_user")


# COMMAND ----------

print(f"catalog: {catalog}")
print(f"schema: {schema}")
print(f"gwb_model_id: {gwb_model_id}")
print(f"deployment_name: {deployment_name}")
print(f"deployment_description: {deployment_description}")
print(f"workload_type: {workload_type}")
print(f"workload_size: {workload_size}")

# COMMAND ----------

import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
        ServedModelInputWorkloadSize,
        ServedModelInputWorkloadType,
        AutoCaptureConfigInput,
        ServingEndpointDetailed
    )
from databricks.sdk import errors
from datetime import datetime, timedelta

def deploy_model(catalog_name: str,
                 schema_name : str,
                 fq_model_uc_name : str,
                 model_version: int,
                 deployment_id: int,
                 workload_type: str,
                 workload_size:str) -> ServingEndpointDetailed:

    w = WorkspaceClient()

    model_name = fq_model_uc_name.split(".")[2]
    endpoint_name = f"gwb_{model_name}_{deployment_id}"
    scale_to_zero = True

    served_entities = [
        ServedEntityInput(
            entity_name=fq_model_uc_name,
            entity_version=model_version,
            name=model_name,
            workload_type=workload_type,
            workload_size=workload_size,
            scale_to_zero_enabled=scale_to_zero,
        )
    ]
    auto_capture_config = AutoCaptureConfigInput(
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name_prefix=f"{endpoint_name}_serving",
        enabled=True,
    )

    print(f"Creating endpoint: {endpoint_name}")
    
    endpoint_details = w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=served_entities,
            auto_capture_config=auto_capture_config
        ),
        timeout = timedelta(minutes=60) #wait upto an hour
    )

    return endpoint_details
        

# COMMAND ----------

#get model details
result_df = spark.sql(f"SELECT model_uc_name, model_uc_version, deployment_ids FROM {catalog}.{schema}.models WHERE model_id = {gwb_model_id}")
deploy_result = None

#create a unique deploy id
deploy_id = np.datetime64('now', 'ms').view('int64')
current_deployment_ids = []
if result_df.count() > 0:
   #gather model details
   model_uc_name, model_uc_version, deployment_ids = result_df.limit(1).toPandas().values[0]
   print(f"Model found: {model_uc_name}, {model_uc_version}, {deployment_ids}")
   current_deployment_ids = deployment_ids.split(",") if len(deployment_ids.strip()) > 0 else []
   current_deployment_ids.append(str(deploy_id))
   print(f"Creating new deployment id: {deploy_id}")
   model_uri = f"models:/{model_uc_name}/{model_uc_version}"
   
   #deploy the model to model serving endpoint
   deploy_result = deploy_model(catalog, schema, model_uc_name, model_uc_version, deploy_id, workload_type, workload_size)
else:
    print("No model found to deploy")

# COMMAND ----------

hostname = spark.conf.get("spark.databricks.workspaceUrl")
#update the model deployment table
spark.sql(f"""
    INSERT INTO {catalog}.{schema}.model_deployments(
        deployment_id,
        deployment_name,
        deployment_description,
        model_id,
        input_adapter,
        output_adapter,
        model_deployed_date,
        model_deployed_by,
        model_deploy_platform, -- modelserving, dcs etc
        model_invoke_url
    ) VALUES (
        {deploy_id},
        '{deployment_name}',
        '{deployment_description}'
        {gwb_model_id},
        ' ',
        ' ',
        CURRENT_TIMESTAMP(),
        '{deploy_user}',
        'model_serving',
        'https://{hostname}/serving-endpoints/{deploy_result.config.served_entities[0].entity_name}/invocations'
    )
""") 


# COMMAND ----------

#update model table
spark.sql(f"""
        UPDATE {catalog}.{schema}.models SET 
            is_model_deployed = true,
            deployment_ids = '{','.join(current_deployment_ids)}'
        WHERE model_id = {gwb_model_id}
""")
