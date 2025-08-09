# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.61.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#parameters to the notebook
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")
dbutils.widgets.text("deploy_model_job_id", "1234", "Deploy Model Job ID")
dbutils.widgets.text("bionemo_esm_finetune_job_id", "1234", "BioNeMo ESM Fine Tune Job ID")
dbutils.widgets.text("bionemo_esm_inference_job_id", "1234", "BioNeMo ESM Inference Job ID")
dbutils.widgets.text("admin_usage_dashboard_id", "1234", "ID of usage dashboard")
dbutils.widgets.text("application_secret_scope", "dbx_genesis_workbench", "Secret Scope used by application")
dbutils.widgets.text("databricks_app_name", "dev-scn-genesis-workbench", "UI Application name")
dbutils.widgets.text("dev_user_prefix", "abc", "Prefix for resources")


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
deploy_model_job_id = dbutils.widgets.get("deploy_model_job_id")
bionemo_esm_finetune_job_id = dbutils.widgets.get("bionemo_esm_finetune_job_id")
bionemo_esm_inference_job_id = dbutils.widgets.get("bionemo_esm_inference_job_id")
admin_usage_dashboard_id = dbutils.widgets.get("admin_usage_dashboard_id")
secret_scope = dbutils.widgets.get("application_secret_scope")
databricks_app_name = dbutils.widgets.get("databricks_app_name")
dev_user_prefix = dbutils.widgets.get("dev_user_prefix")
dev_user_prefix = None if dev_user_prefix.strip() == "" or dev_user_prefix.strip().lower()=="none" else dev_user_prefix

# COMMAND ----------

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

assert catalog and schema, "Catalog and schema must be provided"

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS models")

spark.sql(f"""
CREATE TABLE models (
    model_id BIGINT ,          
    model_name STRING,
    model_display_name STRING,
    model_source_version STRING,
    model_origin STRING, --uc, huggingface, pypi, bionemo, etc
    model_description_url STRING, --website to find more details about model
    model_category STRING, -- feature to which model is mapped
    model_uc_name STRING,
    model_uc_version STRING,
    model_owner STRING,
    model_added_by STRING,
    model_added_date TIMESTAMP,    
    model_input_schema STRING,
    model_output_schema STRING,
    model_params_schema STRING,
    is_model_deployed BOOLEAN,
    deployment_ids STRING,
    is_active BOOLEAN,
    deactivated_timestamp TIMESTAMP
)
""")

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS model_deployments")

spark.sql(f"""
CREATE TABLE model_deployments (
    deployment_id BIGINT,
    deployment_name STRING,
    deployment_description STRING,    
    model_id BIGINT,
    input_adapter STRING,
    output_adapter STRING,
    is_adapter BOOLEAN,
    deploy_model_uc_name STRING,
    deploy_model_uc_version STRING,
    model_deployed_date TIMESTAMP,
    model_deployed_by STRING,
    model_deploy_platform STRING, -- modelserving, dcs etc
    model_endpoint_name STRING,
    model_invoke_url STRING,
    is_active BOOLEAN,
    deactivated_timestamp TIMESTAMP
)
""")


# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS bionemo_weights")

spark.sql(f"""
CREATE TABLE bionemo_weights (
    ft_id BIGINT ,
    ft_label STRING,
    model_type STRING,
    variant STRING,
    experiment_name STRING,
    run_id STRING,
    weights_volume_location STRING,
    created_by STRING,
    created_datetime TIMESTAMP,
    is_active BOOLEAN,
    deactivated_timestamp TIMESTAMP
)
""")

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS settings")

spark.sql(f"""
CREATE TABLE settings (
    key STRING,
    value STRING,
    module STRING
)
""")

# COMMAND ----------

query= f"""
    INSERT INTO settings VALUES
    ('admin_usage_dashboard_id', '{admin_usage_dashboard_id}', 'core'),
    ('databricks_app_name', '{databricks_app_name}','core'),    
    ('deploy_model_job_id', '{deploy_model_job_id}', 'core'),
    ('secret_scope', '{secret_scope}', 'core')
"""

if dev_user_prefix:
    query = query + f", ('dev_user_prefix', '{dev_user_prefix}') "

spark.sql(query)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS user_settings")

spark.sql(f"""
CREATE TABLE user_settings (
    user_email STRING,
    key STRING,
    value STRING
)
""")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
app = w.apps.get(name=databricks_app_name)

# COMMAND ----------

spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog} TO `{app.service_principal_client_id}`")
spark.sql(f"GRANT ALL PRIVILEGES ON SCHEMA {catalog}.{schema} TO `{app.service_principal_client_id}`")

# COMMAND -----------
#Granting dashboard access to the app service principal
# import requests

# db_host = spark.conf.get("spark.databricks.workspaceUrl")
# db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# payload = {
#     "access_control_list": [
#         {
#             "user_name": app.service_principal_client_id,            
#             "permission_level": "CAN_VIEW"    
#         }
#     ]
# }

# headers = {
#     "Authorization": f"Bearer {db_token}",
#     "Content-Type": "application/json"
# }

# response = requests.put(
#     f"https://{db_host}/api/2.0/permissions/dashboards/{dashboard_id}",
#     json=payload,
#     headers=headers
# )

# if response.ok:
#     print("Permissions updated.")
# else:
#     print(f"Error: {response.status_code} - {response.text}")