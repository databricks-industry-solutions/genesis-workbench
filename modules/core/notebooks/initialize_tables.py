# Databricks notebook source
#parameters to the notebook
dbutils.widgets.text("catalog", "genesis_workbench", "Catalog")
dbutils.widgets.text("schema", "dev_srijit_nair_dbx_genesis_workbench_core", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")


# COMMAND ----------

#for testing
#catalog = "genesis_workbench"
#schema = "dev_scn_dbx_genesis_workbench_core"

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
    model_id BIGINT GENERATED ALWAYS AS IDENTITY,          
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
    ft_id BIGINT GENERATED ALWAYS AS IDENTITY,
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
