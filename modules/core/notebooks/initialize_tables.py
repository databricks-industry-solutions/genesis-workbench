# Databricks notebook source
#parameters to the notebook
catalog = dbutils.widgets.text("catalog", "")
schema = dbutils.widgets.text("schema", "")


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
    model_uc_name STRING,
    model_uc_version STRING,
    model_uc_added_by STRING,
    model_uc_added_date TIMESTAMP,
    is_model_deployed BOOLEAN,
    model_deployed_date TIMESTAMP,
    model_deployed_by STRING,
    model_deploy_platform STRING, -- modelserving, dcs etc
    model_invoke_url STRING
)
""")
