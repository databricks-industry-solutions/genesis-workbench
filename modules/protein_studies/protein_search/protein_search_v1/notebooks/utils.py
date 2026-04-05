# Databricks notebook source
# DBTITLE 1,Protein Search UC Configuration

SCHEMA_NAME = "ai_driven_drug_discovery"
ENDPOINT_NAME = "az_openai_gpt4o"

# COMMAND ----------

# DBTITLE 1,Setup UC paths from job parameters

def setup_uc_paths(print_endpoint: bool = False, silent: bool = True) -> dict:
    """
    Setup Unity Catalog resources using DAB job parameters.

    Reads catalog, schema, volume_name, and external_endpoint_name from
    job parameters (passed via dbutils.widgets). The volume_name is
    provisioned by the DAB volumes resource and passed as a job parameter.

    Returns:
        Dictionary with catalog_name, schema_name, volume_name,
        external_endpoint_name, volume_location, schema_path, volume_path
    """
    catalog_name = dbutils.widgets.get("catalog")
    schema_name = dbutils.widgets.get("schema") if _widget_exists("schema") else SCHEMA_NAME
    volume_name = dbutils.widgets.get("volume_name")
    external_endpoint_name = (
        dbutils.widgets.get("external_endpoint_name")
        if _widget_exists("external_endpoint_name")
        else ENDPOINT_NAME
    )

    if not catalog_name:
        raise ValueError("Job parameter 'catalog' is required but was empty.")
    if not volume_name:
        raise ValueError("Job parameter 'volume_name' is required but was empty.")

    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}")

    volume_location = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"
    schema_path = f"{catalog_name}.{schema_name}"
    volume_path = f"{catalog_name}.{schema_name}.{volume_name}"

    uc_config = {
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "volume_name": volume_name,
        "external_endpoint_name": external_endpoint_name,
        "volume_location": volume_location,
        "schema_path": schema_path,
        "volume_path": volume_path,
    }

    if not silent:
        print("=" * 70)
        print("UC Paths Configured (from job parameters)")
        print("=" * 70)
        for key, value in uc_config.items():
            if key == "external_endpoint_name" and not print_endpoint:
                continue
            print(f"{key}: {value}")
        print("=" * 70)

    return uc_config


def _widget_exists(name: str) -> bool:
    """Check whether a widget/job parameter exists."""
    try:
        dbutils.widgets.get(name)
        return True
    except Exception:
        return False
