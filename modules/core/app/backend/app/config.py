import os
from functools import lru_cache
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    catalog: str
    schema: str
    warehouse_id: str
    llm_endpoint_name: str | None
    app_name: str | None
    admin_usage_dashboard_id: str | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        catalog=os.environ["CORE_CATALOG_NAME"],
        schema=os.environ["CORE_SCHEMA_NAME"],
        warehouse_id=os.environ["SQL_WAREHOUSE"],
        llm_endpoint_name=os.environ.get("LLM_ENDPOINT_NAME"),
        app_name=os.environ.get("DATABRICKS_APP_NAME"),
        admin_usage_dashboard_id=os.environ.get("ADMIN_USAGE_DASHBOARD_ID"),
    )
