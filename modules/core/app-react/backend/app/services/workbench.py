"""Thin wrappers around the genesis_workbench library. The lib reaches into
os.environ for catalog/schema/warehouse/job ids — those are populated by
initialize_lib() at app startup."""
from __future__ import annotations

import logging

from genesis_workbench.workbench import (
    execute_select_query,
    execute_non_select_query,
    execute_workflow,
    get_deployed_modules,
    get_user_settings,
    get_workflow_job_status,
    initialize,
    save_user_settings,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


def initialize_lib() -> None:
    s = get_settings()
    logger.info(
        "Initializing genesis_workbench lib: catalog=%s schema=%s warehouse=%s",
        s.catalog,
        s.schema,
        s.warehouse_id,
    )
    initialize(
        core_catalog_name=s.catalog,
        core_schema_name=s.schema,
        sql_warehouse_id=s.warehouse_id,
    )


def get_app_setting(key: str) -> str | None:
    """Read a row from the `settings` Delta table. The genesis_workbench lib
    also exports these as env vars (uppercased) at startup, but a fresh
    direct read is more robust against initialize_lib() failures and
    config-cache timing issues."""
    s = get_settings()
    try:
        df = execute_select_query(
            f"SELECT value FROM {s.catalog}.{s.schema}.settings WHERE key = '{key}' LIMIT 1"
        )
        if df is None or df.empty:
            return None
        return str(df.iloc[0]["value"])
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read setting %s from settings table: %s", key, e)
        return None


__all__ = [
    "execute_select_query",
    "execute_non_select_query",
    "execute_workflow",
    "get_deployed_modules",
    "get_app_setting",
    "get_user_settings",
    "get_workflow_job_status",
    "initialize_lib",
    "save_user_settings",
]
