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


__all__ = [
    "execute_select_query",
    "execute_non_select_query",
    "execute_workflow",
    "get_deployed_modules",
    "get_user_settings",
    "get_workflow_job_status",
    "initialize_lib",
    "save_user_settings",
]
