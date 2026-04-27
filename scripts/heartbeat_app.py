# Databricks notebook source
# ---------------------------------------------------------------------------
# Heartbeat for the gwb-mmt-demo app on e2-demo-field-eng.
#
# Why: FE workspace policy auto-stops running apps not updated in >3 days, then
# auto-deletes stopped apps not updated in >7 days. This notebook refreshes
# the app's update_time by calling apps.update, preventing reaping.
#
# Schedule: runs M/W/F 9am PT (every ~2 days, under the 3-day threshold).
# Owner: may.merkletan@databricks.com
# Added as part of mmt/e2fe_gwb_deploy branch, 2026-04-21.
# ---------------------------------------------------------------------------

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from datetime import datetime, timezone

APP_NAME = "gwb-mmt-demo"
WAREHOUSE_NAME = "mmt_gwb_warehouse"  # resolve by name so sweep+recreate doesn't break this
REMOVE_AFTER = "2027-12-31"

w = WorkspaceClient()
now = datetime.now(timezone.utc).isoformat()

# Warehouse ping (sweep protection): refresh last_query_time on the GWB warehouse.
# A previous sweep deleted the warehouse despite a RemoveAfter tag — adding an
# explicit query keeps the sweeper's "idle warehouse" criteria from triggering.
try:
    target = next((wh for wh in w.warehouses.list() if wh.name == WAREHOUSE_NAME), None)
    if target is None:
        print(f"WARN: warehouse {WAREHOUSE_NAME} not found — sweeper may have struck again")
    else:
        result = w.statement_execution.execute_statement(
            warehouse_id=target.id,
            statement="SELECT 1 AS heartbeat_ping",
            wait_timeout="10s",
        )
        print(f"Warehouse ping: id={target.id}  state={result.status.state}")
except Exception as e:
    # Don't let warehouse ping failures block the app PATCH (which is the
    # critical sweep-protection for the app itself).
    print(f"WARN: warehouse ping failed: {e}")

new_desc = (
    f"MMT Genesis Workbench demo | RemoveAfter: {REMOVE_AFTER} | "
    f"heartbeat: {now}"
)

# Use raw REST call (PATCH /api/2.0/apps/{name}) — avoids SDK version skew on
# serverless runtime where apps.update() kwargs differ between versions.
response = w.api_client.do(
    "PATCH",
    f"/api/2.0/apps/{APP_NAME}",
    body={"description": new_desc},
)

print(f"Heartbeat fired at {now} for app {APP_NAME}")
print(f"Response update_time: {response.get('update_time')}")
print(f"Response description: {response.get('description')}")
