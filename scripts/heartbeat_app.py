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
VS_ENDPOINT_NAME = "gwb_sequence_search_vs_endpoint"
VS_INDEXES = [
    "mmt_gwb.genesis_workbench.sequence_embedding_index",
]
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

# Vector Search endpoint + index pings (sweep protection):
# Same incident class as the warehouse sweep — VS endpoints are workspace-level
# lifecycle resources with NO custom_tags API in this Databricks version, so
# RemoveAfter tagging isn't an option. Ping with a get() to refresh activity.
# Index is a UC entity and IS tagged with RemoveAfter, but ping it too for safety.
try:
    ep = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
    print(f"VS endpoint ping: {VS_ENDPOINT_NAME}  state={ep.endpoint_status}")
except Exception as e:
    print(f"WARN: VS endpoint ping failed: {e}")

for idx_name in VS_INDEXES:
    try:
        idx = w.vector_search_indexes.get_index(idx_name)
        # Get just the readiness flag — minimal touch
        ready = bool(idx.status.ready) if idx.status else None
        print(f"VS index ping: {idx_name}  ready={ready}")
    except Exception as e:
        print(f"WARN: VS index ping failed for {idx_name}: {e}")

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
