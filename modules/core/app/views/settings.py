import streamlit as st
import os
import base64
from datetime import datetime, timedelta
from utils.streamlit_helper import get_user_info
from genesis_workbench.workbench import execute_workflow, execute_select_query
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState

st.title(":material/settings: Settings")

general_tab, endpoint_tab, batch_tab, access_tab = st.tabs(["General", "Endpoint Management", "Batch Models", "Access Management"])

with general_tab:
    col1, col2, col3 = st.columns([1,1,1])

    with col1:

        core_catalog_name = os.environ["CORE_CATALOG_NAME"]
        core_schema_name = os.environ["CORE_SCHEMA_NAME"]
        sql_warehouse_id = os.environ["SQL_WAREHOUSE"]

        st.text_input("Application Schema Location: ", f"{core_catalog_name}.{core_schema_name}")

        st.write(f"SQL Warehouse ID: {sql_warehouse_id}")

    st.divider()
    st.markdown("##### Settings")

    try:
        with st.spinner("Loading settings..."):
            other_settings_df = execute_select_query(
                f"SELECT key, value, module FROM {core_catalog_name}.{core_schema_name}.settings "
                f"WHERE key NOT LIKE '%_job_id' ORDER BY module, key"
            )
        if not other_settings_df.empty:
            other_settings_df.columns = ["Setting", "Value", "Module"]
            other_settings_df["Setting"] = other_settings_df["Setting"].str.replace("_", " ").str.title()
            other_settings_df["Module"] = other_settings_df["Module"].str.replace("_", " ").str.title()
            st.dataframe(other_settings_df, use_container_width=True, hide_index=True)
        else:
            st.info("No additional settings found.")
    except Exception as e:
        st.warning(f"Could not load settings: {e}")

    st.divider()
    st.markdown("##### Registered Workflows")

    try:
        with st.spinner("Loading registered workflows..."):
            workflows_df = execute_select_query(
                f"SELECT key, value, module FROM {core_catalog_name}.{core_schema_name}.settings "
                f"WHERE key LIKE '%_job_id' ORDER BY module, key"
            )
        if not workflows_df.empty:
            workflows_df.columns = ["Workflow", "Job ID", "Module"]
            # Clean up workflow names for display
            workflows_df["Workflow"] = workflows_df["Workflow"].str.replace("_job_id", "").str.replace("_", " ").str.title()
            workflows_df["Module"] = workflows_df["Module"].str.replace("_", " ").str.title()
            st.dataframe(workflows_df, use_container_width=True, hide_index=True)
        else:
            st.info("No workflows registered yet. Deploy modules to register workflows.")
    except Exception as e:
        st.warning(f"Could not load registered workflows: {e}")

def _fetch_endpoint_statuses(catalog, schema):
    """Fetch endpoint statuses — called once and cached in session state."""
    endpoints_df = execute_select_query(
        f"SELECT deployment_name, model_endpoint_name, deploy_model_uc_name "
        f"FROM {catalog}.{schema}.model_deployments "
        f"WHERE is_active = true ORDER BY deployment_name"
    )
    if endpoints_df.empty:
        return endpoints_df

    w_ep = WorkspaceClient()
    statuses = []
    for _, row in endpoints_df.iterrows():
        ep_name = row["model_endpoint_name"]
        try:
            resp = w_ep.api_client.do("GET", f"/api/2.0/serving-endpoints/{ep_name}")
            state = resp.get("state", {})
            ready = state.get("ready", "UNKNOWN")
            config_update = state.get("config_update", "NOT_UPDATING")

            entity_state = ""
            entity_msg = ""
            config = resp.get("config", {})
            entities = config.get("served_entities", config.get("served_models", []))
            if entities:
                e_state = entities[0].get("state", {})
                entity_state = e_state.get("deployment", "")
                entity_msg = e_state.get("deployment_state_message", "")

            if ready == "NOT_READY" and config_update == "IN_PROGRESS":
                status = "🟡 Updating"
            elif ready == "NOT_READY" and config_update == "UPDATE_FAILED":
                status = "🔴 Update failed"
            elif ready == "NOT_READY":
                status = "⚪ Stopped"
            elif entity_state == "DEPLOYMENT_ABORTED":
                status = "🔴 Failed"
            elif entity_msg == "Scaled to zero":
                status = "⚪ Scaled to zero"
            elif "Scaling from zero" in entity_msg:
                status = "🟡 Starting"
            elif ready == "READY" and entity_msg == "":
                status = "🟢 Ready"
            elif ready == "READY":
                status = "🟢 Ready"
            else:
                status = f"⚪ {entity_msg or ready}"
        except Exception:
            status = "⚪ Not Found"
        statuses.append(status)
    endpoints_df["Status"] = statuses
    endpoints_df.columns = ["Deployment", "Endpoint", "Model", "Status"]
    return endpoints_df


with endpoint_tab:
    st.markdown("##### Deployed Endpoints")

    try:
        # Cache endpoint statuses in session state to avoid re-fetching on every rerun
        if "endpoint_statuses_df" not in st.session_state:
            with st.spinner("Loading endpoint statuses..."):
                st.session_state["endpoint_statuses_df"] = _fetch_endpoint_statuses(core_catalog_name, core_schema_name)

        endpoints_display = st.session_state["endpoint_statuses_df"]
        if not endpoints_display.empty:
            st.dataframe(endpoints_display, use_container_width=True, hide_index=True)
        else:
            st.info("No active endpoints deployed yet.")

        if st.button("Refresh Statuses", key="refresh_endpoint_statuses"):
            st.session_state.pop("endpoint_statuses_df", None)
            st.rerun()
    except Exception as e:
        st.warning(f"Could not load endpoints: {e}")

    st.divider()
    st.subheader("Start All Endpoints")
    st.write("Start all deployed model serving endpoints and keep them alive for a selected duration. "
             "This launches a background job that periodically pings each endpoint with sample data to prevent scale-to-zero.")

    job_id = os.environ.get("START_ALL_ENDPOINTS_JOB_ID")
    print(f"[Settings] START_ALL_ENDPOINTS_JOB_ID={job_id}")

    # Check if a run is already in progress
    active_run = None
    if job_id:
        try:
            w_jobs = WorkspaceClient()
            resp = w_jobs.api_client.do(
                "GET", "/api/2.1/jobs/runs/list",
                query={"job_id": str(job_id), "limit": "5"}
            )
            print(f"[Settings] Raw response keys: {list(resp.keys())}, full: {resp}")
            runs = resp.get("runs", [])
            print(f"[Settings] Active runs for job {job_id}: {len(runs)}")
            for run in runs:
                lifecycle = run.get("state", {}).get("life_cycle_state", "")
                print(f"[Settings] Run {run.get('run_id')}: lifecycle={lifecycle}")
                if lifecycle in ("PENDING", "RUNNING", "BLOCKED"):
                    active_run = run
                    break
        except Exception as e:
            print(f"[Settings] Error checking run status: {e}")
            st.warning(f"Could not check run status: {e}")

    if active_run:
        start_ms = active_run.get("start_time")
        start_time = datetime.fromtimestamp(start_ms / 1000) if start_ms else None

        # job_parameters is a list of {"name": ..., "value": ...} dicts
        num_hours_param = None
        for p in active_run.get("job_parameters", []):
            if p.get("name") == "num_hours":
                num_hours_param = p.get("value")
                break

        estimated_end = None
        if start_time and num_hours_param:
            estimated_end = start_time + timedelta(hours=int(num_hours_param))

        st.info(
            f"A keep-alive job is already running (Run ID: {active_run.get('run_id')}).\n\n"
            f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M') if start_time else 'Unknown'}\n\n"
            f"**Duration:** {num_hours_param} hour(s)\n\n"
            f"**Estimated end:** {estimated_end.strftime('%Y-%m-%d %H:%M') if estimated_end else 'Unknown'}"
        )
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            num_hours = st.selectbox("Keep alive duration (hours):", options=list(range(1, 13)), index=3)

            if st.button("Start All Endpoints", type="primary"):
                if not job_id:
                    st.error("Start All Endpoints job is not configured. Please redeploy the core module.")
                else:
                    try:
                        core_catalog_name = os.environ["CORE_CATALOG_NAME"]
                        core_schema_name = os.environ["CORE_SCHEMA_NAME"]
                        sql_warehouse_id = os.environ["SQL_WAREHOUSE"]

                        run_id = execute_workflow(int(job_id), {
                            "catalog": core_catalog_name,
                            "schema": core_schema_name,
                            "sql_warehouse_id": sql_warehouse_id,
                            "num_hours": str(num_hours)
                        })
                        st.success(f"Endpoints are starting! Job run ID: {run_id}")
                        st.warning(f"Endpoints will be kept alive for {num_hours} hour(s). "
                                   "You can monitor the job in the Monitoring tab.")
                    except Exception as e:
                        st.error(f"Failed to start endpoints: {str(e)}")

with batch_tab:
    st.markdown("##### Batch Models")
    st.caption("Models that run as Databricks Jobs (not real-time serving endpoints).")

    try:
        with st.spinner("Loading batch models..."):
            batch_df = execute_select_query(
                f"SELECT model_display_name, model_category, module, cluster_type, "
                f"job_name, job_id, model_added_by, model_added_date, is_active "
                f"FROM {core_catalog_name}.{core_schema_name}.batch_models "
                f"WHERE is_active = true ORDER BY module, model_display_name"
            )
        if not batch_df.empty:
            batch_df.columns = ["Model", "Category", "Module", "Cluster", "Job Name",
                                "Job ID", "Added By", "Added Date", "Active"]
            batch_df["Category"] = batch_df["Category"].str.replace("_", " ").str.title()
            batch_df["Module"] = batch_df["Module"].str.replace("_", " ").str.title()
            st.dataframe(batch_df[["Model", "Category", "Module", "Cluster", "Job Name", "Job ID"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No batch models registered yet. Deploy Parabricks, AlphaFold2, or BioNeMo to register batch models.")
    except Exception as e:
        st.warning(f"Could not load batch models: {e}")
