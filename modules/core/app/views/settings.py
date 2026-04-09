import streamlit as st
import os
import base64
from datetime import datetime, timedelta
from utils.streamlit_helper import get_user_info
from genesis_workbench.workbench import execute_workflow, execute_select_query
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState

st.title(":material/settings: Settings")

general_tab, endpoint_tab, access_tab = st.tabs(["General", "Endpoint Management", "Access Management"])

with general_tab:
    col1, col2, col3 = st.columns([1,1,1])

    with col1:

        core_catalog_name = os.environ["CORE_CATALOG_NAME"]
        core_schema_name = os.environ["CORE_SCHEMA_NAME"]
        sql_warehouse_id = os.environ["SQL_WAREHOUSE"]

        st.text_input("Application Schema Location: ", f"{core_catalog_name}.{core_schema_name}")

        st.write(f"SQL Warehouse ID: {sql_warehouse_id}")

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

with endpoint_tab:
    st.subheader("Start All Endpoints")
    st.write("Start all deployed model serving endpoints and keep them alive for a selected duration. "
             "This launches a background job that periodically pings each endpoint with sample data to prevent scale-to-zero.")

    job_id = os.environ.get("START_ALL_ENDPOINTS_JOB_ID")

    # Check if a run is already in progress
    active_run = None
    if job_id:
        try:
            w = WorkspaceClient()
            for run in w.jobs.list_runs(job_id=int(job_id), limit=5):
                if run.state and run.state.life_cycle_state in (
                    RunLifeCycleState.PENDING,
                    RunLifeCycleState.RUNNING,
                    RunLifeCycleState.BLOCKED,
                ):
                    active_run = run
                    break
        except Exception:
            pass

    if active_run:
        # Calculate estimated end time from the run's parameters
        start_time = datetime.fromtimestamp(active_run.start_time / 1000) if active_run.start_time else None
        num_hours_param = None
        if active_run.overriding_parameters and active_run.overriding_parameters.job_parameters:
            params = active_run.overriding_parameters.job_parameters
            num_hours_param = params.get("num_hours")

        if not num_hours_param and active_run.job_parameters:
            num_hours_param = active_run.job_parameters.get("num_hours")

        estimated_end = None
        if start_time and num_hours_param:
            estimated_end = start_time + timedelta(hours=int(num_hours_param))

        st.info(
            f"A keep-alive job is already running (Run ID: {active_run.run_id}).\n\n"
            f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M') if start_time else 'Unknown'}\n\n"
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
