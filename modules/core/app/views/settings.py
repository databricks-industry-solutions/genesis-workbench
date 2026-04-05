import streamlit as st
import os
import base64
from utils.streamlit_helper import get_user_info
from genesis_workbench.workbench import execute_workflow

st.title(":material/settings: Settings")

general_tab, endpoint_tab, access_tab = st.tabs(["General", "Endpoint Management", "Access Management"])

with general_tab:
    col1, col2, col3 = st.columns([1,1,1])

    with col1:

        core_catalog_name = os.environ["CORE_CATALOG_NAME"]
        core_schema_name = os.environ["CORE_SCHEMA_NAME"]
        sql_warehouse_id = os.environ["SQL_WAREHOUSE"]

        st.text_input("Application Schema Location: ", f"{core_catalog_name}.{core_schema_name}")

        st.write(f"SQL Warehouse Host Name:{sql_warehouse_id}")

with endpoint_tab:
    st.subheader("Start All Endpoints")
    st.write("Start all deployed model serving endpoints and keep them alive for a selected duration. "
             "This launches a background job that periodically pings each endpoint with sample data to prevent scale-to-zero.")

    col1, col2 = st.columns([1, 2])

    with col1:
        num_hours = st.selectbox("Keep alive duration (hours):", options=list(range(1, 13)), index=3)

        if st.button("Start All Endpoints", type="primary"):
            job_id = os.environ.get("START_ALL_ENDPOINTS_JOB_ID")
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
