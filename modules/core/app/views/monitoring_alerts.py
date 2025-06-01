import streamlit as st
import pandas as pd
import json
from genesis_workbench.workbench import get_workflow_job_status
from utils.streamlit_helper import get_user_info

# --- Utility: Get Workflow Runs ---
def get_workflow_runs_df(tag_key: str = "dev", tag_value: str = "guanyu_chen", days_back: int = 7) -> pd.DataFrame:
    user_info = get_user_info()
    creator = user_info.get("user_name", None)

    job_status_dict = get_workflow_job_status(
        tag_key=tag_key,
        tag_value=tag_value,
        days_back=days_back,
        creator_filter=creator
    )

    records = []
    for job_name, job_data in job_status_dict.items():
        for run in job_data.get("runs", []):
            record = {
                "Job Name": job_name,
                "Job ID": job_data["job_id"],
                "Tags": json.dumps(job_data.get("tags", {})),
                "Run ID": run["run_id"],
                "Lifecycle State": run["state"].value if run.get("state") else None,
                "Result State": run["result_state"].value if run.get("result_state") else None,
                "Start Time": pd.to_datetime(run["start_time"], unit="ms") if run.get("start_time") else None,
                "End Time": pd.to_datetime(run["end_time"], unit="ms") if run.get("end_time") else None,
                "Created By": run.get("creator_user_name"),
            }
            records.append(record)
    return pd.DataFrame(records)

# --- Status Indicator Styling ---
def status_color(state):
    if state == "SUCCESS":
        return "🟢"
    elif state == "FAILED":
        return "🔴"
    elif state in ["RUNNING", "PENDING"]:
        return "🟡"
    return "⚪️"

# --- Main UI ---
st.title(":bar_chart: Monitoring and Alerts")

dashboard_tab, alerts_tab = st.tabs(["Dashboard", "Alerts"])

# --- Dashboard Tab ---
with dashboard_tab:
    st.subheader("Workflow Runs")

    # Days filter and refresh button
    col1, col2, col3 = st.columns([4, 2, 1])
    with col1:
        days_back = st.selectbox("Show runs from the last:", options=[7, 15, 30], index=0)
    with col2:
        st.caption("Filter based on current user and recent date range.")
    with col3:
        if st.button("Refresh"):
            st.session_state["workflow_runs_df"] = get_workflow_runs_df(days_back=days_back)

    # Load data
    with st.spinner("Fetching workflow runs..."):
        if "workflow_runs_df" not in st.session_state:
            st.session_state["workflow_runs_df"] = get_workflow_runs_df(days_back=days_back)
        df = st.session_state["workflow_runs_df"]

    if not df.empty:
        # Add visual status indicator
        df["Status"] = df["Result State"].apply(lambda x: f"{status_color(x)} {x if x else 'UNKNOWN'}")

        selected_job = st.selectbox("Select a job to filter:", options=df["Job Name"].unique())
        filtered_df = df[df["Job Name"] == selected_job]

        st.dataframe(
            filtered_df[[
                "Run ID", "Job Name", "Lifecycle State", "Status", "Start Time", "End Time", "Created By"
            ]].sort_values(by="Start Time", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No workflow runs found.")

# --- Alerts Tab ---
with alerts_tab:
    st.subheader("Alerts")
    st.info("Alerts functionality coming soon.")