import streamlit as st
import pandas as pd
import json
from datetime import timedelta

from genesis_workbench.workbench import get_workflow_job_status
from utils.streamlit_helper import get_user_info

def format_duration(duration: timedelta) -> str:
    total_seconds = int(duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"

# --- Utility: Get Workflow Runs ---
def get_workflow_runs_df(tag_key: str = "application", tag_value: str = "genesis_workbench", days_back: int = 7) -> pd.DataFrame:
    user_info = get_user_info()
    creator = user_info.user_name if user_info else None

    job_status_dict = get_workflow_job_status(
        tag_key=tag_key,
        tag_value=tag_value,
        days_back=days_back,
        creator_filter=creator
    )

    records = []
    for job_name, job_data in job_status_dict.items():
        for run in job_data.get("runs", []):
            start_time = pd.to_datetime(run["start_time"], unit="ms") if run.get("start_time") else None
            end_time = pd.to_datetime(run["end_time"], unit="ms") if run.get("end_time") else None
            duration = end_time - start_time if start_time and end_time else None

            record = {
                "Job Name": job_name,
                "Job ID": job_data["job_id"],
                "Tags": json.dumps(job_data.get("tags", {})),
                "Run ID": run["run_id"],
                "Lifecycle State": run["state"].value if run.get("state") else None,
                "Result State": run["result_state"].value if run.get("result_state") else None,
                "Start Time": start_time,
                "End Time": end_time,
                "Duration": format_duration(duration) if duration else "In Progress",
                "Created By": run.get("creator_user_name"),
            }
            records.append(record)
    return pd.DataFrame(records)

# --- Status Indicator Logic ---
def combine_status(lifecycle, result):
    if lifecycle == "RUNNING":
        return "🟡 Running"
    if result == "SUCCESS":
        return "🟢 Success"
    if result == "FAILED":
        return "🔴 Failed"
    if lifecycle == "PENDING":
        return "🟡 Pending"
    if lifecycle == "TERMINATED" and result is None:
        return "⚪️ Terminated"
    return f"⚪️ {result or lifecycle}"

# --- Main UI ---
st.title(":material/monitoring: Monitoring and Alerts")

dashboard_tab, alerts_tab = st.tabs(["Dashboard", "Alerts"])

# --- Dashboard Tab ---
with dashboard_tab:
    st.subheader("Workflow Runs")

    with st.container():
        col1, col2 = st.columns([10, 2])
        with col1:
            # Reduced spacing between label and pills
            st.markdown(
                "<div style='font-weight: 600; margin-bottom: -10px;'>Lookback Period</div>",
                unsafe_allow_html=True
            )
            days_back = st.pills(
                label="",
                options=[7, 15, 30],
                format_func=lambda x: f"{x} Days",
                default=7  # Optional: set a default selection
            )
        with col2:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("Refresh", use_container_width=True):
                st.session_state["workflow_runs_df"] = get_workflow_runs_df(days_back=days_back)

    # Load data
    with st.spinner("Fetching workflow runs..."):
        if "workflow_runs_df" not in st.session_state:
            st.session_state["workflow_runs_df"] = get_workflow_runs_df(days_back=days_back)
        df = st.session_state["workflow_runs_df"]

    if not df.empty:
        df["Status"] = df.apply(
            lambda row: combine_status(row["Lifecycle State"], row["Result State"]), axis=1
        )

        selected_job = st.selectbox("Select a job to filter:", options=df["Job Name"].unique())
        filtered_df = df[df["Job Name"] == selected_job]

        st.dataframe(
            filtered_df[[
                "Run ID", "Job Name", "Status", "Start Time", "End Time", "Duration", "Created By"
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