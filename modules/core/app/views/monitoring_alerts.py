import streamlit as st
import pandas as pd
import json
from genesis_workbench.workbench import get_workflow_job_status

st.set_page_config(page_title="Workflow Monitor", layout="wide")

# --- Utility: Get Workflow Runs ---
def get_workflow_runs_df(tag_key: str = "dev", tag_value: str = "guanyu_chen", max_runs: int = 5) -> pd.DataFrame:
    job_status_dict = get_workflow_job_status(tag_key=tag_key, tag_value=tag_value, max_runs=max_runs)
    records = []
    for job_name, job_data in job_status_dict.items():
        for run in job_data.get("runs", []):
            record = {
                "job_name": job_name,
                "job_id": job_data["job_id"],
                "job_tags": json.dumps(job_data.get("tags", {})),
                "run_id": run["run_id"],
                "state": run["state"].value if run.get("state") else None,
                "result_state": run["result_state"].value if run.get("result_state") else None,
                "start_time": pd.to_datetime(run["start_time"], unit="ms") if run.get("start_time") else None,
                "end_time": pd.to_datetime(run["end_time"], unit="ms") if run.get("end_time") else None,
                "creator_user_name": run.get("creator_user_name"),
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
st.title(":material/monitoring: Monitoring and Alerts")

dashboard_tab, alerts_tab = st.tabs(["Dashboard","Alerts"])

# --- Dashboard Tab ---
with dashboard_tab:
    st.subheader("Workflow Runs")
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.caption("Showing recent runs for jobs.")
    with col2:
        if st.button("🔄 Refresh"):
            st.session_state["workflow_runs_df"] = get_workflow_runs_df()

    # Load data
    with st.spinner("Fetching workflow runs..."):
        if "workflow_runs_df" not in st.session_state:
            st.session_state["workflow_runs_df"] = get_workflow_runs_df()
        df = st.session_state["workflow_runs_df"]

    if not df.empty:
        # Add visual status columns
        df["status"] = df["result_state"].apply(lambda x: f"{status_color(x)} {x if x else 'UNKNOWN'}")

        # Display table
        selected_job = st.selectbox("Select a job:", options=df["job_name"].unique())
        filtered_df = df[df["job_name"] == selected_job]

        st.dataframe(
            filtered_df[[
                "run_id", "job_name", "state", "status", "start_time", "end_time", "creator_user_name"
            ]].sort_values(by="start_time", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Optional: details expander
        with st.expander("📄 View raw job run data"):
            st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No workflow runs found.")

# --- Alerts Tab ---
with alerts_tab:
    st.subheader("Alerts")
    st.info("🚧 Alerts functionality coming soon.")