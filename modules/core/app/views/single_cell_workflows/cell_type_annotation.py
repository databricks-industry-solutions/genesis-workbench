"""Single Cell — Cell Type Annotation tab via SCimilarity endpoints."""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.streamlit_helper import get_user_info
from utils.single_cell_analysis import (download_singlecell_markers_df, search_singlecell_runs)
from utils.scimilarity_tools import annotate_clusters


def render():
    st.markdown("##### Cell Type Annotation")
    st.markdown("Annotate clusters from a completed processing run using SCimilarity's 23M-cell reference database.")

    user_info = get_user_info()

    # Use a separate session key so annotation tab only shows scanpy/rapids runs
    if "singlecell_processing_runs_df" not in st.session_state:
        with st.spinner("Loading available runs..."):
            try:
                # Fetch scanpy and rapids runs separately, then combine
                scanpy_runs = search_singlecell_runs(user_email=user_info.user_email, processing_mode="scanpy")
                rapids_runs = search_singlecell_runs(user_email=user_info.user_email, processing_mode="rapids-singlecell")
                runs_df = pd.concat([scanpy_runs, rapids_runs], ignore_index=True)
                st.session_state["singlecell_processing_runs_df"] = runs_df
            except Exception as e:
                st.error(f"Error loading runs: {e}")
                return

    runs_df = st.session_state.get("singlecell_processing_runs_df", pd.DataFrame())
    if runs_df.empty:
        st.info("No completed processing runs found. Run an analysis first in the Raw Single Cell Processing tab.")
        return

    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["display_name"] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['experiment']}) - {row['start_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['start_time'], 'strftime') else row['start_time']}",
        axis=1,
    )

    col1, col2, col3 = st.columns([5, 3, 2], vertical_alignment="bottom")
    with col1:
        selected_display = st.selectbox("Select a completed processing run:", list(runs_df["display_name"]), key="annotation_run_select")
    with col2:
        cells_per_cluster = st.number_input("Cells per cluster:", min_value=3, max_value=50, value=10, step=5, key="annotation_cpc")
    with col3:
        k_neighbors = st.number_input("Neighbors (k):", min_value=5, max_value=200, value=20, step=5, key="annotation_k")

    run_id = dict(zip(runs_df["display_name"], runs_df["run_id"]))[selected_display]
    annotate_btn = st.button("Annotate Clusters", type="primary", key="annotate_btn")

    annotation_cache_key = f"annotation_{run_id}"

    if annotate_btn:
        with st.spinner("Loading data from MLflow..."):
            try:
                markers_df = download_singlecell_markers_df(run_id)
            except Exception as e:
                st.error(f"Failed to load run data: {e}")
                return

        cluster_col = None
        for c in ["cluster", "leiden", "louvain"]:
            if c in markers_df.columns:
                cluster_col = c
                break
        if not cluster_col:
            st.error("No cluster column found in the data.")
            return

        status_container = st.container()
        with status_container:
            progress = st.progress(0, text="Starting annotation...")
            spinner = st.empty()

        with spinner, st.spinner("Running annotation..."):
            try:
                results_df = annotate_clusters(
                    markers_df, cluster_col=cluster_col,
                    cells_per_cluster=cells_per_cluster, k_neighbors=k_neighbors,
                    progress_callback=lambda pct, text: progress.progress(min(pct, 100), text=text),
                )
                st.session_state[annotation_cache_key] = results_df
                st.session_state[f"annotation_markers_{run_id}"] = markers_df
                st.session_state[f"annotation_cluster_col_{run_id}"] = cluster_col
            except Exception as e:
                st.error(f"Annotation failed: {e}")
                return

    if annotation_cache_key in st.session_state:
        results_df = st.session_state[annotation_cache_key]
        st.markdown("##### Annotation Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        markers_key = f"annotation_markers_{run_id}"
        cluster_col_key = f"annotation_cluster_col_{run_id}"
        if markers_key in st.session_state and "UMAP_0" in st.session_state[markers_key].columns:
            markers_df = st.session_state[markers_key]
            cluster_col = st.session_state.get(cluster_col_key, "cluster")
            cluster_to_type = dict(zip(results_df["Cluster"].astype(str), results_df["Predicted Cell Type"]))
            markers_df["Predicted Cell Type"] = markers_df[cluster_col].astype(str).map(cluster_to_type).fillna("Unknown")

            fig_anno = px.scatter(markers_df, x="UMAP_0", y="UMAP_1", color="Predicted Cell Type",
                                 title="UMAP — Predicted Cell Types", height=550, template="plotly_dark")
            fig_anno.update_traces(marker=dict(size=3, opacity=0.7))
            fig_anno.update_layout(legend=dict(font=dict(size=11)))
            st.plotly_chart(fig_anno, use_container_width=True)
